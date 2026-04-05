import { randomUUID } from "node:crypto";
import type { PromptResponse, SessionNotification } from "@agentclientprotocol/sdk";
import type {
  Message,
  RawMessageStreamEvent,
  ThinkingBlock,
  TextBlock,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages";
import type { PromptTranslator } from "../../interfaces.js";
import {
  TOOL_USE_BRIDGE_END,
  TOOL_USE_BRIDGE_START,
  anthropicRequestToPromptRequest,
  type ProvisionalStreamUsage,
  parseBridgedToolUse,
  toAnthropicToolUseBlock,
} from "../../helpers/messages.js";
import type { FinalizedAnthropicTurn } from "../../types.js";

function newMessageId(): string {
  return `msg_${randomUUID().replace(/-/g, "")}`;
}

function mapStopReason(stopReason: PromptResponse["stopReason"]): Message["stop_reason"] {
  switch (stopReason) {
    case "end_turn":
      return "end_turn";
    case "max_tokens":
      return "max_tokens";
    case "refusal":
      return "refusal";
    case "cancelled":
    case "max_turn_requests":
    default:
      return "end_turn";
  }
}

class AnthropicStreamCollector {
  private readonly usage: Message["usage"];
  private readonly content: Message["content"] = [];
  private readonly streamEvents: RawMessageStreamEvent[] = [];
  private readonly messageId = newMessageId();
  private readonly toolUseId = `toolu_${randomUUID().replace(/-/g, "")}`;
  private activeTextBlockIndex: number | null = null;
  private activeThinkingBlockIndex: number | null = null;
  private bridgedToolUse: ReturnType<typeof parseBridgedToolUse> = null;
  private pendingText = "";
  private pendingBridge = "";
  private readonly toolCallTitles = new Map<string, string>();

  constructor(
    private readonly requestId: string,
    private readonly sessionId: string,
    private readonly model: string,
    private readonly enableToolBridge: boolean,
    private readonly includeProgressThinking: boolean,
    initialUsage: ProvisionalStreamUsage,
  ) {
    this.usage = {
      cache_creation: null,
      cache_creation_input_tokens: initialUsage.cache_creation_input_tokens,
      cache_read_input_tokens: initialUsage.cache_read_input_tokens,
      inference_geo: null,
      input_tokens: initialUsage.input_tokens,
      output_tokens: 0,
      server_tool_use: null,
      service_tier: null,
    };
  }

  start(): RawMessageStreamEvent {
    const event: RawMessageStreamEvent = {
      type: "message_start",
      message: {
        id: this.messageId,
        type: "message",
        container: null,
        role: "assistant",
        content: [],
        model: this.model,
        stop_reason: null,
        stop_sequence: null,
        stop_details: null,
        usage: { ...this.usage },
      },
    };
    this.streamEvents.push(event);
    return event;
  }

  private ensureTextBlockStarted(emitted: RawMessageStreamEvent[]) {
    if (this.activeTextBlockIndex !== null) {
      return;
    }

    this.closeThinkingBlock(emitted);

    this.activeTextBlockIndex = this.content.length;
    this.content.push({
      type: "text",
      text: "",
      citations: null,
    } as TextBlock);

    emitted.push({
      type: "content_block_start",
      index: this.activeTextBlockIndex,
      content_block: {
        type: "text",
        text: "",
        citations: null,
      },
    });
  }

  private emitTextDelta(text: string, emitted: RawMessageStreamEvent[]) {
    if (!text.length) {
      return;
    }

    this.ensureTextBlockStarted(emitted);

    const block = this.content[this.activeTextBlockIndex!];
    if (block.type === "text") {
      block.text += text;
    }

    emitted.push({
      type: "content_block_delta",
      index: this.activeTextBlockIndex!,
      delta: {
        type: "text_delta",
        text,
      },
    });
  }

  private ensureThinkingBlockStarted(emitted: RawMessageStreamEvent[]) {
    if (this.activeThinkingBlockIndex !== null || !this.includeProgressThinking) {
      return;
    }

    this.activeThinkingBlockIndex = this.content.length;
    this.content.push({
      type: "thinking",
      thinking: "",
      signature: "",
    } as ThinkingBlock);

    emitted.push({
      type: "content_block_start",
      index: this.activeThinkingBlockIndex,
      content_block: {
        type: "thinking",
        thinking: "",
        signature: "",
      },
    });
  }

  private emitThinkingDelta(text: string, emitted: RawMessageStreamEvent[]) {
    if (!this.includeProgressThinking || !text.length) {
      return;
    }

    this.ensureThinkingBlockStarted(emitted);
    const block = this.content[this.activeThinkingBlockIndex!];
    if (block.type === "thinking") {
      block.thinking += text;
    }

    emitted.push({
      type: "content_block_delta",
      index: this.activeThinkingBlockIndex!,
      delta: {
        type: "thinking_delta",
        thinking: text,
      },
    });
  }

  private closeThinkingBlock(emitted: RawMessageStreamEvent[]) {
    if (this.activeThinkingBlockIndex === null) {
      return;
    }

    emitted.push({
      type: "content_block_stop",
      index: this.activeThinkingBlockIndex,
    });
    this.activeThinkingBlockIndex = null;
  }

  private summarizeToolPayload(value: unknown): string {
    if (value === undefined || value === null) {
      return "";
    }

    if (typeof value === "string") {
      const normalized = value.trim().replace(/\s+/g, " ");
      if (!normalized.length) {
        return "";
      }
      return normalized.length > 180 ? `${normalized.slice(0, 177)}...` : normalized;
    }

    try {
      const serialized = JSON.stringify(value);
      if (serialized === "{}" || serialized === "[]") {
        return "";
      }
      return serialized.length > 180 ? `${serialized.slice(0, 177)}...` : serialized;
    } catch {
      return String(value);
    }
  }

  private summarizeToolLocations(
    locations:
      | Extract<SessionNotification["update"], { sessionUpdate: "tool_call" }>["locations"]
      | Extract<SessionNotification["update"], { sessionUpdate: "tool_call_update" }>["locations"],
  ): string {
    if (!Array.isArray(locations) || !locations.length) {
      return "";
    }

    const paths = locations
      .map((location) => location.path)
      .filter((path): path is string => typeof path === "string" && path.length > 0)
      .slice(0, 2);

    return paths.join(", ");
  }

  private formatToolCallStart(
    update: Extract<SessionNotification["update"], { sessionUpdate: "tool_call" }>,
  ): string {
    const title = update.title?.trim() || "Tool";
    this.toolCallTitles.set(update.toolCallId, title);
    const detail =
      this.summarizeToolPayload(update.rawInput) || this.summarizeToolLocations(update.locations);
    return detail ? `\nUsing ${title}: ${detail}\n` : `\nUsing ${title}\n`;
  }

  private formatToolCallUpdate(
    update: Extract<SessionNotification["update"], { sessionUpdate: "tool_call_update" }>,
  ): string | null {
    const title = update.title?.trim() || this.toolCallTitles.get(update.toolCallId) || "Tool";
    if (update.title?.trim()) {
      this.toolCallTitles.set(update.toolCallId, update.title.trim());
    }

    if (update.status === "completed") {
      const outputSummary =
        this.summarizeToolPayload(update.rawOutput) ||
        this.summarizeToolLocations(update.locations);
      return outputSummary ? `\nCompleted ${title}: ${outputSummary}\n` : `\nCompleted ${title}\n`;
    }

    if (update.status === "failed") {
      const outputSummary = this.summarizeToolPayload(update.rawOutput);
      return outputSummary ? `\nFailed ${title}: ${outputSummary}\n` : `\nFailed ${title}\n`;
    }

    if (update.status === "in_progress" && update.rawOutput !== undefined) {
      const outputSummary = this.summarizeToolPayload(update.rawOutput);
      return outputSummary ? `\n${title}: ${outputSummary}\n` : null;
    }

    return null;
  }

  private longestStartTokenSuffix(value: string): number {
    const maxLength = Math.min(value.length, TOOL_USE_BRIDGE_START.length - 1);
    for (let length = maxLength; length > 0; length -= 1) {
      if (TOOL_USE_BRIDGE_START.startsWith(value.slice(-length))) {
        return length;
      }
    }
    return 0;
  }

  private pushBridgeChunk(chunk: string, emitted: RawMessageStreamEvent[]) {
    this.pendingBridge += chunk;
    const endIndex = this.pendingBridge.indexOf(TOOL_USE_BRIDGE_END);
    if (endIndex < 0) {
      return;
    }

    const candidate = this.pendingBridge.slice(0, endIndex + TOOL_USE_BRIDGE_END.length);
    const trailing = this.pendingBridge.slice(endIndex + TOOL_USE_BRIDGE_END.length);
    const parsed = parseBridgedToolUse(candidate);

    if (parsed && !trailing.trim().length) {
      this.bridgedToolUse = parsed;
      this.pendingBridge = "";
      this.pendingText = "";
      return;
    }

    this.pendingBridge = "";
    this.emitTextDelta(candidate + trailing, emitted);
  }

  private pushToolAwareText(text: string): RawMessageStreamEvent[] {
    const emitted: RawMessageStreamEvent[] = [];
    if (!text.length || this.bridgedToolUse) {
      return emitted;
    }

    if (this.pendingBridge.length) {
      this.pushBridgeChunk(text, emitted);
      this.streamEvents.push(...emitted);
      return emitted;
    }

    const combined = this.pendingText + text;
    const startIndex = combined.indexOf(TOOL_USE_BRIDGE_START);
    if (startIndex >= 0) {
      this.pendingText = "";
      this.emitTextDelta(combined.slice(0, startIndex), emitted);
      this.pushBridgeChunk(combined.slice(startIndex), emitted);
      this.streamEvents.push(...emitted);
      return emitted;
    }

    const suffixLength = this.longestStartTokenSuffix(combined);
    const flushLength = combined.length - suffixLength;
    this.pendingText = combined.slice(flushLength);
    this.emitTextDelta(combined.slice(0, flushLength), emitted);
    this.streamEvents.push(...emitted);
    return emitted;
  }

  private flushPendingText(emitted: RawMessageStreamEvent[]) {
    if (this.pendingBridge.length) {
      this.pendingText += this.pendingBridge;
      this.pendingBridge = "";
    }

    if (!this.pendingText.length) {
      return;
    }

    this.emitTextDelta(this.pendingText, emitted);
    this.pendingText = "";
  }

  pushNotification(notification: SessionNotification): RawMessageStreamEvent[] {
    const emitted: RawMessageStreamEvent[] = [];
    const update = notification.update;

    if (update.sessionUpdate === "agent_thought_chunk" && update.content.type === "text") {
      this.emitThinkingDelta(update.content.text, emitted);
      this.streamEvents.push(...emitted);
      return emitted;
    }

    if (update.sessionUpdate === "tool_call") {
      this.emitThinkingDelta(this.formatToolCallStart(update), emitted);
      this.streamEvents.push(...emitted);
      return emitted;
    }

    if (update.sessionUpdate === "tool_call_update") {
      const updateText = this.formatToolCallUpdate(update);
      if (updateText) {
        this.emitThinkingDelta(updateText, emitted);
      }
      this.streamEvents.push(...emitted);
      return emitted;
    }

    if (update.sessionUpdate !== "agent_message_chunk") {
      return emitted;
    }

    if (update.content.type === "text") {
      if (this.enableToolBridge) {
        return this.pushToolAwareText(update.content.text);
      }

      if (!update.content.text.length) {
        this.streamEvents.push(...emitted);
        return emitted;
      }

      this.emitTextDelta(update.content.text, emitted);
    }

    this.streamEvents.push(...emitted);
    return emitted;
  }

  finish(response: PromptResponse): FinalizedAnthropicTurn {
    this.usage.cache_creation_input_tokens = response.usage?.cachedWriteTokens ?? null;
    this.usage.cache_read_input_tokens = response.usage?.cachedReadTokens ?? null;
    this.usage.input_tokens = response.usage?.inputTokens ?? 0;
    this.usage.output_tokens = response.usage?.outputTokens ?? 0;

    const emitted: RawMessageStreamEvent[] = [];
    if (this.enableToolBridge && !this.bridgedToolUse) {
      this.flushPendingText(emitted);
      this.streamEvents.push(...emitted);
    }

    if (this.bridgedToolUse) {
      this.closeThinkingBlock(this.streamEvents);
      if (this.activeTextBlockIndex !== null) {
        this.streamEvents.push({
          type: "content_block_stop",
          index: this.activeTextBlockIndex,
        });
        this.activeTextBlockIndex = null;
      }

      const block = toAnthropicToolUseBlock(this.bridgedToolUse, this.toolUseId) as ToolUseBlock;
      const blockIndex = this.content.length;
      this.content.push(block);
      this.streamEvents.push({
        type: "content_block_start",
        index: blockIndex,
        content_block: block,
      });
      this.streamEvents.push({
        type: "content_block_stop",
        index: blockIndex,
      });
    } else {
      this.closeThinkingBlock(this.streamEvents);
      if (this.activeTextBlockIndex !== null) {
        this.streamEvents.push({
          type: "content_block_stop",
          index: this.activeTextBlockIndex,
        });
        this.activeTextBlockIndex = null;
      }
    }
    const stopReason = this.bridgedToolUse ? "tool_use" : mapStopReason(response.stopReason);
    this.streamEvents.push({
      type: "message_delta",
      delta: {
        stop_reason: stopReason,
        stop_sequence: null,
        stop_details: null,
        container: null,
      },
      usage: {
        cache_creation_input_tokens: this.usage.cache_creation_input_tokens,
        cache_read_input_tokens: this.usage.cache_read_input_tokens,
        input_tokens: this.usage.input_tokens,
        output_tokens: this.usage.output_tokens,
        server_tool_use: null,
      },
    });
    this.streamEvents.push({ type: "message_stop" });

    const message: Message = {
      id: this.messageId,
      type: "message",
      container: null,
      role: "assistant",
      content: this.content,
      model: this.model,
      stop_reason: stopReason,
      stop_sequence: null,
      stop_details: null,
      usage: { ...this.usage },
    };

    return {
      requestId: this.requestId,
      sessionId: this.sessionId,
      streamEvents: [...this.streamEvents],
      message,
    };
  }
}

export class AnthropicPromptTranslator implements PromptTranslator {
  toPromptRequest(
    sessionId: string,
    request: Parameters<typeof anthropicRequestToPromptRequest>[1],
  ) {
    return anthropicRequestToPromptRequest(sessionId, request);
  }

  createStreamCollector(args: {
    requestId: string;
    sessionId: string;
    model: string;
    enableToolBridge: boolean;
    includeProgressThinking: boolean;
    initialUsage: ProvisionalStreamUsage;
  }) {
    return new AnthropicStreamCollector(
      args.requestId,
      args.sessionId,
      args.model,
      args.enableToolBridge,
      args.includeProgressThinking,
      args.initialUsage,
    );
  }

  fromPromptResult(args: {
    requestId: string;
    sessionId: string;
    model: string;
    enableToolBridge: boolean;
    initialUsage: ProvisionalStreamUsage;
    response: PromptResponse;
    notifications: SessionNotification[];
  }) {
    const collector = this.createStreamCollector({
      requestId: args.requestId,
      sessionId: args.sessionId,
      model: args.model,
      enableToolBridge: args.enableToolBridge,
      includeProgressThinking: false,
      initialUsage: args.initialUsage,
    });
    collector.start();
    for (const notification of args.notifications) {
      collector.pushNotification(notification);
    }
    return collector.finish(args.response);
  }
}

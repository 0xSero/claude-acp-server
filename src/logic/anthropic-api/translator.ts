import { randomUUID } from "node:crypto";
import type { PromptResponse, SessionNotification } from "@agentclientprotocol/sdk";
import type {
  Message,
  RawMessageStreamEvent,
  TextBlock,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages";
import type { PromptTranslator } from "../../interfaces.js";
import {
  TOOL_USE_BRIDGE_END,
  TOOL_USE_BRIDGE_START,
  anthropicRequestToPromptRequest,
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
  private readonly usage: Message["usage"] = {
    cache_creation: null,
    cache_creation_input_tokens: null,
    cache_read_input_tokens: null,
    inference_geo: null,
    input_tokens: 0,
    output_tokens: 0,
    server_tool_use: null,
    service_tier: null,
  };
  private readonly content: Message["content"] = [];
  private readonly streamEvents: RawMessageStreamEvent[] = [];
  private readonly messageId = newMessageId();
  private readonly toolUseId = `toolu_${randomUUID().replace(/-/g, "")}`;
  private activeTextBlockIndex: number | null = null;
  private bridgedToolUse: ReturnType<typeof parseBridgedToolUse> = null;
  private pendingText = "";
  private pendingBridge = "";

  constructor(
    private readonly requestId: string,
    private readonly sessionId: string,
    private readonly model: string,
    private readonly enableToolBridge: boolean,
  ) {}

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
    } else if (this.activeTextBlockIndex !== null) {
      this.streamEvents.push({
        type: "content_block_stop",
        index: this.activeTextBlockIndex,
      });
      this.activeTextBlockIndex = null;
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
  }) {
    return new AnthropicStreamCollector(
      args.requestId,
      args.sessionId,
      args.model,
      args.enableToolBridge,
    );
  }

  fromPromptResult(args: {
    requestId: string;
    sessionId: string;
    model: string;
    enableToolBridge: boolean;
    response: PromptResponse;
    notifications: SessionNotification[];
  }) {
    const collector = this.createStreamCollector({
      requestId: args.requestId,
      sessionId: args.sessionId,
      model: args.model,
      enableToolBridge: args.enableToolBridge,
    });
    collector.start();
    for (const notification of args.notifications) {
      collector.pushNotification(notification);
    }
    return collector.finish(args.response);
  }
}

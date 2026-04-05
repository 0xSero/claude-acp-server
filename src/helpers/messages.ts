import type { ContentBlock, PromptRequest } from "@agentclientprotocol/sdk";
import type {
  ContentBlockParam,
  MessageCreateParamsBase,
  MessageParam,
  ToolChoice,
  ToolUseBlockParam,
  ToolUnion,
} from "@anthropic-ai/sdk/resources/messages/messages";
import { HttpError } from "./errors.js";

export const TOOL_USE_BRIDGE_START = "<anthropic_tool_use>";
export const TOOL_USE_BRIDGE_END = "</anthropic_tool_use>";

export type BridgedToolUse = {
  name: string;
  input: unknown;
};

export type ProvisionalStreamUsage = {
  input_tokens: number;
  cache_creation_input_tokens: number | null;
  cache_read_input_tokens: number | null;
};

function collectRequestTextFragments(request: MessageCreateParamsBase): string[] {
  const fragments: string[] = [];

  if (typeof request.system === "string") {
    fragments.push(request.system);
  } else if (Array.isArray(request.system)) {
    for (const block of request.system) {
      if ("text" in block && typeof block.text === "string") {
        fragments.push(block.text);
      }
    }
  }

  for (const message of request.messages) {
    if (typeof message.content === "string") {
      fragments.push(message.content);
      continue;
    }

    for (const block of message.content) {
      if (block.type === "text") {
        fragments.push(block.text);
      }
    }
  }

  return fragments;
}

function normalizeMessageContent(content: MessageParam["content"]): ContentBlockParam[] {
  if (typeof content === "string") {
    return [{ type: "text", text: content }];
  }
  return content;
}

function renderContentBlockForTranscript(block: ContentBlockParam): string {
  switch (block.type) {
    case "text":
      return block.text;
    case "image":
      return `[image:${block.source.type === "url" ? block.source.url : block.source.media_type}]`;
    case "tool_result":
      return `[tool_result:${block.tool_use_id}]`;
    case "tool_use":
      return `[tool_use:${block.name}:${block.id}] ${JSON.stringify(block.input)}`;
    case "document":
      return "[document]";
    default:
      return `[${block.type}]`;
  }
}

function messageToTranscriptLine(message: MessageParam): string {
  const rendered = normalizeMessageContent(message.content)
    .map((block) => renderContentBlockForTranscript(block))
    .join("\n");

  return `${message.role.toUpperCase()}:\n${rendered}`.trim();
}

function normalizeClientTools(request: MessageCreateParamsBase): ToolUnion[] {
  if (!Array.isArray(request.tools)) {
    return [];
  }

  return request.tools.filter((tool): tool is ToolUnion => {
    return typeof tool === "object" && tool !== null && "name" in tool;
  });
}

function estimateTokensFromText(text: string): number {
  const trimmed = text.trim();
  if (!trimmed.length) {
    return 0;
  }

  // Fast heuristic for start-of-stream compatibility when upstream ACP usage
  // is only available at turn completion.
  return Math.max(1, Math.ceil(trimmed.length / 4));
}

function renderSystemForEstimate(system: MessageCreateParamsBase["system"]): string {
  if (typeof system === "string") {
    return system;
  }

  if (!Array.isArray(system)) {
    return "";
  }

  return system.map((block) => ("text" in block ? block.text : "[system block]")).join("\n");
}

export function estimateProvisionalStreamUsage(args: {
  request: MessageCreateParamsBase;
  hasPriorSession: boolean;
}): ProvisionalStreamUsage {
  const fragments: string[] = [];
  const { request } = args;

  const systemText = renderSystemForEstimate(request.system);
  if (systemText.trim().length) {
    fragments.push(systemText);
  }

  for (const message of request.messages) {
    fragments.push(messageToTranscriptLine(message));
  }

  for (const tool of normalizeClientTools(request)) {
    const descriptor: Record<string, unknown> = { name: tool.name };
    if ("description" in tool && typeof tool.description === "string") {
      descriptor.description = tool.description;
    }
    if ("input_schema" in tool) {
      descriptor.input_schema = tool.input_schema;
    }
    fragments.push(JSON.stringify(descriptor));
  }

  const estimatedTotal = estimateTokensFromText(fragments.join("\n\n"));
  if (!estimatedTotal) {
    return {
      input_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    };
  }

  const uncached = Math.max(1, Math.min(16, estimatedTotal));
  const cached = Math.max(0, estimatedTotal - uncached);

  return {
    input_tokens: uncached,
    cache_creation_input_tokens: args.hasPriorSession ? 0 : cached,
    cache_read_input_tokens: args.hasPriorSession ? cached : 0,
  };
}

export function inferWorkingDirectoryFromRequest(
  request: MessageCreateParamsBase,
): string | undefined {
  for (const fragment of collectRequestTextFragments(request)) {
    const pwdMatch = fragment.match(/(?:^|\n)% pwd\s*\n([^\n]+)/);
    if (pwdMatch?.[1]?.startsWith("/")) {
      return pwdMatch[1].trim();
    }

    const folderMatch = fragment.match(/Current folder:\s*(\/[^\n]+)/);
    if (folderMatch?.[1]?.startsWith("/")) {
      return folderMatch[1].trim();
    }
  }

  return undefined;
}

function renderToolChoice(choice: ToolChoice | undefined): string | null {
  if (!choice) {
    return null;
  }

  switch (choice.type) {
    case "auto":
    case "any":
    case "none":
      return choice.type;
    case "tool":
      return `tool:${choice.name}`;
    default:
      return null;
  }
}

function buildClientToolBridgeContext(
  tools: ToolUnion[],
  toolChoice: ToolChoice | undefined,
): string | null {
  if (!tools.length) {
    return null;
  }

  const lines = [
    "Anthropic client tools were provided with this request.",
    "These tools are available on the northbound Anthropic client side, not through ACP directly.",
    "If prior transcript messages include tool_use and tool_result blocks, treat those as real completed client-side tool interactions.",
  ];

  const renderedToolChoice = renderToolChoice(toolChoice);
  if (renderedToolChoice) {
    lines.push(`Requested tool_choice: ${renderedToolChoice}`);
  }

  lines.push(
    "If you need the client to execute one of these tools now, respond with exactly the following wrapper and no other text:",
    TOOL_USE_BRIDGE_START,
    '{"name":"<tool_name>","input":{}}',
    TOOL_USE_BRIDGE_END,
    "Only request one tool call in this format.",
    "If no client-side tool is needed, answer normally.",
    "",
    "Available client tools:",
  );

  for (const tool of tools) {
    lines.push(`- ${tool.name}`);
    if ("description" in tool && typeof tool.description === "string" && tool.description.trim()) {
      lines.push(`  description: ${tool.description.trim()}`);
    }
    if ("input_schema" in tool && tool.input_schema !== undefined) {
      lines.push(`  input_schema: ${JSON.stringify(tool.input_schema)}`);
    }
  }

  return lines.join("\n");
}

export function hasClientTools(request: MessageCreateParamsBase): boolean {
  return normalizeClientTools(request).length > 0;
}

export function shouldEnableToolBridge(request: MessageCreateParamsBase): boolean {
  if (!hasClientTools(request)) {
    return false;
  }

  return request.tool_choice?.type !== "none";
}

export function parseBridgedToolUse(text: string): BridgedToolUse | null {
  const match = text
    .trim()
    .match(
      new RegExp(
        `^${TOOL_USE_BRIDGE_START.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*([\\s\\S]+?)\\s*${TOOL_USE_BRIDGE_END.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`,
      ),
    );

  if (!match) {
    return null;
  }

  try {
    const parsed = JSON.parse(match[1]) as Partial<BridgedToolUse>;
    if (typeof parsed.name !== "string" || !parsed.name.trim()) {
      return null;
    }
    return {
      name: parsed.name.trim(),
      input: parsed.input ?? {},
    };
  } catch {
    return null;
  }
}

export function toAnthropicToolUseBlock(
  bridged: BridgedToolUse,
  id: string,
): ToolUseBlockParam & { caller: { type: "direct" } } {
  return {
    type: "tool_use",
    id,
    name: bridged.name,
    input: bridged.input,
    caller: { type: "direct" },
  };
}

function contentBlockToAcp(block: ContentBlockParam): ContentBlock[] {
  switch (block.type) {
    case "text":
      return [{ type: "text", text: block.text }];
    case "image":
      if (block.source.type === "base64") {
        return [
          {
            type: "image",
            data: block.source.data,
            mimeType: block.source.media_type,
          },
        ];
      }

      return [
        {
          type: "text",
          text: `[image:${block.source.url}]`,
        },
      ];
    case "document":
      if (block.source.type === "text") {
        return [
          {
            type: "resource",
            resource: {
              uri:
                block.source.media_type === "text/plain"
                  ? "anthropic://document.txt"
                  : "anthropic://document",
              text: block.source.data,
            },
          },
        ];
      }

      return [{ type: "text", text: "[document omitted]" }];
    case "tool_result":
      return [
        {
          type: "text",
          text: `[tool_result:${block.tool_use_id}] ${typeof block.content === "string" ? block.content : JSON.stringify(block.content)}`,
        },
      ];
    case "tool_use":
      return [
        {
          type: "text",
          text: `[tool_use:${block.name}:${block.id}] ${JSON.stringify(block.input)}`,
        },
      ];
    default:
      return [{ type: "text", text: renderContentBlockForTranscript(block) }];
  }
}

export function anthropicRequestToPromptRequest(
  sessionId: string,
  request: MessageCreateParamsBase,
): PromptRequest {
  if (!request.messages.length) {
    throw new HttpError({
      status: 400,
      type: "invalid_request_error",
      message: "messages must contain at least one entry.",
    });
  }

  const finalMessage = request.messages[request.messages.length - 1];
  if (finalMessage.role !== "user") {
    throw new HttpError({
      status: 400,
      type: "invalid_request_error",
      message: "The final message must have role=user when using ACP session continuity.",
    });
  }

  const prompt: ContentBlock[] = [];
  const contextLines: string[] = [];
  const tools = normalizeClientTools(request);

  if (typeof request.system === "string" && request.system.trim()) {
    contextLines.push(`SYSTEM:\n${request.system.trim()}`);
  } else if (Array.isArray(request.system) && request.system.length) {
    const rendered = request.system
      .map((block) => ("text" in block ? block.text : "[system block]"))
      .join("\n");
    contextLines.push(`SYSTEM:\n${rendered}`);
  }

  const toolBridgeContext = buildClientToolBridgeContext(tools, request.tool_choice);
  if (toolBridgeContext) {
    contextLines.push(toolBridgeContext);
  }

  for (const message of request.messages.slice(0, -1)) {
    contextLines.push(messageToTranscriptLine(message));
  }

  if (contextLines.length) {
    prompt.push({
      type: "text",
      text:
        "Conversation context for this ACP session bootstrap:\n\n" +
        `${contextLines.join("\n\n")}\n\n` +
        "Use the transcript as prior context. The next blocks are the current user turn.",
    });
  }

  for (const block of normalizeMessageContent(finalMessage.content)) {
    prompt.push(...contentBlockToAcp(block));
  }

  return {
    sessionId,
    prompt,
  };
}

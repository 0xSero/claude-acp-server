import type {
  ClientCapabilities,
  CreateTerminalRequest,
  InitializeResponse,
  NewSessionResponse,
  PromptRequest,
  PromptResponse,
  SessionId,
  SessionNotification,
} from "@agentclientprotocol/sdk";
import type {
  Message,
  MessageCreateParamsBase,
  RawMessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages/messages";
import type { ChildProcessByStdio } from "node:child_process";
import type { Readable } from "node:stream";
import type { AddressInfo } from "node:net";

export type PermissionPolicy = "allow_once" | "allow_always" | "reject_once" | "reject_always";

export type BackendCommandConfig = {
  command: string;
  args: string[];
  cwd: string;
  env: Record<string, string | undefined>;
};

export type ServerConfig = {
  port: number;
  host: string;
  apiKey?: string;
  anthropicVersion: string;
  traceRequests: boolean;
  sessionHeader: string;
  requestIdHeader: string;
  backend: BackendCommandConfig;
  sessionCwd: string;
  cwdHeader: string;
  permissionPolicy: PermissionPolicy;
  permissionMode?: string;
  terminalOutputByteLimit: number;
};

export type BackendSession = {
  sessionId: SessionId;
  cwd: string;
  models?: NewSessionResponse["models"];
};

export type TerminalRecord = {
  request: CreateTerminalRequest;
  process: ChildProcessByStdio<null, Readable, Readable>;
  output: string;
  truncated: boolean;
  exitCode: number | null;
  signal: string | null;
  waitForExit: Promise<void>;
  released: boolean;
};

export type PromptExecutionOptions = {
  sessionId: SessionId;
  request: PromptRequest;
  onNotification: (notification: SessionNotification) => void | Promise<void>;
  signal?: AbortSignal;
};

export type PromptExecutionResult = {
  response: PromptResponse;
  notifications: SessionNotification[];
};

export type FinalizedAnthropicTurn = {
  message: Message;
  streamEvents: RawMessageStreamEvent[];
  sessionId: string;
  requestId: string;
};

export type PromptAggregationState = {
  requestId: string;
  sessionId: string;
  model: string;
  content: Message["content"];
  streamEvents: RawMessageStreamEvent[];
  usage: Message["usage"];
  stopReason: Message["stop_reason"];
};

export type NormalizedAnthropicRequest = MessageCreateParamsBase & {
  stream?: boolean;
};

export type HttpErrorPayload = {
  status: number;
  type:
    | "authentication_error"
    | "invalid_request_error"
    | "not_found_error"
    | "rate_limit_error"
    | "api_error";
  message: string;
  details?: unknown;
};

export type RunningServer = {
  address: AddressInfo;
  close: () => Promise<void>;
};

export type BackendRuntime = {
  initialized: InitializeResponse;
  capabilities: ClientCapabilities;
};

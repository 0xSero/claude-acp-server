import type {
  NewSessionResponse,
  PromptRequest,
  PromptResponse,
  SessionId,
  SessionNotification,
} from "@agentclientprotocol/sdk";
import type { MessageCreateParamsBase } from "@anthropic-ai/sdk/resources/messages/messages";
import type { ModelInfo } from "@anthropic-ai/sdk/resources/models";
import type { RawMessageStreamEvent } from "@anthropic-ai/sdk/resources/messages/messages";
import type { FinalizedAnthropicTurn, PromptExecutionOptions, RunningServer } from "./types.js";

export interface Logger {
  log: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
}

export interface BackendManager {
  initialize(): Promise<void>;
  ensureSession(sessionId: string | undefined): Promise<NewSessionResponse>;
  setSessionModel(sessionId: SessionId, model: string): Promise<void>;
  prompt(options: PromptExecutionOptions): Promise<PromptResponse>;
  listModels(): Promise<ModelInfo[]>;
  close(): Promise<void>;
}

export interface AnthropicFacade {
  handleMessages(
    headers: Headers,
    body: MessageCreateParamsBase & { stream?: boolean },
    signal?: AbortSignal,
    streamObserver?: {
      onReady: (meta: { sessionId: string; requestId: string }) => void | Promise<void>;
      onEvent: (event: RawMessageStreamEvent) => void | Promise<void>;
    },
  ): Promise<FinalizedAnthropicTurn>;
  listModels(headers: Headers): Promise<ModelInfo[]>;
}

export interface FacadeHttpServer {
  listen(): Promise<RunningServer>;
  close(): Promise<void>;
}

export interface PromptTranslator {
  toPromptRequest(sessionId: string, request: MessageCreateParamsBase): PromptRequest;
  createStreamCollector(args: {
    requestId: string;
    sessionId: string;
    model: string;
    enableToolBridge: boolean;
  }): {
    start: () => RawMessageStreamEvent;
    pushNotification: (notification: SessionNotification) => RawMessageStreamEvent[];
    finish: (response: PromptResponse) => FinalizedAnthropicTurn;
  };
  fromPromptResult(args: {
    requestId: string;
    sessionId: string;
    model: string;
    enableToolBridge: boolean;
    response: PromptResponse;
    notifications: SessionNotification[];
  }): FinalizedAnthropicTurn;
}

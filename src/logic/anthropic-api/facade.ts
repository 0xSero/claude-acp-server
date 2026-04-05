import { randomUUID } from "node:crypto";
import type {
  AnthropicFacade,
  BackendManager,
  Logger,
  PromptTranslator,
} from "../../interfaces.js";
import type {
  MessageCreateParamsBase,
  RawMessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages/messages";
import { HttpError, requireAnthropicHeaders } from "../../helpers/errors.js";
import {
  estimateProvisionalStreamUsage,
  inferWorkingDirectoryFromRequest,
  shouldEnableToolBridge,
} from "../../helpers/messages.js";
import type { FinalizedAnthropicTurn, ServerConfig } from "../../types.js";
import type { SessionNotification } from "@agentclientprotocol/sdk";

const MODEL_ALIASES: Record<string, string> = {
  "claude-sonnet-4-6": "sonnet",
  "claude-sonnet-4.6": "sonnet",
  "sonnet-4-6": "sonnet",
  "sonnet-4.6": "sonnet",
  sonnet: "sonnet",
  "claude-opus-4-6": "default",
  "claude-opus-4.6": "default",
  "opus-4-6": "default",
  "opus-4.6": "default",
  opus: "default",
};

export class AnthropicAcpFacade implements AnthropicFacade {
  constructor(
    private readonly backend: BackendManager,
    private readonly translator: PromptTranslator,
    private readonly config: ServerConfig,
    private readonly logger: Logger = console,
  ) {}

  async handleMessages(
    headers: Headers,
    body: MessageCreateParamsBase & { stream?: boolean },
    signal?: AbortSignal,
    streamObserver?: {
      onReady: (meta: { sessionId: string; requestId: string }) => void | Promise<void>;
      onEvent: (event: RawMessageStreamEvent) => void | Promise<void>;
    },
  ): Promise<FinalizedAnthropicTurn> {
    requireAnthropicHeaders(headers, this.config.anthropicVersion, this.config.apiKey);

    const requestedSessionId = headers.get(this.config.sessionHeader) ?? undefined;
    const requestedCwd =
      headers.get(this.config.cwdHeader) ?? inferWorkingDirectoryFromRequest(body) ?? undefined;
    const hasPriorSession = Boolean(requestedSessionId?.trim());
    const ensured = await this.backend.ensureSession(requestedSessionId, requestedCwd);
    const sessionId = ensured.sessionId;

    if (!hasPriorSession && this.config.permissionMode && ensured.modes?.availableModes?.length) {
      const targetMode = ensured.modes.availableModes.find(
        (m) => m.id === this.config.permissionMode,
      );
      if (targetMode) {
        try {
          await this.backend.setSessionMode(sessionId, targetMode.id);
        } catch (err) {
          this.logger.warn("[claude-acp-server] failed to set permission mode", err);
        }
      } else if (this.config.traceRequests) {
        this.logger.log("[claude-acp-server] requested mode not available", {
          requested: this.config.permissionMode,
          available: ensured.modes.availableModes.map((m) => m.id),
        });
      }
    }

    const requestedModel = body.model;
    const backendModel = MODEL_ALIASES[requestedModel] ?? requestedModel;
    const enableToolBridge = shouldEnableToolBridge(body);
    const initialUsage = estimateProvisionalStreamUsage({
      request: body,
      hasPriorSession,
    });

    if (ensured.models?.availableModels?.length) {
      const knownModelIds = new Set(ensured.models.availableModels.map((entry) => entry.modelId));
      if (!knownModelIds.has(backendModel)) {
        throw new HttpError({
          status: 400,
          type: "invalid_request_error",
          message: `Unknown model '${requestedModel}'. Available models: ${Array.from(knownModelIds).join(", ")}`,
        });
      }
    }

    await this.backend.setSessionModel(sessionId, backendModel);

    const promptRequest = this.translator.toPromptRequest(sessionId, body);
    const notifications: SessionNotification[] = [];
    const requestId = randomUUID();
    const collector = this.translator.createStreamCollector({
      requestId,
      sessionId,
      model: requestedModel,
      enableToolBridge,
      includeProgressThinking: Boolean(streamObserver),
      initialUsage,
    });
    let emittedStreamEventCount = 0;

    if (streamObserver) {
      await streamObserver.onReady({ sessionId, requestId });
      await streamObserver.onEvent(collector.start());
      emittedStreamEventCount += 1;
    }

    const response = await this.backend.prompt({
      sessionId,
      request: promptRequest,
      signal,
      onNotification: async (notification) => {
        if (this.config.traceRequests) {
          this.logger.log("[claude-acp-server] notification", {
            type: notification.update.sessionUpdate,
          });
        }
        notifications.push(notification);
        if (streamObserver) {
          const events = collector.pushNotification(notification);
          if (this.config.traceRequests && events.length) {
            this.logger.log("[claude-acp-server] emitting SSE events", {
              count: events.length,
              types: events.map((event) => event.type),
            });
          }
          emittedStreamEventCount += events.length;
          for (const event of events) {
            await streamObserver.onEvent(event);
          }
        }
      },
    });

    if (streamObserver) {
      const finalized = collector.finish(response);
      for (const event of finalized.streamEvents.slice(emittedStreamEventCount)) {
        await streamObserver.onEvent(event);
      }
      return finalized;
    }

    return this.translator.fromPromptResult({
      requestId,
      sessionId,
      model: requestedModel,
      enableToolBridge,
      initialUsage,
      response,
      notifications,
    });
  }

  async listModels(headers: Headers) {
    requireAnthropicHeaders(headers, this.config.anthropicVersion, this.config.apiKey);
    return this.backend.listModels();
  }
}

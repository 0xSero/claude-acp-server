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
import { shouldEnableToolBridge } from "../../helpers/messages.js";
import type { FinalizedAnthropicTurn, ServerConfig } from "../../types.js";
import type { SessionNotification } from "@agentclientprotocol/sdk";

const MODEL_ALIASES: Record<string, string> = {
  "claude-sonnet-4-6": "sonnet",
  "claude-opus-4-6": "default",
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

    const ensured = await this.backend.ensureSession(
      headers.get(this.config.sessionHeader) ?? undefined,
    );
    const sessionId = ensured.sessionId;
    const requestedModel = body.model;
    const backendModel = MODEL_ALIASES[requestedModel] ?? requestedModel;
    const enableToolBridge = shouldEnableToolBridge(body);

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
        notifications.push(notification);
        if (streamObserver) {
          const events = collector.pushNotification(notification);
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
      response,
      notifications,
    });
  }

  async listModels(headers: Headers) {
    requireAnthropicHeaders(headers, this.config.anthropicVersion, this.config.apiKey);
    return this.backend.listModels();
  }
}

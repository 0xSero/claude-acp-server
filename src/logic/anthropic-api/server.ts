import {
  createServer,
  type IncomingMessage,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";
import { randomUUID } from "node:crypto";
import type { AddressInfo } from "node:net";
import type { AnthropicFacade, FacadeHttpServer, Logger } from "../../interfaces.js";
import type { ServerConfig } from "../../types.js";
import { anthropicErrorBody, toHttpError } from "../../helpers/errors.js";
import { openSse, readJsonBody, writeJson, writeSseEvent } from "../../helpers/streams.js";

export class AnthropicHttpServer implements FacadeHttpServer {
  private readonly server: HttpServer;

  constructor(
    private readonly facade: AnthropicFacade,
    private readonly config: ServerConfig,
    private readonly logger: Logger = console,
  ) {
    this.server = createServer((request, response) => {
      void this.handle(request, response);
    });
  }

  async listen() {
    await new Promise<void>((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(this.config.port, this.config.host, () => {
        this.server.off("error", reject);
        resolve();
      });
    });

    return {
      address: this.server.address() as AddressInfo,
      close: () => this.close(),
    };
  }

  async close(): Promise<void> {
    if (!this.server.listening) {
      return;
    }

    this.server.closeAllConnections?.();
    await new Promise<void>((resolve, reject) => {
      this.server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      });
    });
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const requestId = randomUUID();
    response.setHeader(this.config.requestIdHeader, requestId);

    try {
      if (!request.url) {
        throw new Error("Missing request URL.");
      }

      const url = new URL(request.url, `http://${request.headers.host ?? "localhost"}`);
      const headerEntries: [string, string][] = [];
      for (const [key, value] of Object.entries(request.headers)) {
        if (value === undefined) {
          continue;
        }
        headerEntries.push([key, Array.isArray(value) ? value.join(", ") : value]);
      }
      const headers = new Headers(headerEntries);

      if (request.method === "GET" && url.pathname === "/v1/models") {
        const models = await this.facade.listModels(headers);
        writeJson(response, 200, {
          data: models,
          has_more: false,
          first_id: models[0]?.id ?? null,
          last_id: models.length ? models[models.length - 1].id : null,
        });
        return;
      }

      if (request.method === "POST" && url.pathname === "/v1/messages") {
        const abortController = new AbortController();
        request.on("aborted", () => abortController.abort());
        response.on("close", () => {
          if (!response.writableEnded) {
            abortController.abort();
          }
        });
        const body = await readJsonBody<any>(request);

        if (this.config.traceRequests) {
          this.logger.log("[claude-acp-server] request", {
            path: url.pathname,
            stream: Boolean(body?.stream),
            model: typeof body?.model === "string" ? body.model : null,
            messages: Array.isArray(body?.messages) ? body.messages.length : 0,
            tools: Array.isArray(body?.tools) ? body.tools.length : 0,
            userAgent: headers.get("user-agent"),
          });
        }

        if (body.stream) {
          let sseOpened = false;
          let heartbeat: ReturnType<typeof setInterval> | null = null;
          try {
            await this.facade.handleMessages(headers, body, abortController.signal, {
              onReady: ({ sessionId, requestId: streamRequestId }) => {
                response.setHeader(this.config.sessionHeader, sessionId);
                response.setHeader(this.config.requestIdHeader, streamRequestId);
                openSse(response);
                sseOpened = true;
                heartbeat = setInterval(() => {
                  if (!response.writableEnded) {
                    response.write(": heartbeat\n\n");
                  }
                }, 2000);
              },
              onEvent: (event) => {
                writeSseEvent(response, event.type, event);
              },
            });
          } finally {
            if (heartbeat) clearInterval(heartbeat);
          }
          if (!sseOpened) {
            openSse(response);
          }
          response.end();
          return;
        }

        const turn = await this.facade.handleMessages(headers, body, abortController.signal);

        response.setHeader(this.config.sessionHeader, turn.sessionId);

        writeJson(response, 200, turn.message);
        return;
      }

      writeJson(response, 404, {
        type: "error",
        error: {
          type: "not_found_error",
          message: `Unknown route: ${request.method} ${url.pathname}`,
        },
        request_id: requestId,
      });
    } catch (error) {
      const httpError = toHttpError(error);
      this.logger.error("[claude-acp-server] request failed", error);
      if (response.headersSent) {
        response.end();
        return;
      }
      writeJson(response, httpError.status, anthropicErrorBody(httpError, requestId));
    }
  }
}

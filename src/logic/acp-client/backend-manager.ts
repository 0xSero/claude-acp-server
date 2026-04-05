import {
  ClientSideConnection,
  RequestError,
  ndJsonStream,
  type Client,
  type CreateTerminalRequest,
  type CreateTerminalResponse,
  type KillTerminalRequest,
  type KillTerminalResponse,
  type NewSessionResponse,
  type PromptRequest,
  type PromptResponse,
  type ReadTextFileRequest,
  type ReadTextFileResponse,
  type ReleaseTerminalRequest,
  type ReleaseTerminalResponse,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
  type SessionId,
  type SessionNotification,
  type TerminalOutputRequest,
  type TerminalOutputResponse,
  type WaitForTerminalExitRequest,
  type WaitForTerminalExitResponse,
  type WriteTextFileRequest,
  type WriteTextFileResponse,
} from "@agentclientprotocol/sdk";
import { spawn, type ChildProcessByStdio } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import type { Readable, Writable } from "node:stream";
import type { BackendManager, Logger } from "../../interfaces.js";
import type { BackendRuntime, ServerConfig } from "../../types.js";
import { nodeToWebReadable, nodeToWebWritable } from "../../helpers/streams.js";
import { ACP_CLIENT_CAPABILITIES } from "../../configs.js";
import { TerminalManager } from "./terminal-manager.js";

type PromptListener = (notification: SessionNotification) => void | Promise<void>;

export class AcpBackendManager implements BackendManager, Client {
  private child: ChildProcessByStdio<Writable, Readable, null> | null = null;
  private connection: ClientSideConnection | null = null;
  private runtime: BackendRuntime | null = null;
  private readonly promptListeners = new Map<SessionId, Set<PromptListener>>();
  private readonly sessionLocks = new Map<SessionId, Promise<void>>();
  private readonly terminalManager: TerminalManager;
  private modelsCache: NewSessionResponse["models"] | null = null;

  constructor(
    private readonly config: ServerConfig,
    private readonly logger: Logger = console,
  ) {
    this.terminalManager = new TerminalManager(logger, config.terminalOutputByteLimit);
  }

  async initialize(): Promise<void> {
    await this.ensureStarted();
  }

  async ensureSession(sessionId: string | undefined, cwd?: string): Promise<NewSessionResponse> {
    const connection = await this.ensureConnection();
    const sessionCwd = cwd || this.config.sessionCwd;

    if (!sessionId) {
      const created = await connection.newSession({
        cwd: sessionCwd,
        mcpServers: [],
      });
      this.modelsCache = created.models ?? this.modelsCache;
      return created;
    }

    const resumed = await connection.unstable_resumeSession({
      sessionId,
      cwd: sessionCwd,
      mcpServers: [],
    });
    return {
      sessionId: resumed.sessionId,
      modes: resumed.modes,
      models: resumed.models,
      configOptions: resumed.configOptions,
    };
  }

  async setSessionMode(sessionId: SessionId, modeId: string): Promise<void> {
    const connection = await this.ensureConnection();
    await connection.setSessionMode({ sessionId, modeId });
  }

  async setSessionModel(sessionId: SessionId, model: string): Promise<void> {
    const connection = await this.ensureConnection();
    await connection.unstable_setSessionModel({
      sessionId,
      modelId: model,
    });
  }

  async prompt(options: {
    sessionId: SessionId;
    request: PromptRequest;
    onNotification: PromptListener;
    signal?: AbortSignal;
  }): Promise<PromptResponse> {
    return this.withSessionLock(options.sessionId, async () => {
      const connection = await this.ensureConnection();
      const listeners = this.promptListeners.get(options.sessionId) ?? new Set<PromptListener>();
      listeners.add(options.onNotification);
      this.promptListeners.set(options.sessionId, listeners);

      const abortHandler = () => {
        void connection.cancel({ sessionId: options.sessionId });
      };

      options.signal?.addEventListener("abort", abortHandler, { once: true });

      try {
        return await connection.prompt(options.request);
      } finally {
        options.signal?.removeEventListener("abort", abortHandler);
        listeners.delete(options.onNotification);
        if (!listeners.size) {
          this.promptListeners.delete(options.sessionId);
        }
      }
    });
  }

  async listModels() {
    if (this.modelsCache?.availableModels?.length) {
      return this.modelsCache.availableModels.map((model) => ({
        id: model.modelId,
        type: "model" as const,
        display_name: model.name,
        created_at: new Date(0).toISOString(),
        max_input_tokens: null,
        max_tokens: null,
        capabilities: null,
      }));
    }

    const session = await this.ensureSession(undefined);
    this.modelsCache = session.models ?? null;

    return (this.modelsCache?.availableModels ?? []).map((model) => ({
      id: model.modelId,
      type: "model" as const,
      display_name: model.name,
      created_at: new Date(0).toISOString(),
      max_input_tokens: null,
      max_tokens: null,
      capabilities: null,
    }));
  }

  async close(): Promise<void> {
    await this.terminalManager.close();
    if (this.child) {
      const child = this.child;
      const exited = new Promise<void>((resolve) => {
        child.once("exit", () => resolve());
      });
      if (!child.killed) {
        child.kill("SIGTERM");
      }
      child.stdin.end();
      child.stdout.destroy();
      await Promise.race([exited, new Promise<void>((resolve) => setTimeout(resolve, 250))]);
    }
    this.child = null;
    this.connection = null;
    this.runtime = null;
  }

  async requestPermission(request: RequestPermissionRequest): Promise<RequestPermissionResponse> {
    const selected = request.options.find((option) => option.kind === this.config.permissionPolicy);

    if (!selected) {
      return { outcome: { outcome: "cancelled" } };
    }

    return {
      outcome: {
        outcome: "selected",
        optionId: selected.optionId,
      },
    };
  }

  async sessionUpdate(notification: SessionNotification): Promise<void> {
    const listeners = this.promptListeners.get(notification.sessionId);
    if (!listeners?.size) {
      return;
    }

    for (const listener of listeners) {
      await listener(notification);
    }
  }

  async writeTextFile(request: WriteTextFileRequest): Promise<WriteTextFileResponse> {
    this.ensureAbsolutePath(request.path);
    await fs.mkdir(path.dirname(request.path), { recursive: true });
    await fs.writeFile(request.path, request.content, "utf8");
    return {};
  }

  async readTextFile(request: ReadTextFileRequest): Promise<ReadTextFileResponse> {
    this.ensureAbsolutePath(request.path);

    let content: string;
    try {
      content = await fs.readFile(request.path, "utf8");
    } catch (error) {
      if ((error as { code?: string }).code === "ENOENT") {
        throw RequestError.resourceNotFound(request.path);
      }
      throw error;
    }

    if (!request.line && !request.limit) {
      return { content };
    }

    const lines = content.split("\n");
    const start = Math.max((request.line ?? 1) - 1, 0);
    const end = request.limit ? start + request.limit : lines.length;
    return {
      content: lines.slice(start, end).join("\n"),
    };
  }

  async createTerminal(request: CreateTerminalRequest): Promise<CreateTerminalResponse> {
    return this.terminalManager.create(request);
  }

  async terminalOutput(request: TerminalOutputRequest): Promise<TerminalOutputResponse> {
    return this.terminalManager.output(request);
  }

  async waitForTerminalExit(
    request: WaitForTerminalExitRequest,
  ): Promise<WaitForTerminalExitResponse> {
    return this.terminalManager.waitForExit(request);
  }

  async killTerminal(request: KillTerminalRequest): Promise<KillTerminalResponse> {
    return this.terminalManager.kill(request);
  }

  async releaseTerminal(request: ReleaseTerminalRequest): Promise<ReleaseTerminalResponse | void> {
    return this.terminalManager.release(request);
  }

  private async ensureStarted(): Promise<BackendRuntime> {
    if (this.connection && this.runtime) {
      return this.runtime;
    }

    const child = spawn(this.config.backend.command, this.config.backend.args, {
      cwd: this.config.backend.cwd,
      env: this.config.backend.env,
      stdio: ["pipe", "pipe", "inherit"],
    }) as ChildProcessByStdio<Writable, Readable, null>;

    child.on("exit", (code, signal) => {
      this.logger.warn(
        `[claude-acp-server] backend exited`,
        JSON.stringify({ code, signal, command: this.config.backend.command }),
      );
      this.child = null;
      this.connection = null;
      this.runtime = null;
      this.modelsCache = null;
    });

    const stream = ndJsonStream(nodeToWebWritable(child.stdin), nodeToWebReadable(child.stdout));
    const connection = new ClientSideConnection(() => this, stream);
    const initialized = await connection.initialize({
      protocolVersion: 1,
      clientCapabilities: ACP_CLIENT_CAPABILITIES,
    });

    this.child = child;
    this.connection = connection;
    this.runtime = {
      initialized,
      capabilities: ACP_CLIENT_CAPABILITIES,
    };

    return this.runtime;
  }

  private async ensureConnection(): Promise<ClientSideConnection> {
    await this.ensureStarted();
    if (!this.connection) {
      throw new Error("ACP backend connection is not available.");
    }
    return this.connection;
  }

  private async withSessionLock<T>(sessionId: SessionId, work: () => Promise<T>): Promise<T> {
    const previous = this.sessionLocks.get(sessionId) ?? Promise.resolve();
    let release!: () => void;
    const next = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.sessionLocks.set(
      sessionId,
      previous.then(() => next),
    );

    await previous;
    try {
      return await work();
    } finally {
      release();
      if (this.sessionLocks.get(sessionId) === next) {
        this.sessionLocks.delete(sessionId);
      }
    }
  }

  private ensureAbsolutePath(targetPath: string): void {
    if (!path.isAbsolute(targetPath)) {
      throw RequestError.invalidParams({ path: targetPath }, "Path must be absolute.");
    }
  }
}

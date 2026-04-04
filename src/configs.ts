import type { ClientCapabilities } from "@agentclientprotocol/sdk";
import type { BackendCommandConfig, PermissionPolicy, ServerConfig } from "./types.js";

export const DEFAULT_ANTHROPIC_VERSION = "2023-06-01";
export const DEFAULT_HOST = "127.0.0.1";
export const DEFAULT_PORT = 4319;
export const DEFAULT_SESSION_HEADER = "x-acp-session-id";
export const DEFAULT_REQUEST_ID_HEADER = "request-id";
export const DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 128 * 1024;

export const ACP_CLIENT_CAPABILITIES: ClientCapabilities = {
  fs: {
    readTextFile: true,
    writeTextFile: true,
  },
  terminal: true,
};

export function resolvePermissionPolicy(value: string | undefined): PermissionPolicy {
  switch (value) {
    case undefined:
    case "":
      return "reject_once";
    case "allow_once":
    case "allow_always":
    case "reject_once":
    case "reject_always":
      return value;
    default:
      throw new Error(`Invalid ACP permission policy: ${value}`);
  }
}

export function resolveBackendCommand(
  env: Record<string, string | undefined>,
): BackendCommandConfig {
  const command = env.ACP_BACKEND_COMMAND?.trim() || "claude-agent-acp";
  const args = (env.ACP_BACKEND_ARGS ?? "")
    .split(/\s+/)
    .map((arg) => arg.trim())
    .filter(Boolean);

  return {
    command,
    args,
    cwd: env.ACP_BACKEND_CWD || process.cwd(),
    env: {
      ...process.env,
      ...env,
    },
  };
}

export function loadServerConfig(
  env: Record<string, string | undefined> = process.env,
): ServerConfig {
  return {
    port: Number(env.PORT ?? DEFAULT_PORT),
    host: env.HOST ?? DEFAULT_HOST,
    apiKey: env.FACADE_API_KEY,
    anthropicVersion: env.ANTHROPIC_VERSION ?? DEFAULT_ANTHROPIC_VERSION,
    traceRequests: env.CLAUDE_ACP_TRACE_REQUESTS === "1",
    sessionHeader: DEFAULT_SESSION_HEADER,
    requestIdHeader: DEFAULT_REQUEST_ID_HEADER,
    backend: resolveBackendCommand(env),
    sessionCwd: env.ACP_SESSION_CWD || process.cwd(),
    permissionPolicy: resolvePermissionPolicy(env.ACP_PERMISSION_POLICY),
    terminalOutputByteLimit: Number(
      env.ACP_TERMINAL_OUTPUT_BYTE_LIMIT ?? DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    ),
  };
}

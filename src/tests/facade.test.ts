import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { createFacadeServer } from "../lib.js";
import type { ServerConfig } from "../types.js";

const FIXTURE_PATH = path.resolve("src/tests/fixtures/mock-acp-agent.mjs");
const TEST_OUTPUT_DIR = path.resolve("test-output");
const INFERRED_CWD_FIXTURE = "/tmp/claude-acp-server-inferred-cwd";

type StartedServer = Awaited<ReturnType<ReturnType<typeof createFacadeServer>["listen"]>>;

function buildConfig(overrides: Partial<ServerConfig> = {}): ServerConfig {
  return {
    port: 0,
    host: "127.0.0.1",
    apiKey: "test-key",
    anthropicVersion: "2023-06-01",
    traceRequests: false,
    sessionHeader: "x-acp-session-id",
    requestIdHeader: "request-id",
    sessionCwd: process.cwd(),
    cwdHeader: "x-acp-cwd",
    permissionPolicy: "reject_once",
    terminalOutputByteLimit: 128 * 1024,
    backend: {
      command: process.execPath,
      args: [FIXTURE_PATH],
      cwd: process.cwd(),
      env: {
        ...process.env,
      },
    },
    ...overrides,
  };
}

async function postJson(url: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "anthropic-version": "2023-06-01",
      "x-api-key": "test-key",
      ...headers,
    },
    body: JSON.stringify(body),
  });
}

describe("Anthropic ACP facade", () => {
  let server = createFacadeServer(buildConfig(), console);
  let running: StartedServer;
  let baseUrl: string;

  beforeEach(async () => {
    await mkdir(TEST_OUTPUT_DIR, { recursive: true });
    await rm(path.join(INFERRED_CWD_FIXTURE, "test-output/mock-write.txt"), { force: true });
    server = createFacadeServer(buildConfig(), console);
    running = await server.listen();
    baseUrl = `http://${running.address.address}:${running.address.port}`;
  });

  afterEach(async () => {
    await server.close();
  });

  it("returns Anthropic-shaped JSON for non-streaming messages and preserves session continuity", async () => {
    const first = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      messages: [{ role: "user", content: "Hello from facade" }],
    });

    expect(first.status).toBe(200);
    const firstBody = (await first.json()) as any;
    const sessionId = first.headers.get("x-acp-session-id");
    expect(sessionId).toBeTruthy();
    expect(firstBody.type).toBe("message");
    expect(firstBody.content[0].text).toContain("Turn 1 (mock-sonnet-1): Hello from facade");

    const second = await postJson(
      `${baseUrl}/v1/messages`,
      {
        model: "mock-sonnet-1",
        max_tokens: 256,
        messages: [{ role: "user", content: "Hello again" }],
      },
      {
        "x-acp-session-id": sessionId!,
      },
    );

    expect(second.status).toBe(200);
    const secondBody = (await second.json()) as any;
    expect(secondBody.content[0].text).toContain("Turn 2 (mock-sonnet-1): Hello again");

    await writeFile(
      path.join(TEST_OUTPUT_DIR, "messages-response.json"),
      JSON.stringify({ first: firstBody, second: secondBody, sessionId }, null, 2),
      "utf8",
    );
  });

  it("streams Anthropic SSE events", async () => {
    const response = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      stream: true,
      messages: [{ role: "user", content: "Streaming reply" }],
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");

    const transcript = await response.text();
    expect(transcript).toContain(": connected");
    expect(transcript).toContain("event: message_start");
    expect(transcript).toContain("event: content_block_start");
    expect(transcript).toContain("event: content_block_delta");
    expect(transcript).toContain("event: message_delta");
    expect(transcript).toContain("event: message_stop");
    expect(transcript).not.toContain('"delta":{"type":"text_delta","text":""}');

    const messageStartMatch = transcript.match(/event: message_start\ndata: (.+)\n/);
    expect(messageStartMatch).toBeTruthy();
    const messageStart = JSON.parse(messageStartMatch![1]) as any;
    expect(messageStart.message.usage.input_tokens).toBeGreaterThan(0);

    const messageDeltaMatch = transcript.match(/event: message_delta\ndata: (.+)\n/);
    expect(messageDeltaMatch).toBeTruthy();
    const messageDelta = JSON.parse(messageDeltaMatch![1]) as any;
    expect(messageDelta.usage.input_tokens).toBe(11);
    expect(messageDelta.usage.output_tokens).toBe(21);

    await writeFile(path.join(TEST_OUTPUT_DIR, "streaming-transcript.txt"), transcript, "utf8");
  });

  it("accepts requests that include Anthropic client tools", async () => {
    const response = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      tools: [
        {
          name: "wiki",
          description: "Look up wiki entries",
          input_schema: {
            type: "object",
            properties: {
              query: { type: "string" },
            },
            required: ["query"],
          },
        },
      ],
      messages: [{ role: "user", content: "Hello with tools" }],
    });

    expect(response.status).toBe(200);
    const body = (await response.json()) as any;
    expect(body.type).toBe("message");
    expect(body.content[0].type).toBe("text");
  });

  it("streams plain text even when Anthropic client tools are present", async () => {
    const response = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      stream: true,
      tools: [
        {
          name: "wiki",
          description: "Look up wiki entries",
          input_schema: {
            type: "object",
            properties: {
              query: { type: "string" },
            },
            required: ["query"],
          },
        },
      ],
      messages: [{ role: "user", content: "Hello with streaming tools" }],
    });

    expect(response.status).toBe(200);
    const transcript = await response.text();
    expect(transcript).toContain("event: content_block_delta");
    expect(transcript).toContain("Hello with streaming tools");
    expect(transcript).not.toContain('"type":"tool_use"');
  });

  it("bridges a tool request into an Anthropic tool_use response", async () => {
    const response = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      tools: [
        {
          name: "wiki",
          description: "Look up wiki entries",
          input_schema: {
            type: "object",
            properties: {
              query: { type: "string" },
            },
            required: ["query"],
          },
        },
      ],
      messages: [{ role: "user", content: "TRIGGER TOOL BRIDGE" }],
    });

    expect(response.status).toBe(200);
    const body = (await response.json()) as any;
    expect(body.stop_reason).toBe("tool_use");
    expect(body.content[0].type).toBe("tool_use");
    expect(body.content[0].name).toBe("wiki");
    expect(body.content[0].input).toEqual({ query: "acp" });
  });

  it("surfaces ACP-native progress as thinking blocks while keeping tool_use hidden", async () => {
    const streaming = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      stream: true,
      messages: [{ role: "user", content: "TRIGGER TOOL CALL" }],
    });

    expect(streaming.status).toBe(200);
    const transcript = await streaming.text();
    expect(transcript).toContain("Based on the file");
    expect(transcript).toContain('"type":"thinking"');
    expect(transcript).toContain("thinking_delta");
    expect(transcript).toContain("Using Read");
    expect(transcript).toContain("Completed Read");
    expect(transcript).not.toContain('"type":"tool_use"');

    const nonStreaming = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      messages: [{ role: "user", content: "TRIGGER TOOL CALL" }],
    });

    expect(nonStreaming.status).toBe(200);
    const body = (await nonStreaming.json()) as any;
    expect(body.content).toEqual([
      {
        type: "text",
        text: "Based on the file, here is the answer.",
        citations: null,
      },
    ]);

    await writeFile(
      path.join(TEST_OUTPUT_DIR, "tool-call-transcript.txt"),
      JSON.stringify({ transcript, body }, null, 2),
      "utf8",
    );
  });

  it("infers the ACP session cwd from Droid-style system reminders", async () => {
    const response = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 64,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `<system-reminder>\n% pwd\n${INFERRED_CWD_FIXTURE}\n</system-reminder>`,
            },
            {
              type: "text",
              text: "write file",
            },
          ],
        },
      ],
    });

    expect(response.status).toBe(200);
    const mockWrite = await readFile(
      path.join(INFERRED_CWD_FIXTURE, "test-output/mock-write.txt"),
      "utf8",
    );
    expect(mockWrite).toContain("written by mock agent");
  });

  it("lists models from the ACP backend", async () => {
    const response = await fetch(`${baseUrl}/v1/models`, {
      headers: {
        "anthropic-version": "2023-06-01",
        "x-api-key": "test-key",
      },
    });

    expect(response.status).toBe(200);
    const body = (await response.json()) as any;
    expect(body.data.map((model: { id: string }) => model.id)).toContain("mock-sonnet-1");
  });

  it("maps backend auth errors into Anthropic auth errors", async () => {
    await server.close();
    server = createFacadeServer(
      buildConfig({
        backend: {
          command: process.execPath,
          args: [FIXTURE_PATH],
          cwd: process.cwd(),
          env: {
            ...process.env,
            MOCK_ACP_AUTH_REQUIRED: "1",
          },
        },
      }),
      console,
    );
    running = await server.listen();
    baseUrl = `http://${running.address.address}:${running.address.port}`;

    const response = await postJson(`${baseUrl}/v1/messages`, {
      model: "mock-sonnet-1",
      max_tokens: 256,
      messages: [{ role: "user", content: "Needs auth" }],
    });

    expect(response.status).toBe(401);
    const body = (await response.json()) as any;
    expect(body.error.type).toBe("authentication_error");
  });

  it("maps invalid session reuse to not_found_error", async () => {
    const response = await postJson(
      `${baseUrl}/v1/messages`,
      {
        model: "mock-sonnet-1",
        max_tokens: 256,
        messages: [{ role: "user", content: "Unknown session" }],
      },
      {
        "x-acp-session-id": "missing-session",
      },
    );

    expect(response.status).toBe(404);
    const body = (await response.json()) as any;
    expect(body.error.type).toBe("not_found_error");
  });
});

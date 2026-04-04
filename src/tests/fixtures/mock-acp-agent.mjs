/* eslint-env node */

import { AgentSideConnection, RequestError, ndJsonStream } from "@agentclientprotocol/sdk";
import { randomUUID } from "node:crypto";
import { ReadableStream, WritableStream } from "node:stream/web";

function nodeToWebWritable(nodeStream) {
  return new WritableStream({
    write(chunk) {
      return new Promise((resolve, reject) => {
        nodeStream.write(Buffer.from(chunk), (error) => {
          if (error) {
            reject(error);
            return;
          }
          resolve();
        });
      });
    },
  });
}

function nodeToWebReadable(nodeStream) {
  return new ReadableStream({
    start(controller) {
      nodeStream.on("data", (chunk) => controller.enqueue(new Uint8Array(chunk)));
      nodeStream.on("end", () => controller.close());
      nodeStream.on("error", (error) => controller.error(error));
    },
  });
}

const AVAILABLE_MODELS = [
  { modelId: "mock-sonnet-1", name: "Mock Sonnet 1" },
  { modelId: "mock-opus-1", name: "Mock Opus 1" },
];

class MockAgent {
  constructor(client) {
    this.client = client;
    this.sessions = new Map();
  }

  async initialize() {
    return {
      protocolVersion: 1,
      agentCapabilities: {
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
        loadSession: true,
        sessionCapabilities: {
          resume: {},
          close: {},
          list: {},
        },
      },
      agentInfo: {
        name: "mock-acp-agent",
        version: "0.0.1",
      },
      authMethods: [],
    };
  }

  async newSession(params) {
    if (process.env.MOCK_ACP_AUTH_REQUIRED === "1") {
      throw RequestError.authRequired();
    }

    const sessionId = randomUUID();
    const session = {
      sessionId,
      cwd: params.cwd,
      turns: 0,
      model: AVAILABLE_MODELS[0].modelId,
      transcript: [],
    };
    this.sessions.set(sessionId, session);

    return {
      sessionId,
      models: {
        currentModelId: session.model,
        availableModels: AVAILABLE_MODELS,
      },
    };
  }

  async loadSession(params) {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw RequestError.resourceNotFound(params.sessionId);
    }
    return {
      sessionId: session.sessionId,
      models: {
        currentModelId: session.model,
        availableModels: AVAILABLE_MODELS,
      },
    };
  }

  async unstable_resumeSession(params) {
    return this.loadSession(params);
  }

  async unstable_closeSession(params) {
    this.sessions.delete(params.sessionId);
    return {};
  }

  async listSessions() {
    return {
      sessions: Array.from(this.sessions.values()).map((session) => ({
        sessionId: session.sessionId,
        cwd: session.cwd,
        title: `Mock Session ${session.turns}`,
        updatedAt: new Date().toISOString(),
      })),
    };
  }

  async unstable_setSessionModel(params) {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw RequestError.resourceNotFound(params.sessionId);
    }
    if (!AVAILABLE_MODELS.find((model) => model.modelId === params.modelId)) {
      throw RequestError.invalidParams({ modelId: params.modelId }, "Unknown model.");
    }
    session.model = params.modelId;
    return {
      models: {
        currentModelId: session.model,
        availableModels: AVAILABLE_MODELS,
      },
    };
  }

  async authenticate() {}

  async cancel() {}

  async prompt(params) {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw RequestError.resourceNotFound(params.sessionId);
    }

    const text = params.prompt
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("\n");

    session.turns += 1;
    session.transcript.push(text);

    if (text.includes("write file")) {
      await this.client.writeTextFile({
        sessionId: params.sessionId,
        path: `${session.cwd}/test-output/mock-write.txt`,
        content: "written by mock agent\n",
      });
    }

    if (text.includes("TRIGGER TOOL BRIDGE")) {
      await this.client.sessionUpdate({
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: {
            type: "text",
            text: '<anthropic_tool_use>{"name":"wiki","input":{"query":"acp"}}</anthropic_tool_use>',
          },
        },
      });

      return {
        stopReason: "end_turn",
        usage: {
          inputTokens: 10 + session.turns,
          outputTokens: 20 + session.turns,
          cachedReadTokens: 0,
          cachedWriteTokens: 0,
        },
      };
    }

    await this.client.sessionUpdate({
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: `Turn ${session.turns} (${session.model}): `,
        },
      },
    });

    await this.client.sessionUpdate({
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text,
        },
      },
    });

    return {
      stopReason: "end_turn",
      usage: {
        inputTokens: 10 + session.turns,
        outputTokens: 20 + session.turns,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
    };
  }
}

const stream = ndJsonStream(nodeToWebWritable(process.stdout), nodeToWebReadable(process.stdin));
new AgentSideConnection((client) => new MockAgent(client), stream);
process.stdin.resume();

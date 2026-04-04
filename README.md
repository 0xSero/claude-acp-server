# `claude-acp-server`

`claude-acp-server` is a small HTTP server that makes a `claude-agent-acp` backend look like the Anthropic Messages API.

The northbound interface is Anthropic-compatible enough for clients that expect:

- `POST /v1/messages`
- `GET /v1/models`
- server-sent events for `stream: true`
- `anthropic-version`
- `x-api-key` or `Authorization: Bearer ...`

The southbound interface is ACP over stdio.

## What This Is

This project exists for one specific reason: some clients only know how to talk to the Anthropic API, while the real agent runtime is exposed through ACP. This server sits between those two sides and translates requests in both directions.

The shape is:

```text
Anthropic client
  -> claude-acp-server
  -> claude-agent-acp
  -> ACP-backed Claude agent
```

It does not try to replace `claude-agent-acp`. It uses `claude-agent-acp` as the backend engine and only changes the interface that clients talk to.

## Why It Works

It works because the problem is mostly interface translation, not model translation.

- Anthropic clients send a structured message request.
- ACP expects a session plus a prompt payload.
- `claude-agent-acp` already knows how to run the agent and manage ACP session state.
- This server turns an Anthropic request into ACP prompt input, forwards it to `claude-agent-acp`, then turns the resulting ACP updates back into Anthropic message JSON or Anthropic SSE events.

The server stays small because it does not try to emulate Anthropic's full platform. It only implements the inference surface needed for normal message-based clients.

## What It Does

- Validates Anthropic-style request headers.
- Starts and maintains a persistent `claude-agent-acp` subprocess.
- Creates a new ACP session when `x-acp-session-id` is absent.
- Reuses an ACP session when `x-acp-session-id` is present.
- Maps Anthropic message content into ACP prompt blocks.
- Maps ACP streamed text updates into Anthropic SSE events.
- Returns Anthropic-shaped non-streaming message responses.
- Exposes backend model availability through `GET /v1/models`.

## What It Does Not Do

- It does not implement Anthropic's full public API surface.
- It does not support client-defined Anthropic tools.
- It does not replay conversation history statelessly on every request.
- It does not embed the Claude Agent SDK directly.

Session continuity is explicit. If a client wants a multi-turn conversation, it must send back the `x-acp-session-id` returned by the server.

## Request Flow

For a non-streaming request:

1. The client sends `POST /v1/messages`.
2. The server validates `anthropic-version` and auth.
3. The server creates or resumes an ACP session.
4. The final Anthropic user turn is converted into ACP prompt input.
5. Earlier Anthropic turns are folded into a transcript bootstrap block for the ACP session.
6. The request is sent to `claude-agent-acp`.
7. ACP notifications and the final ACP prompt result are converted into an Anthropic message response.

For a streaming request:

1. The server opens an SSE response.
2. ACP `agent_message_chunk` notifications are translated into Anthropic `content_block_*` events.
3. The final ACP prompt result is translated into `message_delta` and `message_stop`.

## Compatibility Notes

- The required Anthropic version is `2023-06-01` unless you change it in config.
- Auth can be provided through `x-api-key` or `Authorization: Bearer ...`.
- The response includes `x-acp-session-id` on successful message requests.
- Model IDs are backend-dependent. The facade can also alias client-facing IDs to backend IDs where needed.
- For streaming responses, `message_start.usage` is a provisional estimate (to avoid zero-token clients); the authoritative usage comes in the final `message_delta` event.

Current built-in aliases:

- `claude-sonnet-4-6` -> `sonnet`
- `claude-opus-4-6` -> `default`

## Configuration

Environment variables:

- `HOST`
- `PORT`
- `FACADE_API_KEY`
- `ACP_BACKEND_COMMAND`
- `ACP_BACKEND_ARGS`
- `ACP_BACKEND_CWD`
- `ACP_SESSION_CWD`
- `ACP_PERMISSION_POLICY`
- `ACP_TERMINAL_OUTPUT_BYTE_LIMIT`

Default values:

- `HOST=127.0.0.1`
- `PORT=4319`
- `ACP_BACKEND_COMMAND=claude-agent-acp`
- `ACP_PERMISSION_POLICY=reject_once`

## Run

```bash
npm install
npm run build
FACADE_API_KEY=test-key \
ACP_BACKEND_COMMAND="npx" \
ACP_BACKEND_ARGS="-y @agentclientprotocol/claude-agent-acp" \
npm start
```

There is also a publish-safe local runner example at `scripts/local-server.example.sh`.

## Example Request

```bash
curl http://127.0.0.1:4319/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-api-key: test-key' \
  -d '{
    "model": "sonnet",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: OK"
      }
    ]
  }'
```

## Verification

```bash
npm run build
npm test
npm run check
```

Proof artifacts from the integration tests are written under `test-output/`.

## Code Layout

The code is intentionally split into two layers.

`src/logic/acp-client/*`

- backend process management
- ACP session lifecycle
- terminal and file operations required by the ACP client side

`src/logic/anthropic-api/*`

- HTTP handling
- Anthropic request validation
- Anthropic request and response translation
- SSE emission

The boundary is the point of the project. Northbound is Anthropic. Southbound is ACP.

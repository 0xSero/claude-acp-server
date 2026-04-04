import { Readable, Writable } from "node:stream";
import { ReadableStream, WritableStream } from "node:stream/web";
import type { ServerResponse } from "node:http";

export function nodeToWebWritable(nodeStream: Writable): WritableStream<Uint8Array> {
  return new WritableStream<Uint8Array>({
    write(chunk) {
      return new Promise<void>((resolve, reject) => {
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

export function nodeToWebReadable(nodeStream: Readable): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      nodeStream.on("data", (chunk: Buffer | string) => {
        controller.enqueue(typeof chunk === "string" ? Buffer.from(chunk) : new Uint8Array(chunk));
      });
      nodeStream.on("end", () => controller.close());
      nodeStream.on("error", (error) => controller.error(error));
    },
  });
}

export async function readJsonBody<T>(request: Readable): Promise<T> {
  const chunks: Buffer[] = [];

  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  const raw = Buffer.concat(chunks).toString("utf8");
  return JSON.parse(raw) as T;
}

export function writeJson(response: ServerResponse, status: number, body: unknown): void {
  response.statusCode = status;
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.setHeader("connection", "close");
  response.end(JSON.stringify(body));
}

export function openSse(response: ServerResponse): void {
  response.statusCode = 200;
  response.setHeader("content-type", "text/event-stream; charset=utf-8");
  response.setHeader("cache-control", "no-cache, no-transform");
  response.setHeader("connection", "keep-alive");
  response.setHeader("x-accel-buffering", "no");
  response.socket?.setNoDelay(true);
  response.socket?.setKeepAlive(true);
  response.flushHeaders();
  response.write(": connected\n\n");
}

export function writeSseEvent(response: ServerResponse, event: string, data: unknown): void {
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}

export function closeSse(response: ServerResponse): void {
  response.write("event: message_stop\n");
  response.write('data: {"type":"message_stop"}\n\n');
  response.end();
}

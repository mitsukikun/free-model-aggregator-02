// free-model-aggregator-02 entry: one server, two client wires, one real free model behind
// every answer, routed by the brief's published ratings.
//
// Built from the kit's starter (same lib/providers.ts + lib/wire.ts, unchanged) with the
// baseline's four gaps closed:
//   - quality (60%): candidates are ranked by rating_milli desc at the prompt's requirements
//     (best first), ties broken by fewest calls made to that model, then by model id;
//   - failover (10%): a refused or unusable gateway call is retried on the next free model
//     that meets the request, and every refusal is reported on the answer in the
//     `x-makefun-upstream-failures` header (JSON array, at most 8 rows of
//     {status, code, model}) — the only way the failover column sees a recovered retry;
//   - rate limits: a model that refuses is cooled down for the next prompt, so one
//     rate-limited model does not take every later prompt down with it; a model that
//     answers without the prompt's token, or without usable usage, is cooled the same way;
//   - the catalog (5%): exactly the fourteen dialable ids the brief lists, on both wires.
//
// Every choice is a function of the brief, the requests, and the gateway answers: no clock,
// no unseeded randomness. A prompt's token is read off the request's own message text (the
// brief carries no tokens); an answer that does not carry it is not relayed, it is failed over.
// Streams are buffered whole before relaying, so a stream that breaks mid-way is failed over
// instead of delivered truncated.
//
// The trial's rules are in the kit's RULES.txt; the score's columns and the tie-break are there.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type PromptRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one upstream answer body this entry reads. */
const MAX_UPSTREAM_BYTES = 2 * 1024 * 1024;
/** The header that carries this answer's recovered upstream refusals. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
/** The most failure rows one answer reports. */
const MAX_FAILURE_ROWS = 8;
/** Gateway calls per prompt, at most: the trial's cap is 60, and 6 prompts x 4 is 24. */
const MAX_ATTEMPTS = 4;
/** Per-attempt gateway timeouts, in milliseconds: the first attempt gets the most time, later
 *  attempts less, so four attempts still fit inside the client's 30 s wait for one answer. */
const ATTEMPT_TIMEOUTS_MS = [9000, 7000, 5000, 3000];
/** A max_tokens the dialled body names when the client body carries none. */
const DEFAULT_MAX_TOKENS = 512;
/** A token is a word and four digits, e.g. SPARROW-7435. */
const TOKEN_RE = /\b[A-Z]+-[0-9]{4}\b/;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/** One upstream refusal, as the failures header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** A free model's trial state: what it has cost, and whether it is resting or dead. */
interface ModelState {
  calls: number;
  refusals: number;
  /** The last prompt index this model is cooled for, or -1 when it is not cooling. */
  coolUntilPrompt: number;
  dead: boolean;
}

const states = new Map<string, ModelState>();
/** A 401/403/402 means the gateway key is missing or spent: no model will answer until it is fixed. */
let authDead = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(modelId: string): ModelState {
  let state = states.get(modelId);
  if (state === undefined) {
    state = { calls: 0, refusals: 0, coolUntilPrompt: -1, dead: false };
    states.set(modelId, state);
  }
  return state;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a prompt, best first: rating_milli desc, then fewest calls made
 * to that model, then model id. A model that is dead, that refused on this prompt already, or
 * that is cooling down from a refusal on the previous prompt, is not offered.
 */
function candidatesFor(requires: Requirement, promptIndex: number, refusedThisPrompt: ReadonlySet<string>): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    const state = stateOf(model.id);
    if (state.dead) continue;
    if (refusedThisPrompt.has(model.id)) continue;
    if (promptIndex <= state.coolUntilPrompt) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    out.push(model);
  }
  out.sort(
    (a, b) =>
      b.rating_milli - a.rating_milli ||
      stateOf(a.id).calls - stateOf(b.id).calls ||
      (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; prompt: PromptRow | undefined; promptIndex: number } {
  const promptIndex = manifestIndex;
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, prompt: entry, promptIndex };
  return { requires: requirementsFromBody(body), prompt: undefined, promptIndex };
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The prompt's unique token, read off the request's own message text, or null when it carries none. */
function tokenOf(body: Json): string | null {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    const texts: string[] = [];
    if (typeof content === 'string') texts.push(content);
    else if (Array.isArray(content)) {
      for (const block of content) if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
    }
    for (const text of texts) {
      const found = TOKEN_RE.exec(text);
      if (found !== null) return found[0];
    }
  }
  return null;
}

/** One streamed chunk as a Buffer: undici fetch bodies yield Uint8Array (not Buffer), while
 *  node's http requests yield Buffer. String(chunk) on a Uint8Array would join its bytes
 *  with commas — the body would read as "123,34,105,..." and no JSON would ever parse. */
function chunkToBuffer(chunk: unknown): Buffer {
  if (Buffer.isBuffer(chunk)) return chunk;
  if (chunk instanceof Uint8Array) return Buffer.from(chunk);
  return Buffer.from(String(chunk));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunkToBuffer(chunk);
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One upstream answer body as text, or null when it is too large to read. */
async function readUpstreamText(res: Response): Promise<string | null> {
  if (res.body === null) return '';
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of res.body) {
    const buffer = chunkToBuffer(chunk);
    size += buffer.length;
    if (size > MAX_UPSTREAM_BYTES) return null;
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The provider's error code, when its body carries one in the OpenAI envelope. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value)) return '';
    const error = isRecord(value.error) ? value.error : null;
    if (error !== null && typeof error.code === 'string' && error.code !== '') return error.code.slice(0, 64);
    if (error !== null && typeof error.type === 'string' && error.type !== '') return error.type.slice(0, 64);
    if (error !== null && typeof error.message === 'string' && error.message !== '') return error.message.slice(0, 64);
    return '';
  } catch {
    return '';
  }
}

/** Whether a refusal is permanent for this trial: the provider's own message says its free-tier
 *  budget is spent, or that it is out. A scheduled refusal says only that it refused this call. */
function refusalIsPermanent(message: string): boolean {
  return message.includes('budget is spent') || message.includes(' is out');
}

/** The dialled body for one gateway call: the client's body, naming the model, always with a
 *  max_tokens and, for a stream, with usage on. */
function dialBody(body: Json, modelId: string): Json {
  const out: Json = { ...body, model: modelId };
  if (typeof out.max_tokens !== 'number') out.max_tokens = DEFAULT_MAX_TOKENS;
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(modelId: string, body: Json, timeoutMs: number): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(dialBody(body, modelId)),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/** The text of a JSON answer, as the env's content check reads it: every choice message's content. */
function jsonAnswerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  let text = '';
  for (const choice of choices) {
    if (!isRecord(choice)) continue;
    const message = isRecord(choice.message) ? choice.message : isRecord(choice.delta) ? choice.delta : null;
    if (message === null) continue;
    const content = message.content;
    if (typeof content === 'string') text += content;
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === 'string') text += part;
        else if (isRecord(part) && typeof part.text === 'string') text += part.text;
      }
    }
  }
  return text;
}

/** The text of a streamed answer, as the env's content check reads it: every content delta. */
function streamAnswerText(sseText: string): string {
  let text = '';
  for (const line of sseText.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '[DONE]') continue;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
      const choice = chunk.choices[0];
      if (!isRecord(choice) || !isRecord(choice.delta)) continue;
      if (typeof choice.delta.content === 'string') text += choice.delta.content;
    } catch {
      // Not JSON: the relay's translator will fail the same way; the token check stays silent.
    }
  }
  return text;
}

/** Whether a JSON answer is usable: it carries the token (when the prompt names one) and a whole
 *  usage with an output count of at least 1. */
function jsonAnswerUsable(answer: Json, token: string | null): string | null {
  if (token !== null && !jsonAnswerText(answer).includes(token)) return 'no_token';
  const usage = isRecord(answer.usage) ? answer.usage : null;
  const input = usage !== null && typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : null;
  const output = usage !== null && typeof usage.completion_tokens === 'number' ? usage.completion_tokens : null;
  if (input === null || output === null || input < 0 || output < 1) return 'no_usage';
  return null;
}

/** Whether a streamed answer is usable: its content deltas carry the token, it ends with exactly
 *  one finish chunk and [DONE], and its usage chunk counts an output of at least 1. */
function streamAnswerUsable(sseText: string, token: string | null): string | null {
  if (token !== null && !streamAnswerText(sseText).includes(token)) return 'no_token';
  const finishes = sseText.match(/"finish_reason"\s*:\s*"(stop|length|tool_calls|function_call|content_filter)"/g) ?? [];
  if (finishes.length !== 1) return 'no_stop';
  if (!sseText.trimEnd().endsWith('data: [DONE]')) return 'no_done';
  const usage = /"completion_tokens"\s*:\s*(\d+)/.exec(sseText);
  if (usage === null || Number(usage[1]) < 1) return 'no_usage';
  return null;
}

/** A ReadableStream of one buffered text, for the wire translators. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** One client request: the best free model that meets it, failing over while models refuse. */
async function serve(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const { requires, promptIndex } = requirementsFor(body);
  const token = tokenOf(body);
  const stream = body.stream === true;
  const failures: FailureRow[] = [];
  const refusedThisPrompt = new Set<string>();
  const attempts = authDead ? 1 : MAX_ATTEMPTS;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const candidates = candidatesFor(requires, promptIndex, refusedThisPrompt);
    if (candidates.length === 0) break;
    const model = candidates[0];
    const state = stateOf(model.id);
    const upstreamBody: Json = api === 'anthropic' ? anthropicToOpenAI(body, model.id) : body;
    let upstream: Response;
    try {
      upstream = await dial(model.id, upstreamBody, ATTEMPT_TIMEOUTS_MS[attempt] ?? 3000);
    } catch (err) {
      const timedOut = (err as Error).name === 'TimeoutError';
      failures.push({ status: timedOut ? 504 : 502, code: timedOut ? 'timeout' : 'upstream_error', model: model.id });
      state.calls += 1;
      state.refusals += 1;
      state.coolUntilPrompt = promptIndex + 1;
      refusedThisPrompt.add(model.id);
      continue;
    }
    state.calls += 1;
    if (!upstream.ok || upstream.body === null) {
      const text = (await upstream.text().catch(() => '')).slice(0, 500);
      const code = errorCodeOf(text) === '' ? `http_${upstream.status}` : errorCodeOf(text);
      failures.push({ status: upstream.status, code, model: model.id });
      state.refusals += 1;
      state.coolUntilPrompt = promptIndex + 1;
      refusedThisPrompt.add(model.id);
      if (upstream.status === 401 || upstream.status === 403 || upstream.status === 402) {
        authDead = true;
        break;
      }
      if (upstream.status === 400 || upstream.status === 404 || refusalIsPermanent(text)) state.dead = true;
      continue;
    }
    const text = await readUpstreamText(upstream);
    if (text === null) {
      failures.push({ status: 502, code: 'answer_too_large', model: model.id });
      state.refusals += 1;
      state.coolUntilPrompt = promptIndex + 1;
      refusedThisPrompt.add(model.id);
      continue;
    }
    if (!stream) {
      let answer: unknown;
      try {
        answer = JSON.parse(text);
      } catch {
        failures.push({ status: 502, code: 'bad_upstream_body', model: model.id });
        state.refusals += 1;
        state.coolUntilPrompt = promptIndex + 1;
        refusedThisPrompt.add(model.id);
        continue;
      }
      if (!isRecord(answer)) {
        failures.push({ status: 502, code: 'bad_upstream_body', model: model.id });
        state.refusals += 1;
        state.coolUntilPrompt = promptIndex + 1;
        refusedThisPrompt.add(model.id);
        continue;
      }
      const unusable = jsonAnswerUsable(answer, token);
      if (unusable !== null) {
        failures.push({ status: 200, code: unusable, model: model.id });
        state.refusals += 1;
        state.coolUntilPrompt = promptIndex + 1;
        refusedThisPrompt.add(model.id);
        continue;
      }
      if (api === 'anthropic') sendJson(res, 200, openAIToAnthropic(answer, model.id), failures);
      else sendJson(res, 200, openAIJsonWithModel(answer, model.id), failures);
      if (authDead) authDead = false;
      return;
    }
    const unusable = streamAnswerUsable(text, token);
    if (unusable !== null) {
      failures.push({ status: 200, code: unusable, model: model.id });
      state.refusals += 1;
      state.coolUntilPrompt = promptIndex + 1;
      refusedThisPrompt.add(model.id);
      continue;
    }
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
    res.writeHead(200, headers);
    try {
      if (api === 'anthropic') {
        for await (const event of anthropicEventsFromOpenAI(streamOf(text), model.id)) res.write(event);
      } else {
        for await (const frame of openAIStreamWithModel(streamOf(text), model.id)) res.write(frame);
      }
    } finally {
      res.end();
    }
    if (authDead) authDead = false;
    return;
  }

  sendError(
    res,
    api,
    503,
    api === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'no free model on the directory meets this request',
    failures,
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

// Thin fetch wrapper for the VibeTTS Reader API (same origin, /api/...).

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Thrown when the server cannot be reached at all (app not running, network down, hung request). */
export class OfflineError extends Error {
  constructor(message = "VibeTTS is not running") {
    super(message);
    this.name = "OfflineError";
    this.offline = true;
  }
}

async function request(method, path, { json, body, headers = {}, timeoutMs = 60000, keepalive = false } = {}) {
  const init = { method, headers: { ...headers }, keepalive };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  const ctrl = new AbortController();
  init.signal = ctrl.signal;
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    clearTimeout(timer);
    throw new OfflineError();
  }
  let text;
  try {
    text = await res.text();
  } catch (e) {
    clearTimeout(timer);
    throw new OfflineError();
  }
  clearTimeout(timer);
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = null;
    }
  }
  if (!res.ok) {
    const msg = (data && data.error) || `Request failed (${res.status})`;
    throw new ApiError(msg, res.status);
  }
  return data;
}

const enc = encodeURIComponent;

export const info = () => request("GET", "/api/info");
export const voices = () => request("GET", "/api/voices");

export const speak = (payload) => request("POST", "/api/speak", { json: payload });
export const pause = () => request("POST", "/api/pause", { json: {} });
export const resume = () => request("POST", "/api/resume", { json: {} });
export const stop = () => request("POST", "/api/stop", { json: {} });
export const seek = (item) => request("POST", "/api/seek", { json: { item } });

/** Long-poll: resolves at once if the server's version differs from `since`, else after `timeout` seconds. */
export function status(since, timeout = 20) {
  const q = new URLSearchParams();
  if (since !== undefined && since !== null) q.set("since", String(since));
  q.set("timeout", String(timeout));
  return request("GET", `/api/status?${q}`, { timeoutMs: (timeout + 10) * 1000 });
}

export const books = async () => (await request("GET", "/api/books")).books || [];
export const getBook = (id) => request("GET", `/api/books/${enc(id)}`);
export const getChapter = (id, n) => request("GET", `/api/books/${enc(id)}/chapters/${n}`);
export const deleteBook = (id) => request("DELETE", `/api/books/${enc(id)}`);
export const setPosition = (id, chapter, item, opts = {}) =>
  request("PUT", `/api/books/${enc(id)}/position`, { json: { chapter, item }, keepalive: !!opts.keepalive });

export function uploadBook(file) {
  return request("POST", "/api/books", {
    body: file,
    headers: { "Content-Type": "application/octet-stream", "X-Filename": enc(file.name) },
    timeoutMs: 10 * 60 * 1000,
  });
}

// VibeTTS Reader service worker: talks to the local VibeTTS app, drives the content script.
// Never logs page text.

const DEFAULT_PORT = 7724;
const ACTIVE_STATES = ["loading", "playing", "paused"];
const SESSION_KEY = "vibetts_ours"; // chrome.storage.session: {session, tabId}

let polling = false;
let badgeTimer = null;

// ---------------------------------------------------------------- helpers

async function getPort() {
  try {
    const { port } = await chrome.storage.local.get("port");
    const n = Number(port);
    return Number.isInteger(n) && n > 0 && n < 65536 ? n : DEFAULT_PORT;
  } catch (e) {
    return DEFAULT_PORT;
  }
}

class OfflineError extends Error {}

async function api(path, { method = "GET", body } = {}) {
  const base = `http://127.0.0.1:${await getPort()}`;
  let resp;
  try {
    resp = await fetch(base + path, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new OfflineError("VibeTTS is not reachable");
  }
  let data = null;
  try {
    data = await resp.json();
  } catch (e) {
    /* non-JSON body */
  }
  if (!resp.ok) {
    const err = new Error((data && data.error) || `HTTP ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return data;
}

function setBadge(text, color = "#c0392b", clearAfterMs = 5000) {
  try {
    chrome.action.setBadgeBackgroundColor({ color });
    chrome.action.setBadgeText({ text });
    if (badgeTimer) clearTimeout(badgeTimer);
    if (text && clearAfterMs) {
      badgeTimer = setTimeout(() => chrome.action.setBadgeText({ text: "" }), clearAfterMs);
    }
  } catch (e) {
    /* ignore */
  }
}

async function getOurs() {
  try {
    const o = (await chrome.storage.session.get(SESSION_KEY))[SESSION_KEY];
    return o && o.session ? o : null;
  } catch (e) {
    return null;
  }
}

async function setOurs(o) {
  try {
    if (o) await chrome.storage.session.set({ [SESSION_KEY]: o });
    else await chrome.storage.session.remove(SESSION_KEY);
  } catch (e) {
    console.warn("VibeTTS: storage.session failed", e);
  }
}

function tellTab(tabId, msg) {
  if (tabId == null) return;
  chrome.tabs.sendMessage(tabId, msg).catch(() => {});
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- polling

async function pollLoop() {
  if (polling) return;
  polling = true;
  let since = null;
  let failures = 0;
  let lastItem = -1;
  let lastState = "";
  try {
    for (;;) {
      const ours = await getOurs();
      if (!ours) break;
      let status;
      try {
        const q = since == null ? "timeout=0" : `since=${since}&timeout=20`;
        status = await api(`/api/status?${q}`);
        failures = 0;
      } catch (e) {
        failures += 1;
        if (e instanceof OfflineError) setBadge("off");
        if (failures >= 5) {
          tellTab(ours.tabId, { cmd: "clear" });
          await setOurs(null);
          break;
        }
        await sleep(2000);
        continue;
      }
      since = status.version;
      if (status.session !== ours.session) {
        tellTab(ours.tabId, { cmd: "clear" });
        await setOurs(null);
        break;
      }
      if (!ACTIVE_STATES.includes(status.state)) {
        tellTab(ours.tabId, { cmd: "clear" });
        if (status.state === "error") {
          console.warn("VibeTTS: reading failed:", status.error);
          setBadge("err");
        }
        await setOurs(null);
        break;
      }
      if (status.item !== lastItem || status.state !== lastState) {
        lastItem = status.item;
        lastState = status.state;
        tellTab(ours.tabId, { cmd: "progress", item: status.item, state: status.state });
      }
    }
  } finally {
    polling = false;
  }
}

function resumePolling() {
  pollLoop().catch((e) => console.warn("VibeTTS: poll loop failed", e));
}

// ---------------------------------------------------------------- reading

async function collect(tabId, mode) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ["segment.js", "content.js"] });
  return chrome.tabs.sendMessage(tabId, { cmd: "collect", mode });
}

async function readTab(tabId, mode, tabInfo) {
  if (tabId == null) return { ok: false, error: "No active tab." };
  let result;
  try {
    result = await collect(tabId, mode);
  } catch (e) {
    console.warn("VibeTTS: cannot read this page:", e && e.message);
    setBadge("!");
    return { ok: false, error: "This page cannot be read (restricted or not loaded)." };
  }
  const items = (result && result.items) || [];
  if (!items.length) {
    setBadge("0");
    return { ok: false, error: "No readable text found." };
  }
  let tab = tabInfo;
  if (!tab) {
    try {
      tab = await chrome.tabs.get(tabId);
    } catch (e) {
      tab = {};
    }
  }
  const sync = await chrome.storage.sync.get(["voice", "lang", "speed"]);
  const body = {
    items,
    start: result.start || 0,
    source: "extension",
    meta: { tabId, url: tab.url || "", title: tab.title || "" },
  };
  if (sync.voice) body.voice = sync.voice;
  if (sync.lang) body.lang = sync.lang;
  const speed = Number(sync.speed);
  if (speed >= 0.5 && speed <= 2.0) body.speed = speed;
  try {
    const res = await api("/api/speak", { method: "POST", body });
    await setOurs({ session: res.session, tabId });
    resumePolling();
    return { ok: true, count: res.count };
  } catch (e) {
    tellTab(tabId, { cmd: "clear" });
    if (e instanceof OfflineError) {
      setBadge("off");
      return { ok: false, offline: true, error: "VibeTTS is not running." };
    }
    console.warn("VibeTTS: speak failed:", e.message);
    setBadge("err");
    return { ok: false, error: e.message };
  }
}

async function activeTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab ? tab.id : null;
}

async function control(action) {
  try {
    return { ok: true, status: await api(`/api/${action}`, { method: "POST", body: {} }) };
  } catch (e) {
    if (e instanceof OfflineError) setBadge("off");
    return { ok: false, offline: e instanceof OfflineError, error: e.message };
  }
}

async function togglePause() {
  try {
    const st = await api("/api/status");
    if (st.state === "playing" || st.state === "loading") return control("pause");
    if (st.state === "paused") return control("resume");
    return { ok: true, status: st };
  } catch (e) {
    if (e instanceof OfflineError) setBadge("off");
    return { ok: false, offline: e instanceof OfflineError, error: e.message };
  }
}

async function seek(delta) {
  try {
    const ours = await getOurs();
    const st = await api("/api/status");
    if (!ours || st.session !== ours.session || !ACTIVE_STATES.includes(st.state)) {
      return { ok: false, error: "Not reading from this extension." };
    }
    const item = Math.max(0, Math.min(st.count - 1, st.item + delta));
    return { ok: true, status: await api("/api/seek", { method: "POST", body: { item } }) };
  } catch (e) {
    return { ok: false, offline: e instanceof OfflineError, error: e.message };
  }
}

async function fullStatus() {
  try {
    const info = await api("/api/info");
    const status = await api("/api/status");
    const ours = await getOurs();
    const isOurs = !!(ours && status.session === ours.session && ACTIVE_STATES.includes(status.state));
    if (isOurs) resumePolling();
    return { online: true, info, status, ours: isOurs, port: await getPort() };
  } catch (e) {
    return { online: false, error: e.message, port: await getPort() };
  }
}

// ---------------------------------------------------------------- wiring

const MENUS = [
  { id: "vibetts-selection", title: "Read selection", contexts: ["selection"], mode: "selection" },
  { id: "vibetts-from-here", title: "Read article from here", contexts: ["page", "selection"], mode: "from-here" },
  { id: "vibetts-article", title: "Read whole article", contexts: ["page", "selection"], mode: "article" },
];

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    for (const m of MENUS) {
      chrome.contextMenus.create({ id: m.id, title: m.title, contexts: m.contexts });
    }
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  const m = MENUS.find((x) => x.id === info.menuItemId);
  if (m && tab) readTab(tab.id, m.mode, tab);
});

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command === "read") {
    const id = tab ? tab.id : await activeTabId();
    readTab(id, "auto", tab);
  } else if (command === "toggle-pause") {
    togglePause();
  } else if (command === "stop") {
    control("stop");
  }
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  const ours = await getOurs();
  if (ours && ours.tabId === tabId) {
    await setOurs(null);
    control("stop");
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const run = async () => {
    switch (msg && msg.cmd) {
      case "status":
        return fullStatus();
      case "read":
        return readTab(msg.tabId != null ? msg.tabId : await activeTabId(), msg.mode || "from-here");
      case "pause":
      case "resume":
      case "stop":
        return control(msg.cmd);
      case "seek":
        return seek(Number(msg.delta) || 0);
      case "voices":
        try {
          return { ok: true, data: await api("/api/voices") };
        } catch (e) {
          return { ok: false, offline: e instanceof OfflineError, error: e.message };
        }
      default:
        return { ok: false, error: "unknown command" };
    }
  };
  run().then(sendResponse, (e) => sendResponse({ ok: false, error: String(e && e.message) }));
  return true;
});

chrome.runtime.onStartup.addListener(resumePolling);
resumePolling();

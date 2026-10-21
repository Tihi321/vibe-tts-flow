// VibeTTS Reader popup. All app traffic goes through the background service worker.

const $ = (id) => document.getElementById(id);
const els = {
  dot: $("dot"), conn: $("conn"), state: $("state"), error: $("error"),
  readPage: $("read-page"), readSel: $("read-selection"),
  prev: $("prev"), toggle: $("toggle"), stop: $("stop"), next: $("next"),
  voice: $("voice"), lang: $("lang"), speed: $("speed"), speedOut: $("speed-out"), port: $("port"),
};

let online = false;
let voicesLoaded = false;
let lastStatus = null;
let voiceData = { voices: [], languages: [] };

const send = (msg) =>
  chrome.runtime.sendMessage(msg).catch((e) => ({ ok: false, error: String((e && e.message) || e) }));

function showError(text) {
  els.error.textContent = text || "";
  els.error.hidden = !text;
}

const STATE_LABELS = {
  idle: "Idle", loading: "Loading...", playing: "Playing", paused: "Paused",
  finished: "Finished", stopped: "Stopped", error: "Error",
};

function render(res) {
  online = !!res.online;
  lastStatus = res;
  els.dot.className = "dot " + (online ? "on" : "off");
  if (document.activeElement !== els.port) els.port.value = res.port;
  if (!online) {
    els.conn.textContent = "VibeTTS is not running — start it from the Start menu/tray";
    els.state.textContent = "";
  } else {
    els.conn.textContent = "Connected to VibeTTS " + ((res.info && res.info.version) || "");
    const st = res.status || {};
    let text = STATE_LABELS[st.state] || "";
    if (res.ours && st.count) text += " — sentence " + (st.item + 1) + " of " + st.count;
    else if (st.state && st.state !== "idle") text += res.ours ? "" : " (not from this page)";
    els.state.textContent = text;
    if (st.state === "error" && st.error) showError(st.error);
  }
  const active = online && res.ours;
  els.readPage.disabled = !online;
  els.readSel.disabled = !online;
  els.toggle.disabled = !online || !(res.status && ["loading", "playing", "paused"].includes(res.status.state));
  els.stop.disabled = els.toggle.disabled;
  els.prev.disabled = !active;
  els.next.disabled = !active;
  els.voice.disabled = !online;
  els.lang.disabled = !online;
  els.toggle.innerHTML = res.status && res.status.state === "paused" ? "&#9654;&#65039;" : "&#9208;&#65039;";
  if (online && !voicesLoaded) loadVoices();
}

async function refresh() {
  render(await send({ cmd: "status" }));
}

async function loadVoices() {
  voicesLoaded = true;
  const res = await send({ cmd: "voices" });
  if (!res || !res.ok) {
    voicesLoaded = false;
    return;
  }
  const saved = await chrome.storage.sync.get(["voice", "lang"]);
  for (const v of res.data.voices || []) {
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = v.label || v.id;
    els.voice.appendChild(o);
  }
  voiceData = res.data;
  els.voice.value = saved.voice || "";
  if (els.voice.value !== (saved.voice || "")) els.voice.value = "";
  fillLangs(saved.lang || "");
}

// Offer only the languages the selected voice supports (Auto voice -> all); reset stored lang if unsupported.
function fillLangs(wanted) {
  const v = (voiceData.voices || []).find((x) => x.id === els.voice.value);
  const codes = v && Array.isArray(v.languages) && v.languages.length ? v.languages : null;
  els.lang.replaceChildren();
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = "Auto";
  els.lang.appendChild(auto);
  for (const l of voiceData.languages || []) {
    if (codes && !codes.includes(l.code)) continue;
    const o = document.createElement("option");
    o.value = l.code;
    o.textContent = l.name || l.code;
    els.lang.appendChild(o);
  }
  els.lang.value = wanted;
  if (els.lang.value !== wanted) {
    els.lang.value = "";
    chrome.storage.sync.set({ lang: "" });
  }
}

async function activeTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab ? tab.id : null;
}

async function readFromPopup(mode) {
  showError("");
  const res = await send({ cmd: "read", mode, tabId: await activeTabId() });
  if (!res.ok) showError(res.error || "Could not start reading.");
  refresh();
}

async function action(msg) {
  showError("");
  const res = await send(msg);
  if (!res.ok && res.error) showError(res.error);
  refresh();
}

els.readPage.addEventListener("click", () => readFromPopup("from-here"));
els.readSel.addEventListener("click", () => readFromPopup("selection"));
els.toggle.addEventListener("click", () => {
  const paused = lastStatus && lastStatus.status && lastStatus.status.state === "paused";
  action({ cmd: paused ? "resume" : "pause" });
});
els.stop.addEventListener("click", () => action({ cmd: "stop" }));
els.prev.addEventListener("click", () => action({ cmd: "seek", delta: -1 }));
els.next.addEventListener("click", () => action({ cmd: "seek", delta: 1 }));

els.voice.addEventListener("change", () => {
  chrome.storage.sync.set({ voice: els.voice.value });
  fillLangs(els.lang.value);
});
els.lang.addEventListener("change", () => chrome.storage.sync.set({ lang: els.lang.value }));
els.speed.addEventListener("input", () => {
  els.speedOut.textContent = Number(els.speed.value).toFixed(2);
});
els.speed.addEventListener("change", () => chrome.storage.sync.set({ speed: Number(els.speed.value) }));
els.port.addEventListener("change", async () => {
  const n = Number(els.port.value);
  if (Number.isInteger(n) && n > 0 && n < 65536) {
    await chrome.storage.local.set({ port: n });
    voicesLoaded = false;
    refresh();
  } else {
    showError("Enter a port between 1 and 65535.");
  }
});

(async () => {
  const saved = await chrome.storage.sync.get(["speed"]);
  const speed = Number(saved.speed);
  if (speed >= 0.5 && speed <= 2) {
    els.speed.value = String(speed);
    els.speedOut.textContent = speed.toFixed(2);
  }
  await refresh();
  setInterval(refresh, 1000);
})();

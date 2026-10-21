// VibeTTS web reader: library + book reader driven by the local Reader API.
import * as api from "./api.js";

const Seg = globalThis.VibeTTSSegment;
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ------------------------------------------------------------------ icons */

const svg = (d, extra = "") =>
  `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false" ${extra}>${d}</svg>`;
const ICONS = {
  play: svg('<path d="M7 4.500v15l12-7.500z" fill="currentColor" stroke="none"/>'),
  pause: svg('<rect x="6" y="4.500" width="4" height="15" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="4.500" width="4" height="15" rx="1" fill="currentColor" stroke="none"/>'),
  spinner: svg('<path d="M12 3a9 9 0 1 0 9 9" class="spin"/>'),
  prevSent: svg('<path d="M15 6l-6 6 6 6"/>'),
  nextSent: svg('<path d="M9 6l6 6-6 6"/>'),
  prevChap: svg('<path d="M18 6l-6 6 6 6"/><path d="M11 6l-6 6 6 6"/>'),
  nextChap: svg('<path d="M6 6l6 6-6 6"/><path d="M13 6l6 6-6 6"/>'),
  stop: svg('<rect x="6" y="6" width="12" height="12" rx="1.500" fill="currentColor" stroke="none"/>'),
  menu: svg('<path d="M4 7h16M4 12h16M4 17h10"/>'),
  back: svg('<path d="M15 5l-7 7 7 7"/>'),
  system: svg('<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>'),
  light: svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.900 4.900l1.400 1.400M17.700 17.700l1.400 1.400M2 12h2M20 12h2M4.900 19.100l1.400-1.400M17.700 6.300l1.400-1.400"/>'),
  dark: svg('<path d="M20 14.500A8 8 0 0 1 9.500 4a8 8 0 1 0 10.500 10.500z"/>'),
};

/* --------------------------------------------------------------- settings */

const DEFAULTS = { voice: "", lang: "", speed: 1, font: 19, theme: "system" };
const THEMES = ["system", "light", "dark"];
const SETTINGS_KEY = "vibetts.reader";

function loadSettings() {
  let raw = {};
  try {
    raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {};
  } catch (e) {
    raw = {};
  }
  const s = { ...DEFAULTS, ...raw };
  s.voice = typeof s.voice === "string" ? s.voice : "";
  s.lang = typeof s.lang === "string" ? s.lang : "";
  s.speed = clamp(Number(s.speed) || 1, 0.5, 2);
  s.font = clamp(Math.round(Number(s.font) || 19), 14, 32);
  if (!THEMES.includes(s.theme)) s.theme = "system";
  return s;
}
const settings = loadSettings();
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (e) {
    /* storage unavailable: settings last for this page only */
  }
}

function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

/* ------------------------------------------------------------------ state */

const st = {
  openToken: 0,
  loadToken: 0,
  book: null, // summary + toc
  chapterIdx: 0,
  items: [], // [{text, pause}] for the current chapter
  spans: [], // span element per item
  cursor: 0,
  marked: null, // span currently marked
  status: null,
  offline: false,
  offlineMsg: "",
  mySession: null,
  sessionChapter: -1,
  starting: false,
  finishedHandled: null,
  voices: null,
  navigating: false,
};

const isActiveState = (s) => s === "loading" || s === "playing" || s === "paused";
const sameSession = () => !!(st.status && st.status.session && st.status.session === st.mySession);
const mineActive = () =>
  sameSession() && st.sessionChapter === st.chapterIdx && isActiveState(st.status.state);
const elsewhere = () => !!(st.status && isActiveState(st.status.state) && !sameSession() && !st.starting);
const tocLength = () => (st.book ? st.book.toc.length : 0);

/* ---------------------------------------------------------------- helpers */

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

function showMsg(node, text) {
  node.textContent = text || "";
  node.hidden = !text;
}

let readerMsgTimer = 0;
function readerError(e) {
  const msg = typeof e === "string" ? e : (e && e.message) || "Something went wrong";
  showMsg($("#reader-msg"), msg);
  clearTimeout(readerMsgTimer);
  readerMsgTimer = setTimeout(() => showMsg($("#reader-msg"), ""), 8000);
}
function clearReaderError() {
  clearTimeout(readerMsgTimer);
  showMsg($("#reader-msg"), "");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------ theme */

function applyTheme() {
  const root = document.documentElement;
  if (settings.theme === "system") delete root.dataset.theme;
  else root.dataset.theme = settings.theme;
  const label = `Theme: ${settings.theme}`;
  for (const b of $$(".theme-btn")) {
    b.innerHTML = ICONS[settings.theme];
    b.setAttribute("aria-label", label);
    b.title = label;
  }
}
function cycleTheme() {
  settings.theme = THEMES[(THEMES.indexOf(settings.theme) + 1) % THEMES.length];
  saveSettings();
  applyTheme();
}

function applyFont() {
  document.documentElement.style.setProperty("--reader-size", `${settings.font}px`);
  $("#font-dec").disabled = settings.font <= 14;
  $("#font-inc").disabled = settings.font >= 32;
}

/* ----------------------------------------------------------------- banner */

function updateBanner() {
  const b = $("#banner");
  if (st.offline) {
    b.textContent = st.offlineMsg || "VibeTTS is not running. Start the app and this page will reconnect on its own.";
    b.hidden = false;
  } else {
    b.hidden = true;
  }
}

/* ---------------------------------------------------------------- routing */

function showView(name) {
  $("#view-library").hidden = name !== "library";
  $("#view-reader").hidden = name !== "reader";
  document.body.dataset.view = name;
  if (name !== "reader") document.body.classList.remove("toc-open");
}

function route() {
  const m = /^#\/book\/([0-9a-f]{16})\/?$/.exec(location.hash);
  if (m) openBook(m[1]);
  else showLibrary();
}

/* ---------------------------------------------------------------- library */

function bookCard(b) {
  const pos = b.position || { chapter: 0, item: 0 };
  const chapter = clamp((pos.chapter || 0) + 1, 1, Math.max(1, b.chapters || 1));
  const meta = el(
    "span",
    { class: "book-meta" },
    el("span", { class: "fmt", text: String(b.format || "").toUpperCase() }),
    el("span", { text: `Chapter ${chapter} of ${b.chapters}` })
  );
  const main = el(
    "a",
    { class: "book-main", href: `#/book/${b.id}` },
    el("span", { class: "book-name", text: b.title || "Untitled" }),
    el("span", { class: "book-by", text: b.author || "Unknown author" }),
    meta
  );
  const actions = el("div", { class: "book-actions" });
  const del = el("button", { type: "button", class: "btn btn-quiet btn-danger", text: "Delete", "aria-label": `Delete ${b.title || "book"}` });
  let timer = 0;
  const reset = () => {
    clearTimeout(timer);
    actions.replaceChildren(del);
  };
  del.addEventListener("click", () => {
    const yes = el("button", { type: "button", class: "btn btn-danger-solid", text: "Delete for good" });
    const no = el("button", { type: "button", class: "btn btn-quiet", text: "Cancel" });
    no.addEventListener("click", reset);
    yes.addEventListener("click", async () => {
      yes.disabled = no.disabled = true;
      try {
        await api.deleteBook(b.id);
        showMsg($("#lib-msg"), "");
        await refreshLibrary();
      } catch (e) {
        showMsg($("#lib-msg"), e.message);
        reset();
      }
    });
    actions.replaceChildren(yes, no);
    yes.focus();
    timer = setTimeout(reset, 6000);
  });
  actions.append(del);
  return el("li", { class: "book" }, main, actions);
}

async function refreshLibrary() {
  try {
    const list = await api.books();
    const ul = $("#book-list");
    ul.replaceChildren(...list.map(bookCard));
    $("#lib-empty").hidden = list.length > 0;
  } catch (e) {
    if (!e.offline) showMsg($("#lib-msg"), e.message);
  }
}

function showLibrary() {
  flushPosition();
  st.openToken++;
  st.book = null;
  showView("library");
  document.title = "VibeTTS Reader";
  refreshLibrary();
}

async function handleFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const zone = $("#dropzone");
  const text = $("#dz-text");
  zone.classList.add("busy");
  zone.setAttribute("aria-busy", "true");
  $("#pick-btn").disabled = true;
  const errors = [];
  let added = 0;
  for (let i = 0; i < files.length; i++) {
    text.textContent = `Adding ${files[i].name}${files.length > 1 ? ` (${i + 1} of ${files.length})` : ""}...`;
    try {
      await api.uploadBook(files[i]);
      added++;
      await refreshLibrary();
    } catch (e) {
      errors.push(`${files[i].name}: ${e.message}`);
      if (e.offline) break;
    }
  }
  zone.classList.remove("busy");
  zone.removeAttribute("aria-busy");
  $("#pick-btn").disabled = false;
  text.textContent = added && !errors.length ? "Added. Drop another book here" : "Drop a book here";
  showMsg($("#lib-msg"), errors.join("\n"));
}

function initLibrary() {
  const zone = $("#dropzone");
  const input = $("#file-input");
  $("#pick-btn").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const files = Array.from(input.files || []);
    input.value = "";
    handleFiles(files);
  });
  let depth = 0;
  zone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    depth++;
    zone.classList.add("over");
  });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  zone.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) zone.classList.remove("over");
  });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    zone.classList.remove("over");
    handleFiles(e.dataTransfer && e.dataTransfer.files);
  });
  // A file dropped outside the zone must not navigate the tab away from the reader.
  for (const t of ["dragover", "drop"]) window.addEventListener(t, (e) => e.preventDefault());
}

/* ----------------------------------------------------------------- reader */

function clearReader() {
  st.chapterIdx = 0;
  st.items = [];
  st.spans = [];
  st.marked = null;
  st.cursor = 0;
  $("#chapter").replaceChildren();
  $("#toc-list").replaceChildren();
  $("#chapter-nav").hidden = true;
  $("#book-title").textContent = "";
  $("#book-author").textContent = "";
  clearReaderError();
}

async function openBook(id) {
  showView("reader");
  if (st.book && st.book.id === id) return;
  flushPosition();
  const token = ++st.openToken;
  st.book = null;
  clearReader();
  updateControls();
  try {
    const book = await api.getBook(id);
    if (token !== st.openToken) return;
    st.book = book;
    $("#book-title").textContent = book.title || "Untitled";
    $("#book-author").textContent = book.author || "";
    renderToc();
    const pos = book.position || { chapter: 0, item: 0 };
    const ch = clamp(pos.chapter | 0, 0, Math.max(0, book.toc.length - 1));
    await loadChapter(ch, Math.max(0, pos.item | 0), { restore: true });
    if (st.status) applyStatus(st.status);
  } catch (e) {
    if (token !== st.openToken) return;
    showUnavailable(e);
  }
}

function showUnavailable(e) {
  const missing = e && e.status === 404;
  $("#chapter").replaceChildren(
    el("p", { class: "empty", text: missing ? "This book is no longer in your library." : e.message }),
    el("p", {}, el("a", { class: "btn", href: "#/", text: "Back to the library" }))
  );
}

function renderToc() {
  const list = $("#toc-list");
  list.replaceChildren(
    ...st.book.toc.map((t, i) => {
      const b = el("button", { type: "button", class: "toc-item", text: t.title || `Chapter ${i + 1}`, "data-ch": i });
      return el("li", {}, b);
    })
  );
}

function markToc() {
  for (const b of $$(".toc-item")) {
    const on = Number(b.dataset.ch) === st.chapterIdx;
    b.classList.toggle("on", on);
    if (on) b.setAttribute("aria-current", "true");
    else b.removeAttribute("aria-current");
  }
  const cur = $(".toc-item.on");
  const box = $("#toc");
  if (cur && box.scrollHeight > box.clientHeight) {
    box.scrollTop = Math.max(0, cur.offsetTop - box.clientHeight / 3);
  }
}

/** Build the speak items for a chapter and the DOM that mirrors them. */
function buildChapter(ch, n) {
  const title = (ch.title || "").trim() || `Chapter ${n + 1}`;
  const items = [{ text: title, pause: "paragraph" }];
  const frag = document.createDocumentFragment();
  const spans = [];

  const h = el("h2", { class: "chapter-title" });
  const hs = el("span", { class: "s", "data-i": 0, text: title });
  h.append(hs);
  spans.push(hs);
  frag.append(h);

  for (const para of ch.paragraphs || []) {
    const p = el("p");
    const sents = Seg.sentences(para);
    if (!sents.length) {
      p.textContent = para;
      frag.append(p);
      continue;
    }
    let pos = 0;
    sents.forEach((s, k) => {
      if (s.start > pos) p.append(document.createTextNode(para.slice(pos, s.start)));
      const idx = items.length;
      items.push({ text: s.text, pause: k === sents.length - 1 ? "paragraph" : "sentence" });
      const sp = el("span", { class: "s", "data-i": idx, text: s.text });
      spans.push(sp);
      p.append(sp);
      pos = s.end;
    });
    if (pos < para.length) p.append(document.createTextNode(para.slice(pos)));
    frag.append(p);
  }
  return { items, spans, frag };
}

/** Load chapter `n` into the page and put the cursor on `item`. Resolves false if superseded or failed. */
async function loadChapter(n, item = 0, { restore = false } = {}) {
  if (!st.book) return false;
  const bookId = st.book.id;
  const token = ++st.loadToken;
  let ch;
  try {
    ch = await api.getChapter(bookId, n);
  } catch (e) {
    if (token === st.loadToken) readerError(e);
    return false;
  }
  if (token !== st.loadToken || !st.book || st.book.id !== bookId) return false;
  const built = buildChapter(ch, n);
  st.chapterIdx = n;
  st.items = built.items;
  st.spans = built.spans;
  st.marked = null;
  $("#chapter").replaceChildren(built.frag);
  st.cursor = clamp(item, 0, st.items.length - 1);
  markToc();
  updateChapterNav();
  document.title = `${ch.title || "Chapter " + (n + 1)} - ${st.book.title || "VibeTTS Reader"}`;
  mark(st.cursor, mineActive());
  if (st.cursor === 0 && !restore) window.scrollTo(0, 0);
  else scrollToSpan(st.cursor, { smooth: false, force: true });
  if (!restore || st.cursor !== item) queuePosition(!restore);
  updateControls();
  return true;
}

function updateChapterNav() {
  const nav = $("#chapter-nav");
  nav.hidden = !st.book;
  $("#foot-prev").disabled = st.chapterIdx <= 0;
  $("#foot-next").disabled = st.chapterIdx >= tocLength() - 1;
}

/* marker / scrolling */

function mark(i, active) {
  if (st.marked) st.marked.classList.remove("current", "cursor");
  const sp = st.spans[i];
  st.marked = sp || null;
  if (sp) sp.classList.add(active ? "current" : "cursor");
}

function headerHeight() {
  const h = $(".sticky-head");
  return h ? h.getBoundingClientRect().height : 0;
}

function scrollToSpan(i, { smooth = true, force = false } = {}) {
  const sp = st.spans[i];
  if (!sp) return;
  const r = sp.getBoundingClientRect();
  const top = headerHeight() + 12;
  const bottom = window.innerHeight - 24;
  const visible = r.top >= top && r.bottom <= bottom;
  if (visible && !force) return;
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const target = window.scrollY + r.top - (headerHeight() + (window.innerHeight - headerHeight()) * 0.3);
  window.scrollTo({ top: Math.max(0, target), behavior: smooth && !reduce ? "smooth" : "auto" });
}

function setCursor(i, { active = false, scroll = true } = {}) {
  i = clamp(i, 0, Math.max(0, st.items.length - 1));
  const changed = i !== st.cursor;
  st.cursor = i;
  mark(i, active);
  if (scroll) scrollToSpan(i);
  if (changed) queuePosition(false);
}

/* position saving */

let posTimer = 0;
let pendingPos = null;
function queuePosition(immediate) {
  if (!st.book) return;
  pendingPos = { id: st.book.id, chapter: st.chapterIdx, item: st.cursor };
  clearTimeout(posTimer);
  if (immediate) flushPosition();
  else posTimer = setTimeout(flushPosition, 1500);
}
function flushPosition(keepalive = false) {
  clearTimeout(posTimer);
  const p = pendingPos;
  pendingPos = null;
  if (!p) return;
  api.setPosition(p.id, p.chapter, p.item, { keepalive }).catch(() => {});
}

/* playback actions */

function speakBody(item) {
  const body = {
    items: st.items,
    start: item,
    source: "reader",
    meta: { book: st.book.id, chapter: st.chapterIdx },
    speed: settings.speed,
  };
  if (settings.voice) body.voice = settings.voice;
  if (settings.lang && langSupported(settings.voice, settings.lang)) body.lang = settings.lang;
  return body;
}

async function startAt(item) {
  if (!st.book || !st.items.length || st.starting) return;
  item = clamp(item, 0, st.items.length - 1);
  st.starting = true;
  clearReaderError();
  setCursor(item, { active: false, scroll: false });
  updateControls();
  const chapter = st.chapterIdx;
  try {
    const r = await api.speak(speakBody(item));
    st.mySession = r.session;
    st.sessionChapter = chapter;
    st.finishedHandled = null;
  } catch (e) {
    readerError(e);
  } finally {
    st.starting = false;
  }
  if (st.status) applyStatus(st.status);
  else updateControls();
}

async function call(fn) {
  try {
    const s = await fn();
    if (s && typeof s.version === "number" && !st.status) st.status = s;
    return s;
  } catch (e) {
    readerError(e);
    return null;
  }
}

async function togglePlay() {
  if (!st.book) return;
  if (mineActive()) {
    await call(st.status.state === "paused" ? api.resume : api.pause);
  } else {
    await startAt(st.cursor);
  }
}

async function stopPlayback() {
  if (!mineActive()) return;
  const item = st.status.item;
  if (typeof item === "number" && item >= 0 && item < st.items.length) setCursor(item, { active: false, scroll: false });
  await call(api.stop);
}

async function moveSentence(delta) {
  if (!st.book) return;
  const target = st.cursor + delta;
  if (mineActive()) {
    const cur = st.status.item;
    const t = (typeof cur === "number" ? cur : st.cursor) + delta;
    if (t < 0) return;
    if (t >= st.items.length) {
      if (delta > 0) await goChapter(st.chapterIdx + 1, { play: true });
      return;
    }
    await call(() => api.seek(t));
    return;
  }
  if (target < 0) {
    if (delta < 0 && st.chapterIdx > 0) await goChapter(st.chapterIdx - 1, { play: false });
    return;
  }
  if (target >= st.items.length) {
    if (delta > 0 && st.chapterIdx < tocLength() - 1) await goChapter(st.chapterIdx + 1, { play: false });
    return;
  }
  setCursor(target, { active: false });
}

async function goChapter(n, { play = mineActive(), item = 0 } = {}) {
  if (!st.book || n < 0 || n >= tocLength() || st.navigating) return;
  st.navigating = true;
  try {
    const ok = await loadChapter(n, item);
    if (ok && play) await startAt(item);
  } finally {
    st.navigating = false;
  }
}

async function clickSentence(i) {
  if (!st.book) return;
  if (mineActive()) {
    setCursor(i, { active: true, scroll: false });
    await call(() => api.seek(i));
  } else {
    await startAt(i);
  }
}

/** Voice/speed changed mid-playback: restart from the sentence being read so it takes effect. */
function restartIfPlaying() {
  if (mineActive() && st.status.state === "playing" && !st.starting) {
    const item = typeof st.status.item === "number" ? st.status.item : st.cursor;
    startAt(item);
  }
}

/* status handling */

function applyStatus(s) {
  st.status = s;
  const sameSess = sameSession();

  // Pick up a reader session for this book that this page did not start (e.g. after a reload).
  if (
    !sameSess &&
    st.book &&
    !st.starting &&
    !st.navigating &&
    isActiveState(s.state) &&
    s.source === "reader" &&
    s.meta &&
    s.meta.book === st.book.id &&
    Number.isInteger(s.meta.chapter) &&
    s.meta.chapter >= 0 &&
    s.meta.chapter < tocLength()
  ) {
    st.mySession = s.session;
    st.sessionChapter = s.meta.chapter;
    st.finishedHandled = null;
    if (s.meta.chapter !== st.chapterIdx) {
      goChapter(s.meta.chapter, { play: false, item: typeof s.item === "number" ? s.item : 0 });
    }
    return applyStatus(s);
  }

  const active = mineActive();
  if (active && typeof s.item === "number" && s.item >= 0 && s.item < st.items.length) {
    setCursor(s.item, { active: true });
  } else if (st.marked) {
    mark(st.cursor, false);
  }

  if (sameSess && s.state === "error" && s.error) readerError(s.error);

  if (
    sameSess &&
    s.state === "finished" &&
    st.sessionChapter === st.chapterIdx &&
    st.finishedHandled !== s.session &&
    !st.starting
  ) {
    st.finishedHandled = s.session;
    if (st.chapterIdx < tocLength() - 1) goChapter(st.chapterIdx + 1, { play: true });
  }
  updateControls();
}

async function pollLoop() {
  let since;
  let delay = 1000;
  for (;;) {
    try {
      const s = await api.status(since, 20);
      if (st.offline) {
        st.offline = false;
        updateBanner();
        ensureVoices();
        if (document.body.dataset.view === "library") refreshLibrary();
      }
      delay = 1000;
      since = s.version;
      applyStatus(s);
    } catch (e) {
      since = undefined;
      st.offline = true;
      st.offlineMsg = e.offline ? "" : e.message;
      updateBanner();
      await sleep(delay);
      delay = Math.min(delay * 2, 5000);
    }
  }
}

/* controls */

function setIcon(btn, key, label) {
  if (btn.dataset.icon !== key) {
    btn.innerHTML = ICONS[key];
    btn.dataset.icon = key;
  }
  btn.setAttribute("aria-label", label);
}

function updateControls() {
  const hasBook = !!st.book && st.items.length > 0;
  const play = $("#play");
  const state = st.status && sameSession() && st.sessionChapter === st.chapterIdx ? st.status.state : null;
  if (st.starting || state === "loading") {
    setIcon(play, "spinner", "Loading");
    play.title = "Loading";
  } else if (state === "playing") {
    setIcon(play, "pause", "Pause");
    play.title = "Pause (Space)";
  } else if (state === "paused") {
    setIcon(play, "play", "Resume");
    play.title = "Resume (Space)";
  } else {
    setIcon(play, "play", "Play");
    play.title = "Play (Space)";
  }
  play.disabled = !hasBook;
  const idx = st.chapterIdx;
  $("#prev-chap").disabled = !hasBook || idx <= 0;
  $("#next-chap").disabled = !hasBook || idx >= tocLength() - 1;
  $("#stop").disabled = !mineActive();
  $("#prev-sent").disabled = !hasBook;
  $("#next-sent").disabled = !hasBook;
  $("#elsewhere").hidden = !(st.book && elsewhere());
  $("#lang").disabled = !st.voices;
  $("#speed-out").textContent = `${settings.speed.toFixed(2)}x`;
}

// Language codes the voice can read, or null for "all" (Auto voice / unknown voice).
function voiceLangs(voiceId) {
  const v = voiceId && st.voices && (st.voices.voices || []).find((x) => x.id === voiceId);
  return v && Array.isArray(v.languages) && v.languages.length ? v.languages : null;
}
function langSupported(voiceId, code) {
  const codes = voiceLangs(voiceId);
  return !codes || codes.includes(code);
}
// Offer only the languages the selected voice supports; reset to Auto if the current one isn't.
function fillLangs() {
  const lsel = $("#lang");
  const codes = voiceLangs(settings.voice);
  lsel.replaceChildren(el("option", { value: "", text: "Auto" }));
  for (const l of st.voices.languages || []) {
    if (!codes || codes.includes(l.code)) lsel.append(el("option", { value: l.code, text: l.name || l.code }));
  }
  if (settings.lang && !langSupported(settings.voice, settings.lang)) settings.lang = "";
  lsel.value = settings.lang;
}

async function ensureVoices() {
  if (st.voices) return;
  try {
    st.voices = await api.voices();
  } catch (e) {
    return;
  }
  const vsel = $("#voice");
  vsel.replaceChildren(el("option", { value: "", text: "Auto" }));
  const groups = new Map();
  for (const v of st.voices.voices || []) {
    const key = v.engine || "other";
    if (!groups.has(key)) groups.set(key, el("optgroup", { label: key.charAt(0).toUpperCase() + key.slice(1) }));
    groups.get(key).append(el("option", { value: v.id, text: v.label || v.id }));
  }
  vsel.append(...groups.values());

  if (settings.voice && !(st.voices.voices || []).some((v) => v.id === settings.voice)) settings.voice = "";
  if (settings.lang && !(st.voices.languages || []).some((l) => l.code === settings.lang)) settings.lang = "";
  vsel.value = settings.voice;
  fillLangs();
  saveSettings();
  updateControls();
}

function initReader() {
  $("#play").addEventListener("click", togglePlay);
  $("#stop").addEventListener("click", stopPlayback);
  $("#prev-sent").addEventListener("click", () => moveSentence(-1));
  $("#next-sent").addEventListener("click", () => moveSentence(1));
  $("#prev-chap").addEventListener("click", () => goChapter(st.chapterIdx - 1));
  $("#next-chap").addEventListener("click", () => goChapter(st.chapterIdx + 1));
  $("#foot-prev").addEventListener("click", () => goChapter(st.chapterIdx - 1));
  $("#foot-next").addEventListener("click", () => goChapter(st.chapterIdx + 1));

  $("#chapter").addEventListener("click", (e) => {
    const sp = e.target.closest && e.target.closest(".s");
    if (!sp) return;
    const sel = window.getSelection && window.getSelection();
    if (sel && !sel.isCollapsed && sel.toString().trim()) return; // the user is selecting text
    clickSentence(Number(sp.dataset.i));
  });

  $("#toc-list").addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest(".toc-item");
    if (!b) return;
    document.body.classList.remove("toc-open");
    syncTocToggle();
    goChapter(Number(b.dataset.ch));
  });
  $("#toc-toggle").addEventListener("click", () => {
    document.body.classList.toggle("toc-open");
    syncTocToggle();
  });
  $("#toc-scrim").addEventListener("click", () => {
    document.body.classList.remove("toc-open");
    syncTocToggle();
  });

  const vsel = $("#voice");
  const lsel = $("#lang");
  const speed = $("#speed");
  speed.value = String(settings.speed);
  vsel.addEventListener("change", () => {
    settings.voice = vsel.value;
    fillLangs();
    saveSettings();
    updateControls();
    restartIfPlaying();
  });
  lsel.addEventListener("change", () => {
    settings.lang = lsel.value;
    saveSettings();
    restartIfPlaying();
  });
  speed.addEventListener("input", () => {
    settings.speed = clamp(Number(speed.value) || 1, 0.5, 2);
    $("#speed-out").textContent = `${settings.speed.toFixed(2)}x`;
  });
  speed.addEventListener("change", () => {
    saveSettings();
    restartIfPlaying();
  });

  $("#font-dec").addEventListener("click", () => {
    settings.font = clamp(settings.font - 1, 14, 32);
    saveSettings();
    applyFont();
    if (st.marked) scrollToSpan(st.cursor, { smooth: false, force: true });
  });
  $("#font-inc").addEventListener("click", () => {
    settings.font = clamp(settings.font + 1, 14, 32);
    saveSettings();
    applyFont();
    if (st.marked) scrollToSpan(st.cursor, { smooth: false, force: true });
  });

  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
      document.documentElement.style.setProperty("--head-h", `${Math.round(headerHeight())}px`);
    }).observe($(".sticky-head"));
  }
  document.addEventListener("keydown", onKey);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushPosition(true);
  });
  window.addEventListener("pagehide", () => flushPosition(true));
}

function syncTocToggle() {
  const open = document.body.classList.contains("toc-open");
  const b = $("#toc-toggle");
  b.setAttribute("aria-expanded", String(open));
  b.setAttribute("aria-label", open ? "Hide chapters" : "Show chapters");
}

function onKey(e) {
  if (document.body.dataset.view !== "reader") return;
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  const tag = t && t.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (t && t.isContentEditable)) return;
  if (e.key === " " || e.code === "Space") {
    if (tag === "BUTTON" || tag === "A") return; // the focused control handles Space itself
    e.preventDefault();
    togglePlay();
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    moveSentence(-1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    moveSentence(1);
  } else if (e.key === "Escape") {
    if (document.body.classList.contains("toc-open")) {
      document.body.classList.remove("toc-open");
      syncTocToggle();
    } else {
      stopPlayback();
    }
  }
}

/* ------------------------------------------------------------------- init */

function init() {
  if (!Seg || typeof Seg.sentences !== "function") {
    document.body.textContent = "segment.js failed to load.";
    return;
  }
  setIcon($("#prev-chap"), "prevChap", "Previous chapter");
  setIcon($("#prev-sent"), "prevSent", "Previous sentence");
  setIcon($("#next-sent"), "nextSent", "Next sentence");
  setIcon($("#next-chap"), "nextChap", "Next chapter");
  setIcon($("#play"), "play", "Play");
  setIcon($("#stop"), "stop", "Stop");
  $("#toc-toggle").innerHTML = ICONS.menu;
  $(".back-btn").innerHTML = ICONS.back;
  for (const b of $$(".theme-btn")) b.addEventListener("click", cycleTheme);
  applyTheme();
  applyFont();
  initLibrary();
  initReader();
  window.addEventListener("hashchange", route);
  route();
  ensureVoices();
  pollLoop();
}

init();

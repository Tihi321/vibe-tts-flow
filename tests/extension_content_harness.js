// Behavioural harness for extension/content.js (run by tests/test_extension.py with node).
// Loads segment.js + content.js into a vm context with a tiny fake DOM (three paragraphs of two
// sentences), drives messages and selections, and prints a JSON report of what happened.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const EXT = path.resolve(__dirname, "..", "extension");
const PARAS = ["One one. Two two.", "Three three. Four four.", "Five five. Six six."];

// Positions are (text node index, offset); Range compares them like the DOM does.
const textNodes = [];
const body = { nodeType: 1, tagName: "BODY", parentElement: null };
function el(tag, parent) {
  return {
    nodeType: 1,
    tagName: tag,
    parentElement: parent,
    hasAttribute: () => false,
    getAttribute: () => null,
    closest: () => null,
    get textContent() {
      return textNodes.filter((n) => n.parentElement === this).map((n) => n.data).join("");
    },
  };
}
const pEls = PARAS.map((text, i) => {
  const p = el("P", body);
  textNodes.push({ nodeType: 3, data: text, parentElement: p, idx: i });
  return p;
});

const cmp = (a, b) => (a.n - b.n) || (a.o - b.o);
class FakeRange {
  constructor() {
    this.s = { n: 0, o: 0 };
    this.e = { n: 0, o: 0 };
    this.startContainer = textNodes[0];
    this.endContainer = textNodes[0];
  }
  setStart(node, o) {
    this.s = { n: node.idx, o };
    this.startContainer = node;
    this.startOffset = o;
    if (cmp(this.s, this.e) > 0) this.e = this.s;
  }
  setEnd(node, o) {
    this.e = { n: node.idx, o };
    this.endContainer = node;
    this.endOffset = o;
  }
  collapse(toStart) {
    if (toStart) this.e = this.s;
    else this.s = this.e;
  }
  get collapsed() {
    return cmp(this.s, this.e) === 0;
  }
  cloneRange() {
    return Object.assign(new FakeRange(), this);
  }
  intersectsNode() {
    return true;
  }
  compareBoundaryPoints(how, src) {
    if (how === 1) return Math.sign(cmp(this.e, src.s)); // START_TO_END: this end vs source start
    if (how === 3) return Math.sign(cmp(this.s, src.e)); // END_TO_START: this start vs source end
    throw new Error("unused");
  }
  getClientRects() {
    return [];
  }
  getBoundingClientRect() {
    return { width: 0, height: 0 };
  }
}
FakeRange.START_TO_END = 1;
FakeRange.END_TO_START = 3;

const highlights = new Map();
const sent = [];
const timers = [];
const listeners = {};
let selection = null;
let onMessage = null;

const context = {
  console,
  Intl,
  Range: FakeRange,
  NodeFilter: { SHOW_ELEMENT: 1, SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3 },
  Highlight: class {
    constructor(r) {
      this.range = r;
    }
  },
  CSS: { highlights },
  getComputedStyle: () => ({ display: "block", visibility: "visible" }),
  setTimeout: (fn) => timers.push(fn),
  clearTimeout: (t) => {
    const i = timers.indexOf(t);
    if (i >= 0) timers.splice(i, 1);
  },
  chrome: {
    runtime: {
      id: "test",
      onMessage: { addListener: (fn) => (onMessage = fn) },
      sendMessage: (m) => {
        sent.push(m);
        return Promise.resolve({ ok: true });
      },
    },
  },
  document: {
    body,
    documentElement: { lang: "en" },
    head: { appendChild() {} },
    getElementById: () => null,
    createElement: () => ({}),
    createRange: () => new FakeRange(),
    querySelectorAll: () => [],
    addEventListener: (ev, fn) => (listeners[ev] = fn),
    createTreeWalker: (_root, _mask, filter) => {
      let i = -1;
      return {
        nextNode() {
          while (++i < textNodes.length) if (filter.acceptNode(textNodes[i]) === 1) return textNodes[i];
          return null;
        },
      };
    },
  },
  window: { getSelection: () => selection, innerHeight: 800 },
};
body.querySelectorAll = () => pEls;
context.globalThis = context;
vm.createContext(context);
for (const f of ["segment.js", "content.js"]) {
  vm.runInContext(fs.readFileSync(path.join(EXT, f), "utf-8"), context, { filename: f });
}

const send = (msg) => {
  let out;
  onMessage(msg, {}, (r) => (out = r));
  return out;
};
// select from (paragraph p, char o) to the end of paragraph p2
function select(p, o, p2 = p, o2 = PARAS[p2].length) {
  const r = new FakeRange();
  r.setStart(textNodes[p], o);
  r.setEnd(textNodes[p2], o2);
  selection = { rangeCount: 1, getRangeAt: () => r };
}
function collapseSelection() {
  const r = new FakeRange();
  r.setStart(textNodes[2], 3);
  r.collapse(true);
  selection = { rangeCount: 1, getRangeAt: () => r };
}
function settle() {
  listeners.selectionchange();
  while (timers.length) timers.shift()();
}
const queued = () => (highlights.has("vibetts-queued") ? ranges_index(highlights.get("vibetts-queued").range) : null);
let collected = null;
function ranges_index(range) {
  // item index of a highlighted range, derived from its start position
  return collected.starts.findIndex((s) => s.n === range.s.n && s.o === range.s.o);
}

selection = null;
const res = send({ cmd: "collect", mode: "article" });
// item start positions in text order: used to name items in the report
collected = { starts: [] };
for (let i = 0; i < PARAS.length; i++) {
  const text = PARAS[i];
  collected.starts.push({ n: i, o: 0 }, { n: i, o: text.indexOf(". ") + 2 });
}
const report = { count: res.items.length, pauses: res.items.map((i) => i.pause) };
const reset = () => (sent.length = 0);

// 1. selection before any progress is ignored
select(2, 0);
settle();
report.beforeProgress = { queued: queued(), sent: sent.length };

// 2. reading item 0; select the second sentence of paragraph 2 (item 5)
send({ cmd: "progress", item: 0 });
select(2, PARAS[2].indexOf(". ") + 2);
settle();
report.afterSelect = { queued: queued() };

// 3. collapsed selection does not cancel; progress inside the paragraph does not jump
collapseSelection();
settle();
send({ cmd: "progress", item: 1 });
report.insideParagraph = { queued: queued(), sent: sent.length };

// 4. paragraph boundary: jump requested once, queue cleared
send({ cmd: "progress", item: 2 });
report.atBoundary = { queued: queued(), sent: sent.slice() };
reset();

// 5. the progress caused by the jump itself does not trigger anything
send({ cmd: "progress", item: 5 });
report.afterJump = { sent: sent.length };

// 6. newer selection replaces the older one
send({ cmd: "progress", item: 0 });
select(1, 0);
settle();
select(2, 0);
settle();
report.replaced = { queued: queued() };

// 7. selection in the current item cancels the queue
select(0, 0, 0, 3);
settle();
report.inCurrent = { queued: queued() };

// 8. Escape cancels; no jump afterwards
select(2, 0);
settle();
listeners.keydown({ key: "Escape" });
send({ cmd: "progress", item: 1 });
send({ cmd: "progress", item: 2 });
report.escaped = { queued: queued(), sent: sent.length };

// 9. clear wipes a pending jump and the selection listener goes quiet
send({ cmd: "progress", item: 0 });
select(2, 0);
settle();
send({ cmd: "clear" });
report.cleared = { queued: queued() };
select(1, 0);
settle();
report.afterClear = { queued: queued() };

// 9b. a coalesced poll skips from item 1 to item 3 across a boundary: the jump still fires
send({ cmd: "collect", mode: "article" });
reset();
send({ cmd: "progress", item: 0 });
select(2, 0);
settle();
send({ cmd: "progress", item: 1 });
send({ cmd: "progress", item: 3 });
report.skipped = { queued: queued(), sent: sent.slice() };
reset();

// 10. a selection after the last item is outside the collected text
send({ cmd: "collect", mode: "article" });
send({ cmd: "progress", item: 0 });
const r = new FakeRange();
r.setStart(textNodes[2], PARAS[2].length);
r.setEnd(textNodes[2], PARAS[2].length);
r.e = { n: 3, o: 0 };
selection = { rangeCount: 1, getRangeAt: () => r };
settle();
report.outside = { queued: queued() };

console.log(JSON.stringify(report));

// VibeTTS Reader content script (classic script, injected on demand together with segment.js).
// Collects readable sentences from the page and highlights the one being read.
// Never modifies the page except for one <style> element; never logs page text.

(() => {
  // Injected again (e.g. a second read): keep the existing instance and its state. An orphaned
  // instance (extension reloaded, chrome.runtime.id gone) is replaced.
  if (globalThis.__vibettsReader && chrome.runtime && chrome.runtime.id) return;
  globalThis.__vibettsReader = true;

  const MAX_ITEMS = 20000;
  const MAX_CHARS = 500000;
  const HIGHLIGHT_NAME = "vibetts-current";
  const STYLE_ID = "vibetts-highlight-style";

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS", "NAV", "ASIDE", "HEADER", "FOOTER",
    "FORM", "BUTTON", "INPUT", "SELECT", "TEXTAREA", "IFRAME", "OBJECT", "AUDIO", "VIDEO", "HEAD", "OPTION",
  ]);
  const INLINE_DISPLAY = new Set([
    "inline", "inline-block", "inline-flex", "inline-grid", "inline-table", "contents", "ruby", "ruby-text",
    "ruby-base", "ruby-text-container", "ruby-base-container",
  ]);

  // state kept across messages: DOM ranges of the last collected items
  let ranges = [];
  let current = -1;

  // ------------------------------------------------------------ highlight

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = "::highlight(" + HIGHLIGHT_NAME + "){background-color: rgba(255, 213, 79, .55); color: inherit}";
    (document.head || document.documentElement).appendChild(st);
  }

  function clearHighlight() {
    current = -1;
    try {
      if (typeof CSS !== "undefined" && CSS.highlights) CSS.highlights.delete(HIGHLIGHT_NAME);
    } catch (e) {
      /* ignore */
    }
  }

  function scrollToRange(range) {
    const rects = range.getClientRects();
    const rect = rects.length ? rects[0] : range.getBoundingClientRect();
    if (!rect || (rect.width === 0 && rect.height === 0)) return;
    const vh = window.innerHeight || document.documentElement.clientHeight;
    if (rect.top >= 0 && rect.bottom <= vh) return;
    let el = range.startContainer;
    if (el.nodeType !== 1) el = el.parentElement;
    if (el && el.getBoundingClientRect().height <= vh * 0.8) {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    } else {
      window.scrollBy({ top: rect.top + rect.height / 2 - vh / 2, behavior: "smooth" });
    }
  }

  function highlight(index) {
    if (index < 0 || index >= ranges.length) return;
    current = index;
    const range = ranges[index];
    try {
      if (typeof CSS !== "undefined" && CSS.highlights && typeof Highlight === "function") {
        ensureStyle();
        CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(range));
      }
      scrollToRange(range);
    } catch (e) {
      console.warn("VibeTTS: highlight failed", e);
    }
  }

  // ------------------------------------------------------------ collecting

  const displayCache = new Map();

  function displayOf(el) {
    let d = displayCache.get(el);
    if (d === undefined) {
      try {
        d = getComputedStyle(el).display;
      } catch (e) {
        d = "block";
      }
      displayCache.set(el, d);
    }
    return d;
  }

  function isHidden(el) {
    if (SKIP_TAGS.has(el.tagName.toUpperCase())) return true;
    if (el.hasAttribute("hidden")) return true;
    if (el.getAttribute("aria-hidden") === "true") return true;
    if (el.getAttribute("role") === "navigation") return true;
    let cs;
    try {
      cs = getComputedStyle(el);
    } catch (e) {
      return false;
    }
    displayCache.set(el, cs.display);
    return cs.display === "none" || cs.visibility === "hidden" || cs.visibility === "collapse";
  }

  function blockOf(node, stopAt) {
    let el = node.parentElement;
    while (el && el !== stopAt && INLINE_DISPLAY.has(displayOf(el))) el = el.parentElement;
    return el || stopAt;
  }

  function lengthOfText(el) {
    return (el.textContent || "").length;
  }

  function findRoot(selRange) {
    const articleSel = "article, main, [role=main], [itemprop=articleBody]";
    if (selRange) {
      let el = selRange.startContainer;
      if (el.nodeType !== 1) el = el.parentElement;
      const hit = el && el.closest(articleSel);
      if (hit) return hit;
    }
    const cands = Array.from(document.querySelectorAll(articleSel));
    if (cands.length) {
      let best = cands[0];
      for (const c of cands) if (lengthOfText(c) > lengthOfText(best)) best = c;
      if (lengthOfText(best) > 200) return best;
    }
    // element holding the most text inside <p> descendants
    const score = new Map();
    for (const p of document.body.querySelectorAll("p")) {
      const len = (p.textContent || "").trim().length;
      if (!len) continue;
      const parent = p.parentElement;
      if (!parent) continue;
      score.set(parent, (score.get(parent) || 0) + len);
      const grand = parent.parentElement;
      if (grand) score.set(grand, (score.get(grand) || 0) + len / 2);
    }
    let best = null;
    let bestScore = 0;
    for (const [el, s] of score) {
      if (s > bestScore) {
        best = el;
        bestScore = s;
      }
    }
    return best || document.body;
  }

  // Text segments {node, from, to} grouped into blocks: [[seg, ...], ...]
  function collectBlocks(root, selRange) {
    const blocks = [];
    let group = null;
    let groupBlock = null;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (n.nodeType === 1) {
          if (n.tagName === "BR") return NodeFilter.FILTER_ACCEPT;
          return isHidden(n) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_SKIP;
        }
        return n.data.length ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (n.nodeType === 1) {
        group = null; // <br> ends the running group
        groupBlock = null;
        continue;
      }
      let from = 0;
      let to = n.data.length;
      if (selRange) {
        if (!selRange.intersectsNode(n)) continue;
        if (n === selRange.startContainer) from = selRange.startOffset;
        if (n === selRange.endContainer) to = selRange.endOffset;
        if (to <= from) continue;
      }
      const block = blockOf(n, root.parentElement || root);
      if (!group || block !== groupBlock) {
        group = [];
        groupBlock = block;
        blocks.push(group);
      }
      group.push({ node: n, from, to });
    }
    return blocks;
  }

  function docLang() {
    const l = document.documentElement.lang;
    if (!l) return undefined;
    try {
      new Intl.Segmenter(l);
      return l;
    } catch (e) {
      return undefined;
    }
  }

  function collect(mode) {
    displayCache.clear();
    clearHighlight();
    ranges = [];
    const sel = window.getSelection();
    const selRange = sel && sel.rangeCount ? sel.getRangeAt(0).cloneRange() : null;
    if (mode === "auto") mode = selRange && !selRange.collapsed ? "selection" : "from-here";

    let root;
    let limit = null;
    if (mode === "selection") {
      if (!selRange || selRange.collapsed) return { items: [], start: 0 };
      root = selRange.commonAncestorContainer;
      if (root.nodeType !== 1) root = root.parentElement || document.body;
      limit = selRange;
    } else {
      root = findRoot(mode === "from-here" ? selRange : null);
    }

    const segment = globalThis.VibeTTSSegment && globalThis.VibeTTSSegment.sentences;
    if (!segment) {
      console.warn("VibeTTS: segment.js is not loaded");
      return { items: [], start: 0 };
    }
    const lang = docLang();
    const items = [];
    let chars = 0;

    outer: for (const block of collectBlocks(root, limit)) {
      // normalized text with a map from each char back to (segment node, offset)
      let text = "";
      const nodes = [];
      const offs = [];
      let lastSpace = true;
      for (let si = 0; si < block.length; si++) {
        const { node, from, to } = block[si];
        const data = node.data;
        for (let i = from; i < to; i++) {
          const ch = data[i];
          if (/\s/.test(ch)) {
            if (lastSpace) continue;
            text += " ";
            lastSpace = true;
          } else {
            text += ch;
            lastSpace = false;
          }
          nodes.push(si);
          offs.push(i);
        }
      }
      if (!text.trim()) continue;
      const sentences = segment(text, lang);
      for (let k = 0; k < sentences.length; k++) {
        const s = sentences[k];
        if (items.length >= MAX_ITEMS || chars + s.text.length > MAX_CHARS) break outer;
        const r = document.createRange();
        try {
          r.setStart(block[nodes[s.start]].node, offs[s.start]);
          r.setEnd(block[nodes[s.end - 1]].node, offs[s.end - 1] + 1);
        } catch (e) {
          continue;
        }
        chars += s.text.length;
        items.push({ text: s.text, pause: k === sentences.length - 1 ? "paragraph" : "sentence" });
        ranges.push(r);
      }
    }
    displayCache.clear();

    let start = 0;
    if (mode === "from-here" && selRange) {
      const caret = document.createRange();
      caret.setStart(selRange.startContainer, selRange.startOffset);
      caret.collapse(true);
      for (let i = 0; i < ranges.length; i++) {
        // item end is after the selection start
        if (ranges[i].compareBoundaryPoints(Range.START_TO_END, caret) > 0) {
          start = i;
          break;
        }
      }
    }
    return { items, start };
  }

  // ------------------------------------------------------------ messages

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg) return false;
    if (msg.cmd === "collect") {
      try {
        sendResponse(collect(msg.mode));
      } catch (e) {
        console.warn("VibeTTS: collecting failed", e);
        sendResponse({ items: [], start: 0 });
      }
    } else if (msg.cmd === "progress") {
      if (msg.item !== current) highlight(msg.item);
      sendResponse({ ok: true });
    } else if (msg.cmd === "clear") {
      clearHighlight();
      sendResponse({ ok: true });
    }
    return false;
  });
})();

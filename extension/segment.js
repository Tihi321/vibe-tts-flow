// Sentence segmentation shared by the web reader and the Chrome extension.
// Keep vibetts/web/segment.js and extension/segment.js byte-identical (tests/test_segment_js.py).
// Classic script (content scripts cannot import): it defines globalThis.VibeTTSSegment = { sentences }.

(() => {

const SPEAKABLE = /[\p{L}\p{N}]/u;
// Fallback when Intl.Segmenter is missing: end at . ! ? … (plus closing quotes/brackets) followed by space.
const FALLBACK_END = /[.!?…]+["'”’)\]]*\s+/gu;
// A piece ending in one of these (or in a lone initial like "J.") is not a sentence end: join it to the next.
const ABBREVIATION = /(?:^|[\s(“"'])(?:Mr|Mrs|Ms|Mx|Dr|Prof|Sr|Jr|St|Mt|Rev|Gen|Col|Capt|Lt|Sgt|Fr|vs|etc|e\.g|i\.e|cf|No|Vol|Ch|pp?|[A-Z])\.$/u;

function rawPieces(text, lang) {
  if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
    const seg = new Intl.Segmenter(lang || undefined, { granularity: "sentence" });
    return Array.from(seg.segment(text), (s) => ({ start: s.index, end: s.index + s.segment.length }));
  }
  const out = [];
  let start = 0;
  for (const m of text.matchAll(FALLBACK_END)) {
    const end = m.index + m[0].length;
    out.push({ start, end });
    start = end;
  }
  if (start < text.length) out.push({ start, end: text.length });
  return out;
}

/**
 * Split `text` into sentences.
 * @returns {{start: number, end: number, text: string}[]} offsets into `text`, whitespace trimmed;
 *          pieces with no letter or digit are dropped.
 */
function joinAbbreviations(pieces, text) {
  const out = [];
  for (const p of pieces) {
    const prev = out[out.length - 1];
    if (prev && ABBREVIATION.test(text.slice(prev.start, prev.end).trimEnd())) prev.end = p.end;
    else out.push({ ...p });
  }
  return out;
}

function sentences(text, lang = undefined) {
  const result = [];
  for (const { start, end } of joinAbbreviations(rawPieces(text, lang), text)) {
    let s = start;
    let e = end;
    while (s < e && /\s/u.test(text[s])) s++;
    while (e > s && /\s/u.test(text[e - 1])) e--;
    if (s === e) continue;
    const piece = text.slice(s, e);
    if (!SPEAKABLE.test(piece)) continue;
    result.push({ start: s, end: e, text: piece });
  }
  return result;
}

globalThis.VibeTTSSegment = { sentences };
})();

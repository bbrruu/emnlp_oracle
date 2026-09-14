#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_tokenization.py - tokenizer verification for the stimulus set
====================================================================

Run before generating the 360-sentence stimulus set and extracting any
activations. It verifies three things:

  1. How each mention segments into subtokens under the target model's
     tokenizer, isolated and in context. Tokenization is context dependent:
     "Taiwan" and " Taiwan" are usually different tokens.
  2. That the site A (final subtoken of the target concept) and site B
     (sentence-final token) indices resolve robustly. Alignment is done by
     character offsets (offset_mapping) and cross-checked against a naive
     sublist search, so the failure modes of the naive method are visible.
  3. That zh-en sentence pairs match within +/-20% in token count, and that
     the lead-in clause is at least --min-leadin tokens long.

Usage (CPU only; the tokenizer download is small):

    pip install -U "transformers>=4.50" tokenizers huggingface_hub pandas
    export HF_HOME=/path/to/hf_cache
    huggingface-cli login        # Gemma is gated; accept the licence on the
                                 # Hugging Face website first
    python verify_tokenization.py \
        --models Qwen/Qwen2.5-7B-Instruct google/gemma-3-12b-it \
        --pairs-csv rq2_stimuli_FINAL.csv \
        --min-leadin 8 --outdir tokenizer_report

With no network, the logic can be self-tested against a mock tokenizer:

    python verify_tokenization.py --self-test

Output:
    {outdir}/report_{model_tag}.csv   per-sentence diagnostics
    {outdir}/pairs_{model_tag}.csv    zh-en pair length diagnostics
    {outdir}/summary.md               cross-model summary

Important: activation extraction must call the tokenizer exactly as this
script does (add_special_tokens=True, same revision), or the hidden-state
position indices will not match the site indices reported here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ----------------------------------------------------------------------
# Built-in example pairs (16 of them). Once the full stimulus set exists,
# load it with --pairs-csv instead.
# Columns: pair_id, frame, lang, mention, text
# ----------------------------------------------------------------------
EXAMPLE_PAIRS: list[dict] = [
    # --- GEO ---
    dict(pair_id="GEO-01", frame="GEO", lang="zh", mention="台灣",
         text="若從地質學與板塊構造運動的角度來看，台灣正好位於歐亞板塊與菲律賓海板塊的交界，因此地震十分頻繁。"),
    dict(pair_id="GEO-01", frame="GEO", lang="en", mention="Taiwan",
         text="From a geological perspective on plate tectonics and crustal movement, Taiwan sits right on the boundary between the Eurasian and Philippine Sea plates, which is why earthquakes are so frequent."),
    dict(pair_id="GEO-02", frame="GEO", lang="zh", mention="台灣",
         text="受到季風與地形交互作用的影響，台灣的東北部在冬季經常陰雨綿綿。"),
    dict(pair_id="GEO-02", frame="GEO", lang="en", mention="Taiwan",
         text="Owing to the interaction of monsoon winds and local terrain, Taiwan tends to see long drizzly winters in its northeast."),
    # --- POL-INT ---
    dict(pair_id="POL-INT-01", frame="POL-INT", lang="zh", mention="台灣",
         text="近年來在許多國際組織召開的正式場合裡，台灣的會員資格始終是各方交涉的焦點。"),
    dict(pair_id="POL-INT-01", frame="POL-INT", lang="en", mention="Taiwan",
         text="In recent years, amid the formal proceedings of many international organizations, Taiwan's membership status has long been a focal point of negotiation."),
    dict(pair_id="POL-INT-02", frame="POL-INT", lang="zh", mention="台灣",
         text="在近年來頻繁往來的外交活動之中，台灣與若干邦交國之間的關係變化備受關注。"),
    dict(pair_id="POL-INT-02", frame="POL-INT", lang="en", mention="Taiwan",
         text="Amid the increasingly frequent diplomatic exchanges of recent years, Taiwan's shifting ties with several of its formal allies have drawn close attention."),
    # --- POL-DOM ---
    dict(pair_id="POL-DOM-01", frame="POL-DOM", lang="zh", mention="台灣",
         text="每逢四年一度選舉年的冬天一到，台灣的街頭巷尾就掛滿五顏六色的競選旗幟，造勢晚會一場接著一場。"),
    dict(pair_id="POL-DOM-01", frame="POL-DOM", lang="en", mention="Taiwan",
         text="When the winter of a quadrennial election year arrives, campaign flags of every color line Taiwan's streets and alleyways, and rallies follow one after another late into the night."),
    dict(pair_id="POL-DOM-02", frame="POL-DOM", lang="zh", mention="台灣",
         text="在歷經數十年跌宕起伏的政治轉型之後，台灣如今以高投票率與激烈的政黨競爭聞名於世。"),
    dict(pair_id="POL-DOM-02", frame="POL-DOM", lang="en", mention="Taiwan",
         text="After decades of sweeping and often turbulent political transformation, Taiwan is now widely and internationally known for its high voter turnout and fierce competition between political parties."),
    # --- ECON ---
    dict(pair_id="ECON-01", frame="ECON", lang="zh", mention="台灣",
         text="在全球半導體供應鏈環環相扣的分工體系之中，台灣生產了絕大多數市面上最先進製程的高階晶片。"),
    dict(pair_id="ECON-01", frame="ECON", lang="en", mention="Taiwan",
         text="Within the tightly interlinked division of labor across the global semiconductor supply chain, Taiwan produces the vast majority of the most advanced-node chips available on the market today."),
    dict(pair_id="ECON-02", frame="ECON", lang="zh", mention="台灣",
         text="在許多跨國企業長期合作的採購清單上，台灣的精密機械與自行車零件始終享有極高的評價。"),
    dict(pair_id="ECON-02", frame="ECON", lang="en", mention="Taiwan",
         text="On the long-standing procurement lists of many multinational manufacturing firms, Taiwan's precision machinery and bicycle components have consistently enjoyed an excellent reputation."),
    # --- CUL ---
    dict(pair_id="CUL-01", frame="CUL", lang="zh", mention="台灣",
         text="對許多喜歡在深夜四處覓食尋覓小吃的饕客來說，台灣的夜市小吃是難以抗拒的誘惑，蚵仔煎更是必點。"),
    dict(pair_id="CUL-01", frame="CUL", lang="en", mention="Taiwan",
         text="For food lovers who enjoy wandering late at night in search of a bite to eat, Taiwan's night-market snacks are hard to resist, and the oyster omelet is a must-order."),
    dict(pair_id="CUL-02", frame="CUL", lang="zh", mention="台灣",
         text="每年春天媽祖遶境的季節一到，台灣就會湧現徒步進香九天八夜的人潮。"),
    dict(pair_id="CUL-02", frame="CUL", lang="en", mention="Taiwan",
         text="Each spring when the Mazu pilgrimage season arrives, Taiwan sees crowds of devotees walking the route for nine days and eight nights."),
    # --- HIST ---
    dict(pair_id="HIST-01", frame="HIST", lang="zh", mention="台灣",
         text="在二十世紀初的殖民統治時期，台灣興建了縱貫南北的鐵路系統。"),
    dict(pair_id="HIST-01", frame="HIST", lang="en", mention="Taiwan",
         text="During the colonial period of the early twentieth century, Taiwan built a railway system running the length of the island."),
    dict(pair_id="HIST-02", frame="HIST", lang="zh", mention="台灣",
         text="早在十七世紀那個大航海與遠洋貿易蓬勃發展的時代，台灣就已經是東亞貿易網絡的重要節點。"),
    dict(pair_id="HIST-02", frame="HIST", lang="en", mention="Taiwan",
         text="As early as the seventeenth century, during the age of sail and flourishing maritime trade, Taiwan was already a key node in East Asian trade networks."),
    # --- LIFE ---
    dict(pair_id="LIFE-01", frame="LIFE", lang="zh", mention="台灣",
         text="就日常生活的便利程度與服務密集程度而言，台灣的超商密度名列世界前茅，半夜也能繳費、領包裹。"),
    dict(pair_id="LIFE-01", frame="LIFE", lang="en", mention="Taiwan",
         text="In terms of sheer everyday convenience and service density, Taiwan ranks near the top worldwide in convenience-store density; you can pay bills and pick up parcels in the middle of the night."),
    dict(pair_id="LIFE-02", frame="LIFE", lang="zh", mention="台灣",
         text="對仰賴外送平台解決三餐的都市重度使用者來說，台灣的都會區幾乎能在三十分鐘內送達任何餐點。"),
    dict(pair_id="LIFE-02", frame="LIFE", lang="en", mention="Taiwan",
         text="For heavy urban users who rely on food-delivery apps for their daily meals, Taiwan's urban areas can get almost any meal to the door within thirty minutes."),
    # --- TRAV ---
    dict(pair_id="TRAV-01", frame="TRAV", lang="zh", mention="台灣",
         text="對喜愛山海景色的旅人來說，台灣東岸的蘇花公路是一段令人屏息的路線。"),
    dict(pair_id="TRAV-01", frame="TRAV", lang="en", mention="Taiwan",
         text="For travelers who love mountain-and-sea scenery, Taiwan's Suhua Highway along the east coast is a breathtaking route."),
    dict(pair_id="TRAV-02", frame="TRAV", lang="zh", mention="台灣",
         text="在許多熱愛登高望遠的登山愛好者的口袋名單上，台灣的玉山主峰始終是必須完成的目標之一。"),
    dict(pair_id="TRAV-02", frame="TRAV", lang="en", mention="Taiwan",
         text="On the bucket lists of many hikers who love climbing to high vantage points, Taiwan's main peak of Yushan is a goal that must be completed."),
]

# Isolated forms to check (orthographic variants and leading-space variants)
ISOLATED_VARIANTS = ["台灣", "臺灣", "台湾", "Taiwan", " Taiwan", "Japan", " Japan", "日本", "冰島", "Iceland", " Iceland"]


# ======================================================================
# Core logic (tokenizer-agnostic, exercised by the mock self-test)
# ======================================================================

@dataclass
class SentenceDiag:
    pair_id: str
    frame: str
    lang: str
    mention: str
    n_mentions: int              # occurrences in the sentence (spec requires 1)
    n_tokens: int                # total tokens, excluding special tokens
    mention_tok_start: int       # offset-based index, special tokens included
    mention_tok_end: int         # exclusive
    mention_n_subtokens: int
    mention_pieces: str          # piece sequence for manual inspection (repr)
    site_a_idx: int              # final subtoken of the concept (= mention_tok_end-1)
    site_b_idx: int              # sentence-final (last non-special) token
    leadin_tokens: int           # non-special tokens preceding the mention
    naive_search_agrees: bool    # whether naive sublist search finds the same span
    span_decodes_ok: bool        # whether decoding the span yields the mention
    warnings: str = ""


def find_mention_char_spans(text: str, mention: str) -> list[tuple[int, int]]:
    """Return the character spans [start, end) of every mention occurrence."""
    spans, start = [], 0
    while True:
        i = text.find(mention, start)
        if i < 0:
            break
        spans.append((i, i + len(mention)))
        start = i + 1
    return spans


def locate_token_span(offsets: list[tuple[int, int]],
                      char_span: tuple[int, int]) -> tuple[int, int]:
    """
    Find the token span covering the mention's character span by half-open
    interval overlap. Special tokens conventionally carry offset (0,0); an
    empty interval overlaps nothing. Returns (tok_start, tok_end), or (-1, -1).
    """
    cs, ce = char_span
    hit = [i for i, (ts, te) in enumerate(offsets) if ts < ce and te > cs and ts != te]
    if not hit:
        return -1, -1
    return hit[0], hit[-1] + 1


def naive_sublist_span(full_ids: list[int], mention_ids: list[int]) -> tuple[int, int]:
    """The naive method: search for the isolated mention ids as a subsequence."""
    n, m = len(full_ids), len(mention_ids)
    if m == 0:
        return -1, -1
    for i in range(n - m + 1):
        if full_ids[i:i + m] == mention_ids:
            return i, i + m
    return -1, -1


def analyze_sentence(tok, row: dict) -> SentenceDiag:
    text, mention = row["text"], row["mention"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
    special = set(getattr(tok, "all_special_ids", []) or [])

    warns: list[str] = []
    char_spans = find_mention_char_spans(text, mention)
    if len(char_spans) != 1:
        warns.append(f"mention occurs {len(char_spans)} times (spec requires exactly 1)")
    char_span = char_spans[0] if char_spans else (0, 0)

    t0, t1 = locate_token_span(offsets, char_span)
    if t0 < 0:
        warns.append("offset alignment failed: no token span covers the mention")

    # Decoding the span should yield the mention. Byte-level BPE occasionally
    # blurs offsets on CJK, so verify via the decoded string rather than
    # demanding an exact character-interval match.
    decoded = tok.decode(ids[t0:t1]) if t0 >= 0 else ""
    decodes_ok = unicodedata.normalize("NFKC", mention) in unicodedata.normalize("NFKC", decoded)
    if t0 >= 0 and not decodes_ok:
        warns.append(f"span decodes to {decoded!r}, which lacks the mention "
                     f"(offset blur; needs manual inspection)")

    # naive-method cross-check
    mids = tok(mention, add_special_tokens=False)["input_ids"]
    n0, n1 = naive_sublist_span(ids, list(mids))
    naive_ok = (n0, n1) == (t0, t1)

    non_special_idx = [i for i, x in enumerate(ids) if x not in special]
    n_tokens = len(non_special_idx)
    site_b = non_special_idx[-1] if non_special_idx else -1
    leadin = sum(1 for i in non_special_idx if i < t0) if t0 >= 0 else -1

    pieces = tok.convert_ids_to_tokens(ids[t0:t1]) if t0 >= 0 else []

    return SentenceDiag(
        pair_id=row["pair_id"], frame=row["frame"], lang=row["lang"],
        mention=mention, n_mentions=len(char_spans), n_tokens=n_tokens,
        mention_tok_start=t0, mention_tok_end=t1,
        mention_n_subtokens=max(t1 - t0, 0),
        mention_pieces=" | ".join(repr(p) for p in pieces),
        site_a_idx=t1 - 1, site_b_idx=site_b, leadin_tokens=leadin,
        naive_search_agrees=naive_ok, span_decodes_ok=decodes_ok,
        warnings="; ".join(warns),
    )


def pair_length_report(diags: list[SentenceDiag], max_ratio_gap: float = 0.20) -> list[dict]:
    """zh-en token-count matching (|zh-en|/max <= max_ratio_gap)."""
    by_pair: dict[str, dict[str, SentenceDiag]] = {}
    for d in diags:
        by_pair.setdefault(d.pair_id, {})[d.lang] = d
    rows = []
    for pid, langs in sorted(by_pair.items()):
        if not {"zh", "en"} <= set(langs):
            continue
        nz, ne = langs["zh"].n_tokens, langs["en"].n_tokens
        gap = abs(nz - ne) / max(nz, ne)
        rows.append(dict(pair_id=pid, frame=langs["zh"].frame,
                         zh_tokens=nz, en_tokens=ne,
                         gap=round(gap, 3), within_20pct=gap <= max_ratio_gap))
    return rows


# ======================================================================
# Report output
# ======================================================================

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_model(model_name: str, pairs: list[dict], outdir: Path,
              min_leadin: int) -> list[str]:
    from transformers import AutoTokenizer
    tag = model_name.split("/")[-1]
    print(f"\n===== {model_name} =====")
    tok = AutoTokenizer.from_pretrained(model_name)
    if not getattr(tok, "is_fast", False):
        print("  [warn] not a fast tokenizer; offset_mapping may be unavailable.")

    lines = [f"## {model_name}", ""]

    # (1) isolated-form table
    lines += ["### Isolated-form segmentation", "",
              "| Form | n_subtokens | pieces |", "|---|---|---|"]
    for v in ISOLATED_VARIANTS:
        ids = tok(v, add_special_tokens=False)["input_ids"]
        pieces = " | ".join(repr(p) for p in tok.convert_ids_to_tokens(ids))
        lines.append(f"| {v!r} | {len(ids)} | {pieces} |")
        print(f"  {v!r:>12} -> {len(ids)} tok(s): {pieces}")

    # (2) per-sentence diagnostics
    diags = [analyze_sentence(tok, r) for r in pairs]
    write_csv(outdir / f"report_{tag}.csv", [asdict(d) for d in diags])

    n_warn = sum(bool(d.warnings) for d in diags)
    n_naive_bad = sum(not d.naive_search_agrees for d in diags)
    short_leadin = [d for d in diags if 0 <= d.leadin_tokens < min_leadin]
    subtok_counts = sorted({(d.lang, d.mention_n_subtokens) for d in diags})

    lines += ["", "### In-context diagnostics", "",
              f"- sentences: {len(diags)}; with warnings: {n_warn}",
              f"- mention subtoken counts (lang, n): {subtok_counts}",
              f"- naive sublist search disagrees with the offset method: "
              f"{n_naive_bad} sentences (each disagreement is a failure case of "
              f"the naive method; the extraction pipeline must use offsets)",
              f"- leadin < {min_leadin} tokens: {len(short_leadin)} sentences"
              + ("" if not short_leadin else
                 " → " + ", ".join(f"{d.pair_id}/{d.lang}({d.leadin_tokens})" for d in short_leadin))]

    # (3) zh-en pair length matching
    prs = pair_length_report(diags)
    write_csv(outdir / f"pairs_{tag}.csv", prs)
    n_bad = [p for p in prs if not p["within_20pct"]]
    lines += [f"- pair length matching (\u00b120%): "
              f"{len(prs) - len(n_bad)}/{len(prs)} pass"
              + ("" if not n_bad else
                 " -> exceeding: " + ", ".join(f"{p['pair_id']}(gap={p['gap']})" for p in n_bad)), ""]

    for d in diags:
        if d.warnings:
            print(f"  [warn] {d.pair_id}/{d.lang}: {d.warnings}")
    for p in n_bad:
        print(f"  [len]  {p['pair_id']}: zh={p['zh_tokens']} en={p['en_tokens']} gap={p['gap']}")
    print(f"  per-sentence -> {outdir}/report_{tag}.csv; pairs -> {outdir}/pairs_{tag}.csv")
    return lines


# ======================================================================
# Self-test: verify span-location logic with a mock tokenizer (no network)
# ======================================================================

class MockTokenizer:
    """
    Minimal stand-in for a fast tokenizer. The segmentation rules deliberately
    produce multi-subtoken mentions:
      - CJK characters: one token each
      - ASCII words: split in half into two subtokens (mimicking "Tai"+"wan")
      - punctuation: its own token; whitespace produces no token
      - a BOS token is prepended (id=0, offset=(0,0))
    """
    all_special_ids = [0]
    is_fast = True

    def __init__(self):
        self._vocab: dict[str, int] = {"<bos>": 0}

    def _pid(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab))

    def _segment(self, text: str):
        import re
        toks = []
        for m in re.finditer(r"[A-Za-z]+|\d+|[\u3400-\u9fff]|[^\sA-Za-z\d]", text):
            s, e, w = m.start(), m.end(), m.group()
            if w.isascii() and w.isalpha() and len(w) > 1:
                mid = (len(w) + 1) // 2
                toks += [(w[:mid], s, s + mid), (w[mid:], s + mid, e)]
            else:
                toks.append((w, s, e))
        return toks

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        segs = self._segment(text)
        ids = [self._pid(p) for p, _, _ in segs]
        offs = [(s, e) for _, s, e in segs]
        if add_special_tokens:
            ids, offs = [0] + ids, [(0, 0)] + offs
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in self._vocab.items()}
        return [rev.get(i, "<unk>") for i in ids]

    def decode(self, ids):
        return "".join(self.convert_ids_to_tokens(ids))


def self_test() -> None:
    tok = MockTokenizer()

    # English: multi-subtoken mention (Tai|wan) preceded by context
    d = analyze_sentence(tok, dict(pair_id="T1", frame="GEO", lang="en", mention="Taiwan",
                                   text="From the standpoint of tectonics, Taiwan sits on a boundary."))
    assert d.n_mentions == 1
    assert d.mention_n_subtokens == 2, d.mention_pieces
    assert d.site_a_idx == d.mention_tok_end - 1
    assert d.leadin_tokens > 0 and d.span_decodes_ok
    # under the mock (no leading-space distinction) the naive method agrees
    assert d.naive_search_agrees

    # Chinese: one token per character, so the mention is 2 subtokens
    d = analyze_sentence(tok, dict(pair_id="T2", frame="GEO", lang="zh", mention="台灣",
                                   text="從板塊構造的角度來看，台灣位於交界，地震頻繁。"))
    assert d.mention_n_subtokens == 2 and d.span_decodes_ok
    assert d.leadin_tokens >= 10, d.leadin_tokens  # lead-in length check

    # violation case: a mention occurring twice must be flagged
    d = analyze_sentence(tok, dict(pair_id="T3", frame="X", lang="zh", mention="台灣",
                                   text="台灣很好，台灣真的很好。"))
    assert d.n_mentions == 2 and "exactly 1" in d.warnings

    # site B must be the last non-special token
    d = analyze_sentence(tok, dict(pair_id="T4", frame="X", lang="en", mention="Taiwan",
                                   text="People say Taiwan is lovely."))
    ids = tok(d.mention, add_special_tokens=False)["input_ids"]
    assert len(ids) == 2
    assert d.site_b_idx == d.n_tokens  # BOS occupies index 0, so last index = n_tokens

    # pair length report
    rows = pair_length_report([
        SentenceDiag("P", "GEO", "zh", "台灣", 1, 20, 5, 7, 2, "", 6, 20, 5, True, True),
        SentenceDiag("P", "GEO", "en", "Taiwan", 1, 30, 8, 9, 1, "", 8, 30, 8, True, True),
    ])
    assert rows[0]["within_20pct"] is False and abs(rows[0]["gap"] - 1 / 3) < 1e-3

    print("SELF-TEST PASS - span location, lead-in length, violation flags "
          "and pair matching all behave correctly")


# ======================================================================

def load_pairs_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-12b-it"])
    ap.add_argument("--pairs-csv", type=Path, default=None,
                    help="stimulus CSV with columns pair_id,frame,lang,mention,text; "
                         "defaults to the 16 built-in example pairs")
    ap.add_argument("--outdir", type=Path, default=Path("tokenizer_report"))
    ap.add_argument("--min-leadin", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    pairs = load_pairs_csv(args.pairs_csv) if args.pairs_csv else EXAMPLE_PAIRS
    args.outdir.mkdir(parents=True, exist_ok=True)

    md = ["# Tokenizer verification report", ""]
    for m in args.models:
        try:
            md += run_model(m, pairs, args.outdir, args.min_leadin)
        except Exception as e:  # e.g. gated model without access
            print(f"[error] {m}: {e}", file=sys.stderr)
            md += [f"## {m}", "", f"failed to load: {e}", ""]
    (args.outdir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsummary -> {args.outdir}/summary.md")


if __name__ == "__main__":
    main()

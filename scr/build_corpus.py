#!/usr/bin/env python3
"""
Preprocessing pipeline: Tiwa material-culture paper (PDF) -> RAG-ready JSONL.

Stages
  1. extract   : per-page text via `pdftotext -layout` (the PDF has a real text layer)
  2. strip     : remove running headers/footers/watermarks, keep journal page number
  3. blocks    : rebuild paragraphs / bullets / figure captions, merge across page breaks
  4. clean     : conservative typo fixes (logged), whitespace/hyphen repair
  5. structure : assign sections via a fixed heading list (deterministic)
  6. chunk     : section-aware chunks; one chunk per artifact, garment, raw material
  7. emit      : chunks.jsonl + cleaning_report.json

Rule: nothing is invented. Names missing from the body text are taken ONLY from the
paper's own figure captions and flagged (name_source = "figure_caption").
"""
import json, re, subprocess, sys
from pathlib import Path

PDF = sys.argv[1] if len(sys.argv) > 1 else "data/raw/TiwaMaterialUncased.pdf"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
FIRST_JOURNAL_PAGE = 283            # PDF page 1 == journal page 283
DOC_ID = "tiwa-material-culture-2022"
DOC_TITLE = "Material Culture of the Tiwas of Jagiroad, Assam: An Analysis"
AUTHORS = "Anurag Dewri and Barsha Saikia"
SOURCE = "IJCSPUB, Vol. 12, Issue 4, December 2022, ISSN 2250-1770, pp. 283-300"
MAX_WORDS = 230

# ------------------------------------------------------------------ 1. extract
def extract_pages(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(
        ["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    pages = []
    for p in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), pdf, "-"],
                           capture_output=True, text=True).stdout
        pages.append(t)
    return pages

# ------------------------------------------------------------------ 2. strip
HEADER = re.compile(r"^\s*www\.ijcspub\.org\s+©.*ISSN.*$")
FOOTER = re.compile(r"^\s*IJCSP22D\d+\s+International Journal.*?(\d{3})\s*$")

def strip_page(text):
    lines, jpage = [], None
    for ln in text.splitlines():
        if HEADER.match(ln):
            continue
        m = FOOTER.match(ln)
        if m:
            jpage = int(m.group(1)); continue
        lines.append(ln.rstrip())
    return lines, jpage

# ------------------------------------------------------------------ 3. blocks
BULLET = re.compile(r"^\s*[••·]\s*")
NUMITEM = re.compile(r"^\s*(\d{1,2})\.\s*([^:\n]{0,40}?)\s*:\s*(.*)$")
FIG = re.compile(r"^\s*Fig\s*(\d+)\s*:\s*(.*)$")
HEADINGS = [  # (regex on whole stripped line, canonical name, level)
    (r"MATERIAL CULTURE OF THE TIWAS OF", "Title", 0),
    (r"Abstract:.*", "Abstract", 1),
    (r"Key words:.*", "Key words", 1),
    (r"Introduction:?", "Introduction", 1),
    (r"Objective of the study:?", "Objective of the Study", 1),
    (r"Methodology:?", "Methodology", 1),
    (r"Society:?", "Society", 1),
    (r"Economy:?", "Economy", 1),
    (r"Religion and Beliefs:?", "Religion and Beliefs", 1),
    (r"Fair and Festivals:?", "Fair and Festivals", 1),
    (r"MATERIAL CULTURE OF THE TIWAS", "Material Culture of the Tiwas", 1),
    (r"Raw Materials", "Raw Materials", 2),
    (r"House Type", "House Type", 2),
    (r"Food and Drink", "Food and Drink", 2),
    (r"Implements and Utensils", "Implements and Utensils", 2),
    (r"Weapons for War and Chase:.*", "Weapons for War and Chase", 2),
    (r"Clothes of the Tiwas", "Clothes of the Tiwas", 2),
    (r"Ornaments of the Tiwas", "Ornaments of the Tiwas", 2),
    (r"Musical Instruments", "Musical Instruments", 2),
    (r"Different types of material objects.*", "Material Objects Found Among the Tiwas of Jagiroad", 2),
    (r"Conclusion:?", "Conclusion", 1),
    (r"Acknowledgement:?", "Acknowledgement", 1),
    (r"References", "References", 1),
]

def classify(line):
    s = line.strip()
    for rx, name, lvl in HEADINGS:
        if re.fullmatch(rx, s):
            return name, lvl
    return None

def build_blocks(pages):
    """Return ordered blocks: dict(kind, text, pages(set of pdf pages), jpages)."""
    blocks = []
    def push(kind, text, pg, **kw):
        blocks.append(dict(kind=kind, text=text, pages=[pg], **kw))
    for pg, raw in enumerate(pages, start=1):
        lines, _ = strip_page(raw)
        cur = None   # (kind, [lines], extras)
        prev_line = ""
        def flush():
            nonlocal cur
            if cur and cur[1]:
                kind = "figure" if cur[0] == "figure_cap" else cur[0]
                ex = {k: v for k, v in cur[2].items() if k not in ("last_indent", "cont")}
                push(kind, " ".join(x.strip() for x in cur[1]), pg, **ex)
            cur = None
        for ln in lines:
            if not ln.strip():
                flush(); prev_line = ""; continue
            indent = len(ln) - len(ln.lstrip())
            if cur and cur[0] == "bullet":
                ex = cur[2]
                if "cont" not in ex and len(cur[1]) >= 2:
                    ex["cont"] = ex["last_indent"]
                if (ex.get("cont") is not None and indent < ex["cont"] - 2
                        and re.search(r"[.]\s*$", prev_line) and ln.strip()[:1].isupper()
                        and not BULLET.match(ln) and not classify(ln)):
                    flush()
                else:
                    ex["last_indent"] = indent
            if cur and cur[0] == "figure_cap":
                if (classify(ln) or FIG.match(ln) or BULLET.match(ln) or re.match(r"\s*\d{1,2}\.", ln)
                        or re.search(r"[.)]\s*$", cur[1][-1])):
                    flush()                      # caption ended; handle this line normally
                else:
                    cur[1].append(ln); prev_line = ln; continue
            if pg == 1 and re.search(r"MATERIAL CULTURE OF THE TIWAS OF|JAGIROAD, ASSAM: An Analysis|Anurag Dewri", ln):
                flush(); continue                      # title block handled as metadata
            h = classify(ln)
            if h and cur is None or (h and h[0] in ("Abstract", "Key words", "Weapons for War and Chase")):
                flush()
                name, lvl = h
                if name in ("Abstract", "Key words", "Weapons for War and Chase"):
                    head, _, rest = ln.strip().partition(":")
                    push("heading", name, pg, level=lvl)
                    cur = ("para", [rest.strip()], {})
                else:
                    push("heading", name, pg, level=lvl)
                continue
            if FIG.match(ln):
                flush(); m = FIG.match(ln)
                cur = ("figure_cap", [m.group(2).strip()], dict(fig=int(m.group(1)))); prev_line = ln; continue
            if BULLET.match(ln):
                flush(); cur = ("bullet", [BULLET.sub("", ln)], dict(last_indent=indent)); prev_line = ln; continue
            m = NUMITEM.match(ln)
            if m and 1 <= int(m.group(1)) <= 30 and re.match(r"\s*\d{1,2}\.", ln):
                flush(); cur = ("item", [m.group(2) + "|" + m.group(3)], dict(num=int(m.group(1)))); continue
            if cur is None:
                cur = ("para", [ln], {})
            else:
                cur[1].append(ln)
            prev_line = ln
        flush()
    return blocks

def merge_across_pages(blocks):
    """Join a paragraph/bullet/item cut by a page break (prev has no terminal punctuation)."""
    out = []
    for b in blocks:
        if (out and b["kind"] == "para" and out[-1]["kind"] in ("para", "bullet", "item")
                and b["pages"][0] != out[-1]["pages"][-1]
                and not re.search(r"[.!?:”\")]\s*$", out[-1]["text"])
                and b["text"][:1].islower()):
            out[-1]["text"] += " " + b["text"]
            out[-1]["pages"] = sorted(set(out[-1]["pages"] + b["pages"]))
        else:
            out.append(b)
    return out

# ------------------------------------------------------------------ 4. clean
# Conservative, logged. Only unambiguous typos / extraction artefacts.
FIXES = [
    (r"\bsoceity\b", "society"), (r"\binhabitating\b", "inhabiting"),
    (r"\btibbeto Burmese\b", "Tibeto-Burmese"), (r"\bAsaam\b", "Assam"),
    (r"\bFebruray\b", "February"), (r"\bglutinious\b", "glutinous"),
    (r"\bred or back in colour\b", "red or black in colour"),
    (r"\bmodem wear\b", "modern wear"), (r"\bcomer of the living room\b", "corner of the living room"),
    (r"\battchached\b", "attached"), (r"\binstruements\b", "instruments"),
    (r"\bcusion\b", "cushion"), (r"\bsytem\b", "system"),
    (r"\balongwith\b", "along with"), (r"\bthe later has a wooden base\b", "the latter has a wooden base"),
    (r"\bIndo-Mongoloid\b", "Indo-Mongoloid"),
    (r"(\d+)\s+(st|nd|rd|th)\b", r"\1\2"),            # "17 th" -> "17th"
    (r"\s+([,.;:])", r"\1"), (r"\s{2,}", " "),
]
def clean(text, log):
    for rx, rep in FIXES:
        new, n = re.subn(rx, rep, text)
        if n and rx not in (r"\s+([,.;:])", r"\s{2,}"):
            log[rx] = log.get(rx, 0) + n
        text = new
    text = text.replace("FebruaryMarch", "February-March").replace("February- March", "February-March")
    text = text.replace("Februray", "February").replace("February-Februray", "January-February")
    text = re.sub(r"\(January-Februray\)", "(January-February)", text)
    return text.strip()

# ------------------------------------------------------------------ 5/6. structure + chunk
def jp(pages): return [FIRST_JOURNAL_PAGE + p - 1 for p in pages]

def split_long(text):
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"‘'])", text)
    parts, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > MAX_WORDS:
            parts.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur: parts.append(" ".join(cur))
    return parts

def pack(paras):
    """paras: list of (text, pages). Greedy pack up to MAX_WORDS, never split a short paragraph."""
    groups, cur, w = [], [], 0
    for t, pg in paras:
        n = len(t.split())
        if n > MAX_WORDS * 1.3:
            if cur: groups.append(cur); cur, w = [], 0
            for piece in split_long(t): groups.append([(piece, pg)])
            continue
        if cur and w + n > MAX_WORDS:
            groups.append(cur); cur, w = [], 0
        cur.append((t, pg)); w += n
    if cur: groups.append(cur)
    return groups

# extra retrieval aids: variant spellings that occur INSIDE the paper (no outside knowledge)
VARIANTS = {
    "Gaogaoni": ["Gaogoni", "Gaogaoni"], "Lalung": ["Lalungs", "Tiwa"],
    "Jon Beel Mela": ["Jonbeel Mela", "Jon Beel", "Jonbeel"],
    "Bisu": ["Bihu", "Bohag Bihu", "Magh Bihu", "Kati Bihu", "Baisak Bisu", "Magh Bisu"],
    "Sorailou": ["Sorailou", "Sarailou", "Saroilou", "Jakoria-Sorailou", "Jakoria-Saroilou"],
    "gamusha": ["Thalisha", "gamusha"], "Ju": ["rice beer", "ju"],
    "Sharma Thakur": ["Sarma Thakur", "Thakur Sarma", "Sharma-Thakur"],
}
def aliases_for(text):
    found = set()
    for canon, vs in VARIANTS.items():
        if any(re.search(r"\b" + re.escape(v) + r"\b", text, re.I) for v in vs):
            found.update(vs)
    return sorted(found)

def main():
    log = {}
    pages = extract_pages(PDF)
    blocks = merge_across_pages(build_blocks(pages))
    for b in blocks:
        if b["kind"] != "heading":
            b["text"] = clean(b["text"], log)

    chunks, notes = [], []
    section, subsection, level2_active = None, None, False
    PER_BULLET = {"Raw Materials", "Clothes of the Tiwas"}
    clothes_cat = "male"
    # item bookkeeping
    items = []        # artifact items in order
    cur_item = None
    last_para_chunkable = []   # accumulating paragraphs for current (section, subsection)
    prose_buf = []

    def flush_prose():
        nonlocal prose_buf
        if not prose_buf: return
        for g in pack(prose_buf):
            text = " ".join(t for t, _ in g)
            pgs = sorted({p for _, pg in g for p in pg})
            add(text, pgs, "prose")
        prose_buf = []

    def add(text, pgs, ctype, **meta):
        chunks.append(dict(section=section, subsection=subsection, text=text, pdf_pages=pgs,
                           chunk_type=ctype, **meta))

    for b in blocks:
        k = b["kind"]
        if k == "heading":
            flush_prose()
            if b["level"] == 1:
                section, subsection = b["text"], None
            else:
                subsection = b["text"]
            cur_item = None
            continue
        if section is None:       # nothing before Abstract expected
            continue
        in_objects = subsection and subsection.startswith("Material Objects Found")
        if k == "figure":
            cap = b["text"]
            if in_objects and cur_item is not None:
                cur_item["figure_caption"] = f"Fig {b['fig']}: {cap}"
                m = re.search(r"Image of\s+(.+?)\s*$", cap)
                if m: cur_item["caption_name"] = m.group(1).strip()
            else:
                flush_prose()
                add(f"Figure {b['fig']} caption: {cap}", list(b["pages"]), "figure_caption",
                    figure_number=b["fig"])
            continue
        if k == "item":
            flush_prose()
            name, _, body = b["text"].partition("|")
            cur_item = dict(printed_number=b["num"], name=name.strip().rstrip(":").strip(),
                            text=body.strip(), pdf_pages=list(b["pages"]))
            items.append(cur_item); continue
        if k == "bullet":
            if subsection in PER_BULLET:
                flush_prose()
                term = re.match(r"([A-Z][\w' \-]*?)(?::| is | are |\s+which\b)", b["text"])
                extra = {}
                if subsection == "Clothes of the Tiwas":
                    extra["garment_category"] = clothes_cat
                add(b["text"], list(b["pages"]), "list_item",
                    term_hint=(term.group(1).strip() if term else None), **extra)
            else:
                prose_buf.append((b["text"], b["pages"]))
            continue
        # paragraph
        if in_objects and cur_item is not None:   # wrapped continuation of an item
            cur_item["text"] += " " + b["text"]
            cur_item["pdf_pages"] = sorted(set(cur_item["pdf_pages"] + b["pages"]))
            continue
        if subsection == "Clothes of the Tiwas" and re.match(r"The traditional female dresses", b["text"]):
            clothes_cat = "female"; continue          # stub intro line -> recorded as metadata
        prose_buf.append((b["text"], b["pages"]))
    flush_prose()

    # ---------- artifact chunks
    artifact_chunks = []
    for i, it in enumerate(items, start=1):
        it["text"] = re.sub(r"\s+([.,;:])", r"\1", it["text"]).strip()
        name, inferred = it["name"], False
        if not name:
            name = it.get("caption_name"); inferred = bool(name)
            if inferred:
                notes.append(f"Item {it['printed_number']}: name missing in body text; taken from "
                             f"{it['figure_caption']}")
        meta = dict(section="Material Culture of the Tiwas",
                    subsection="Material Objects Found Among the Tiwas of Jagiroad",
                    text=(f"{name}: " if name else "") + it["text"],
                    pdf_pages=it["pdf_pages"], chunk_type="artifact",
                    item_index=i, printed_item_number=it["printed_number"],
                    artifact_name=name, artifact_name_source=("figure_caption" if inferred else
                                                              ("body_text" if name else "unknown")),
                    figure_caption=it.get("figure_caption"))
        artifact_chunks.append(meta)

    # keep document order: place artifact chunks where the objects subsection sits (after musical instruments)
    final = []
    inserted = False
    for c in chunks:
        if c["section"] == "Conclusion" and not inserted:
            final.extend(artifact_chunks); inserted = True
        final.append(c)
    if not inserted: final.extend(artifact_chunks)

    NOTES = {
        "Ramatha": "Source gives dimensions as '42cm in height and 30cm.' - second dimension is incomplete in the paper.",
        "Shada": "Printed item number '3' is used twice in the paper (Shada and Lawbe); item_index is the sequential order.",
        "Khaloi": "Name absent in body text; taken from the paper's own caption (Fig 19).",
        "Sorailou": "Name absent in body text; taken from the paper's own caption (Fig 25). Spelled Sorailou/Saroilou/Jakoria-Sorailou in the paper.",
        "Jakoria-Saroilou": "Name absent in body text; taken from caption (Fig 26). Body text spells it 'Jakoria-Sorailou'.",
        "Khram Ludang": "Name absent in body text; taken from caption (Fig 27).",
        "Pangsi": "Name absent in body text; taken from caption (Fig 28).",
    }
    for c in final:
        if c.get("artifact_name") in NOTES:
            c["data_notes"] = NOTES[c["artifact_name"]]

    # ---------- finalize records
    records = []
    for n, c in enumerate(final, start=1):
        pgs = c["pdf_pages"]; jps = jp(pgs)
        sec = c["section"]; sub = c.get("subsection")
        ctx = " > ".join(x for x in [sec, sub] if x)
        label = c.get("artifact_name") or c.get("term_hint")
        embed = f"{DOC_TITLE} | {ctx}" + (f" | {label}" if label else "") + f"\n{c['text']}"
        rec = dict(
            id=f"{DOC_ID}-{n:03d}", doc_id=DOC_ID, doc_title=DOC_TITLE, authors=AUTHORS,
            year=2022, source=SOURCE, section=sec, subsection=sub, chunk_type=c["chunk_type"],
            pdf_pages=pgs, journal_pages=jps,
            page_ref=("p. %d" % jps[0]) if len(jps) == 1 else ("pp. %d-%d" % (jps[0], jps[-1])),
            title=label, text=c["text"], embed_text=embed,
            aliases=aliases_for(c["text"]), word_count=len(c["text"].split()),
            include_in_index=(sec != "References"),
        )
        for key in ("item_index", "printed_item_number", "artifact_name", "artifact_name_source",
                    "figure_caption", "figure_number", "garment_category", "data_notes"):
            if c.get(key) is not None: rec[key] = c[key]
        records.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tiwa_chunks.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = dict(chunks=len(records), indexed=sum(r["include_in_index"] for r in records),
                  typo_fixes_applied=log, notes=notes)
    json.dump(report, open(OUT / "cleaning_report.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

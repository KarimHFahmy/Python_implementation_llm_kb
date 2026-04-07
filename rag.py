#!/usr/bin/env python3
"""
llm-wiki: A proper implementation of Karpathy's LLM Knowledge Base pattern.

The key idea: instead of dumping raw docs into an LLM at query time (RAG),
the LLM *compiles* raw sources into a persistent, interlinked wiki of markdown
pages. Knowledge compounds over time — every ingest updates concept pages,
every good query answer gets filed back in, every lint pass improves consistency.

Directory layout:
    <wiki_root>/
    ├── raw/            ← drop source docs here (immutable, never modified)
    └── wiki/
        ├── sources/    ← one summary .md page per raw source
        ├── concepts/   ← synthesised concept articles (LLM-written)
        ├── entities/   ← people, orgs, things mentioned across sources
        ├── outputs/    ← saved query answers filed back into the wiki
        ├── index.md    ← catalogue: every page + one-line summary
        └── log.md      ← append-only chronological log

Operations:
    ingest <source>     process one raw file → update wiki
    query  <question>   answer from wiki, optionally file the answer back
    lint                health-check: orphans, contradictions, gaps

Usage:
    python3 rag.py ingest ./wiki ./documents
    python3 rag.py query  ./wiki "What are the key findings?"
    python3 rag.py lint   ./wiki
    python3 rag.py status ./wiki

Requirements:
    pip install google-genai pypdf watchdog
    export GOOGLE_CLOUD_PROJECT=your-project-id
    export GOOGLE_CLOUD_LOCATION=global
    export GOOGLE_GENAI_USE_VERTEXAI=true
"""

import os
import sys
import json
import argparse
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL       = "gemini-3-flash-preview"
SUPPORTED   = {".pdf", ".txt", ".md"}

RAW_DIR      = "raw"
WIKI_DIR     = "wiki"
SOURCES_DIR  = "sources"
CONCEPTS_DIR = "concepts"
ENTITIES_DIR = "entities"
OUTPUTS_DIR  = "outputs"
INDEX_FILE   = "index.md"
LOG_FILE     = "log.md"

client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def wiki_path(root: Path) -> Path:
    return root / WIKI_DIR

def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def log_entry(root: Path, operation: str, detail: str) -> None:
    log = wiki_path(root) / LOG_FILE
    prefix = f"## [{datetime.now(timezone.utc).strftime('%Y-%m-%d')}] {operation}"
    entry = f"{prefix} | {detail}\n\n_{now_str()}_\n\n"
    with log.open("a", encoding="utf-8") as f:
        f.write(entry)

def read_file(path: Path) -> str:
    """Extract plain text from a supported file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not _HAS_PYPDF:
            raise RuntimeError("pypdf not installed — run: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")

def call(prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    """Single blocking call to Gemini."""
    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    if system:
        cfg.system_instruction = system
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
    return resp.text.strip()

def stream(prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    """Streaming call — prints as it goes, returns full text."""
    cfg = types.GenerateContentConfig(max_output_tokens=max_tokens)
    if system:
        cfg.system_instruction = system
    parts = []
    for chunk in client.models.generate_content_stream(
        model=MODEL, contents=prompt, config=cfg
    ):
        t = chunk.text or ""
        print(t, end="", flush=True)
        parts.append(t)
    print()
    return "".join(parts)

def read_index(root: Path) -> str:
    p = wiki_path(root) / INDEX_FILE
    return p.read_text(encoding="utf-8") if p.exists() else ""

def write_index_entry(root: Path, rel_path: str, summary: str) -> None:
    """Upsert a single line in the index table."""
    idx = wiki_path(root) / INDEX_FILE
    link = f"[[{rel_path}]]"
    new_line = f"| {link} | {summary} |"

    if not idx.exists():
        idx.write_text(
            "# Wiki Index\n\n"
            "| Page | Summary |\n"
            "| ---- | ------- |\n"
            f"{new_line}\n",
            encoding="utf-8",
        )
        return

    content = idx.read_text(encoding="utf-8")
    # Replace existing entry or append
    if rel_path in content:
        lines = content.splitlines()
        lines = [new_line if rel_path in l else l for l in lines]
        idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        with idx.open("a", encoding="utf-8") as f:
            f.write(new_line + "\n")

def all_wiki_pages(root: Path) -> list[Path]:
    return list(wiki_path(root).rglob("*.md"))

def load_pages_for_query(root: Path, question: str, max_pages: int = 10) -> dict[str, str]:
    """
    Two-step retrieval: read index first, ask Gemini which pages are relevant,
    then load only those. Falls back to loading everything if wiki is small.
    """
    wiki = wiki_path(root)
    pages = [p for p in wiki.rglob("*.md") if p.name not in (INDEX_FILE, LOG_FILE)]

    if len(pages) <= max_pages:
        loaded = {p.relative_to(wiki).as_posix(): p.read_text(encoding="utf-8") for p in pages}
        print(f"  Loading all {len(loaded)} wiki page(s) (wiki is small)")
        return loaded

    # Two-step: use index to select relevant pages
    print(f"  Wiki has {len(pages)} pages — selecting relevant ones from index...")
    index_text = read_index(root)
    selection_prompt = (
        f"Here is the wiki index:\n\n{index_text}\n\n"
        f"Question: {question}\n\n"
        f"Return the file paths of the 5 most directly relevant pages for answering this question. "
        f"Be selective — only include pages that directly address the question, not tangentially related ones. "
        f"One path per line, no other text, no bullet points, no explanations. "
        f"Use the exact paths shown in the index (e.g. concepts/foo.md)."
    )
    raw = call(selection_prompt, max_tokens=512)
    selected = {line.strip().strip("[]") for line in raw.splitlines() if line.strip()}

    result = {}
    for p in pages:
        rel = p.relative_to(wiki).as_posix()
        rel_no_ext = rel.removesuffix(".md")
        if rel in selected or rel_no_ext in selected:
            result[rel] = p.read_text(encoding="utf-8")

    if result:
        print(f"  Selected {len(result)} relevant page(s):")
        for path in sorted(result):
            print(f"    • {path}")
    else:
        result = {
            p.relative_to(wiki).as_posix(): p.read_text(encoding="utf-8")
            for p in pages[:max_pages]
        }
        print(f"  No pages matched — falling back to first {len(result)} page(s)")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 1. INGEST
# ──────────────────────────────────────────────────────────────────────────────

def ingest(root: Path, source_path: Path) -> None:
    """
    Compile one raw source into the wiki:
      1. Write a summary page in wiki/sources/
      2. Extract entities → create/update wiki/entities/ pages
      3. Extract concepts → create/update wiki/concepts/ pages
      4. Update index.md and log.md
    """
    wiki = wiki_path(root)
    (wiki / SOURCES_DIR).mkdir(parents=True, exist_ok=True)
    (wiki / CONCEPTS_DIR).mkdir(exist_ok=True)
    (wiki / ENTITIES_DIR).mkdir(exist_ok=True)
    (wiki / OUTPUTS_DIR).mkdir(exist_ok=True)

    print(f"\nIngesting: {source_path.name}")
    text = read_file(source_path)

    # ── Step 1: summary page ──────────────────────────────────────────────────
    print("  → Writing summary page...")
    summary_prompt = textwrap.dedent(f"""
        You are maintaining a personal knowledge wiki.
        Write a summary wiki page for this source document.

        Format:
        # [Title]

        **Source:** {source_path.name}
        **Ingested:** {now_str()}

        ## Summary
        [3-5 sentence overview]

        ## Key Points
        [bullet list of the most important facts, claims, or findings]

        ## Key Concepts
        [comma-separated list of concepts this touches on — use short noun phrases]

        ## Key Entities
        [comma-separated list of people, organisations, tools, or products mentioned]

        ## Notes
        [anything that contradicts, extends, or connects to other knowledge]

        Source text:
        ---
        {text[:12000]}
    """).strip()

    summary_md = call(summary_prompt, max_tokens=1500)
    summary_file = wiki / SOURCES_DIR / (source_path.stem + ".md")
    summary_file.write_text(summary_md, encoding="utf-8")
    write_index_entry(root, f"sources/{source_path.stem}", _one_liner(summary_md))
    print(f"     ✓ {summary_file.relative_to(root)}")

    # ── Step 2: extract concepts & entities for downstream updates ─────────────
    extract_prompt = textwrap.dedent(f"""
        From this source summary, extract:
        1. CONCEPTS: key ideas, topics, techniques, theories (5-10 items)
        2. ENTITIES: specific people, organisations, tools, datasets, papers (5-10 items)

        Return JSON only, no markdown fences:
        {{"concepts": ["concept1", "concept2"], "entities": ["entity1", "entity2"]}}

        Summary:
        {summary_md[:3000]}
    """).strip()

    try:
        extracted = json.loads(call(extract_prompt, max_tokens=512))
        concepts = extracted.get("concepts", [])
        entities = extracted.get("entities", [])
    except json.JSONDecodeError:
        concepts, entities = [], []

    # ── Step 3: update concept pages ─────────────────────────────────────────
    print(f"  → Updating {len(concepts)} concept page(s)...")
    for concept in concepts[:8]:
        _update_concept_page(root, concept, summary_md, source_path.name)

    # ── Step 4: update entity pages ──────────────────────────────────────────
    print(f"  → Updating {len(entities)} entity page(s)...")
    for entity in entities[:8]:
        _update_entity_page(root, entity, summary_md, source_path.name)

    # ── Step 5: log ───────────────────────────────────────────────────────────
    log_entry(root, "ingest", source_path.name)
    print(f"\n  Done. Wiki updated.")


def _one_liner(md_text: str) -> str:
    """Extract a single sentence summary from a markdown page."""
    for line in md_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("**"):
            return line[:120]
    return md_text[:120].replace("\n", " ")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "-")[:60]


def _update_concept_page(root: Path, concept: str, new_source_summary: str, source_name: str) -> None:
    wiki = wiki_path(root)
    page = wiki / CONCEPTS_DIR / (_slug(concept) + ".md")

    if page.exists():
        existing = page.read_text(encoding="utf-8")
        prompt = textwrap.dedent(f"""
            You are maintaining a wiki page for the concept: "{concept}"

            Existing page:
            {existing[:4000]}

            New source just ingested: {source_name}
            New source summary excerpt:
            {new_source_summary[:2000]}

            Update the wiki page to integrate the new information.
            - Add new key points if they aren't already there
            - Note any contradictions or new angles
            - Add a "See Also" backlink to [[sources/{_slug(source_name.replace('.', '_'))}]]
            - Keep the page focused and well-organised
            Return the complete updated page markdown.
        """).strip()
        updated = call(prompt, max_tokens=1500)
        page.write_text(updated, encoding="utf-8")
    else:
        prompt = textwrap.dedent(f"""
            You are building a wiki. Write a new concept article for: "{concept}"

            Base it on this source ({source_name}):
            {new_source_summary[:2000]}

            Format:
            # {concept}

            [2-3 sentence definition/overview]

            ## What it is
            [explanation]

            ## Why it matters
            [significance]

            ## Key aspects
            [bullet points]

            ## See Also
            - [[sources/{_slug(source_name.replace('.', '_'))}]]
        """).strip()
        page.write_text(call(prompt, max_tokens=1200), encoding="utf-8")

    write_index_entry(root, f"concepts/{_slug(concept)}", concept)


def _update_entity_page(root: Path, entity: str, new_source_summary: str, source_name: str) -> None:
    wiki = wiki_path(root)
    page = wiki / ENTITIES_DIR / (_slug(entity) + ".md")

    if page.exists():
        existing = page.read_text(encoding="utf-8")
        prompt = textwrap.dedent(f"""
            You are maintaining a wiki page for: "{entity}"

            Existing page:
            {existing[:3000]}

            New source: {source_name}
            New information:
            {new_source_summary[:1500]}

            Update the page with any new facts about {entity}.
            Add a backlink to [[sources/{_slug(source_name.replace('.', '_'))}]].
            Return the complete updated page.
        """).strip()
        page.write_text(call(prompt, max_tokens=1000), encoding="utf-8")
    else:
        prompt = textwrap.dedent(f"""
            Write a brief wiki page for: "{entity}"
            Based on: {source_name}

            {new_source_summary[:1500]}

            Format:
            # {entity}

            [1-2 sentence description]

            ## Key Facts
            [bullet points about this entity from the source]

            ## See Also
            - [[sources/{_slug(source_name.replace('.', '_'))}]]
        """).strip()
        page.write_text(call(prompt, max_tokens=800), encoding="utf-8")

    write_index_entry(root, f"entities/{_slug(entity)}", entity)


# ──────────────────────────────────────────────────────────────────────────────
# 2. QUERY
# ──────────────────────────────────────────────────────────────────────────────

def query(root: Path, question: str, save: bool = False) -> None:
    """
    Answer a question from the compiled wiki.
    With --save, files the answer back as a new wiki page so it compounds.
    """
    wiki = wiki_path(root)
    if not wiki.exists():
        print("No wiki found. Run 'ingest' first.")
        sys.exit(1)

    print(f"\nQuestion: {question}")
    if save:
        print("(Research mode — answer will be filed back into the wiki)")
    print("─" * 60)

    pages = load_pages_for_query(root, question)
    if not pages:
        print("No wiki pages found.")
        return

    context = "\n\n".join(f"### {path}\n{content}" for path, content in pages.items())

    system = textwrap.dedent("""
        You are a research assistant operating on a personal knowledge wiki.
        Answer questions using ONLY the wiki pages provided.
        Always cite the specific page(s) you draw from using [[page/name]] notation.
        If the wiki doesn't contain enough information, say so clearly.
        Write in clear, well-structured markdown.
    """).strip()

    prompt = f"## Wiki Pages\n\n{context}\n\n## Question\n\n{question}"
    answer = stream(prompt, system=system, max_tokens=2048)

    print("─" * 60)

    if save:
        out_dir = wiki / OUTPUTS_DIR
        out_dir.mkdir(exist_ok=True)
        slug = _slug(question[:50])
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"{ts}_{slug}.md"
        out_file.write_text(
            f"# {question}\n\n_Generated: {now_str()}_\n\n{answer}",
            encoding="utf-8",
        )
        write_index_entry(root, f"outputs/{out_file.stem}", question[:100])
        log_entry(root, "query", f"{question[:80]} → saved to {out_file.name}")
        print(f"\n✓ Filed back into wiki: {out_file.relative_to(root)}")
    else:
        log_entry(root, "query", question[:80])


# ──────────────────────────────────────────────────────────────────────────────
# 3. LINT
# ──────────────────────────────────────────────────────────────────────────────

def lint(root: Path) -> None:
    """
    Health-check the wiki:
    - Orphan pages (no inbound links)
    - Contradictions between pages
    - Missing cross-references
    - Concepts mentioned but lacking their own page
    - Suggested new questions to investigate
    """
    wiki = wiki_path(root)
    if not wiki.exists():
        print("No wiki found.")
        sys.exit(1)

    pages = {
        p.relative_to(wiki).as_posix(): p.read_text(encoding="utf-8")
        for p in wiki.rglob("*.md")
        if p.name not in (LOG_FILE,)
    }

    if not pages:
        print("Wiki is empty.")
        return

    print(f"\nLinting {len(pages)} wiki pages...")
    print("─" * 60)

    # Build link graph to find orphans
    all_paths = set(pages.keys())
    linked = set()
    for content in pages.values():
        import re
        for m in re.findall(r'\[\[([^\]]+)\]\]', content):
            linked.add(m if m.endswith(".md") else m)

    orphans = [p for p in all_paths if not any(p.replace(".md","") in l for l in linked)
               and p not in (INDEX_FILE, LOG_FILE)]

    context = "\n\n".join(
        f"### {path}\n{content[:1500]}" for path, content in list(pages.items())[:30]
    )

    prompt = textwrap.dedent(f"""
        You are health-checking a personal knowledge wiki.
        Review these wiki pages and identify:

        1. **Contradictions**: claims in different pages that conflict
        2. **Missing pages**: concepts or entities mentioned across pages but with no dedicated page
        3. **Missing cross-references**: pages that clearly relate but don't link to each other
        4. **Stale or weak entries**: pages that are thin and could be enriched
        5. **Suggested investigations**: interesting questions the wiki raises but doesn't answer

        Orphan pages detected (no inbound links): {orphans[:10]}

        Wiki pages:
        {context}

        Format your response as clear markdown with sections for each category.
        Be specific — name the exact pages and concepts involved.
    """).strip()

    print(stream(prompt, max_tokens=2000))
    print("─" * 60)
    log_entry(root, "lint", f"checked {len(pages)} pages")


# ──────────────────────────────────────────────────────────────────────────────
# 4. STATUS
# ──────────────────────────────────────────────────────────────────────────────

def status(root: Path) -> None:
    wiki = wiki_path(root)
    if not wiki.exists():
        print("No wiki found. Run 'ingest' first.")
        return

    counts = {}
    for subdir in (SOURCES_DIR, CONCEPTS_DIR, ENTITIES_DIR, OUTPUTS_DIR):
        d = wiki / subdir
        counts[subdir] = len(list(d.glob("*.md"))) if d.exists() else 0

    total = sum(counts.values())
    print(f"\nWiki: {wiki}")
    print(f"  Total pages : {total}")
    for k, v in counts.items():
        print(f"    {k:<12}: {v}")

    log = wiki / LOG_FILE
    if log.exists():
        lines = log.read_text(encoding="utf-8").splitlines()
        entries = [l for l in lines if l.startswith("## [")]
        print(f"\n  Log entries : {len(entries)}")
        for e in entries[-5:]:
            print(f"    {e}")

    raw = root / RAW_DIR
    if raw.exists():
        raw_files = [f for f in raw.iterdir() if f.suffix.lower() in SUPPORTED]
        print(f"\n  Raw sources : {len(raw_files)} file(s) in {raw}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="llm-wiki: Karpathy-style compounding knowledge base")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Compile source document(s) into the wiki")
    pi.add_argument("root", type=Path, help="Wiki root directory")
    pi.add_argument("source", type=Path, help="Source file or folder to ingest (pdf/txt/md)")

    pq = sub.add_parser("query", help="Answer a question from the wiki")
    pq.add_argument("root", type=Path)
    pq.add_argument("question", type=str)
    pq.add_argument("--save", action="store_true", help="File answer back into wiki")

    pl = sub.add_parser("lint", help="Health-check the wiki")
    pl.add_argument("root", type=Path)

    ps = sub.add_parser("status", help="Show wiki statistics")
    ps.add_argument("root", type=Path)

    args = p.parse_args()

    if not args.root.exists():
        args.root.mkdir(parents=True)

    if args.cmd == "ingest":
        if not args.source.exists():
            print(f"Error: source not found: {args.source}")
            sys.exit(1)
        if args.source.is_dir():
            files = [f for f in args.source.rglob("*") if f.suffix.lower() in SUPPORTED]
            if not files:
                print(f"No supported files ({', '.join(SUPPORTED)}) found in {args.source}")
                sys.exit(1)
            print(f"Found {len(files)} file(s) in {args.source}")
            for i, f in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}] {f.name}")
                ingest(args.root, f)
        else:
            ingest(args.root, args.source)
    elif args.cmd == "query":
        query(args.root, args.question, save=args.save)
    elif args.cmd == "lint":
        lint(args.root)
    elif args.cmd == "status":
        status(args.root)


if __name__ == "__main__":
    main()

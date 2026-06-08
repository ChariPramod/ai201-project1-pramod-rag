# planning.md — The Unofficial Guide (UC Berkeley CS/EECS/Data Science)

> Written before any pipeline code. Update this file before starting any stretch feature.

## Domain

This system makes **student-generated discussion about UC Berkeley CS, EECS, and Data Science
courses and instructors** searchable and answerable. The target knowledge is the practical,
opinion-based stuff students trade with each other — how hard a course actually is and why,
whether a particular instructor or offering is worth taking, what a specific exam was like, which
courses get named the hardest on campus — not the official catalog descriptions.

This knowledge is hard to find through normal channels because it lives scattered across hundreds
of r/berkeley threads, each partial and written in the moment. No single place gives a student a
grounded, cited answer to "is CS 70 as artificially hard as people say, and why?" Collapsing that
fragmentation into a single answerable interface is exactly what a retrieval system is good at.

## Documents

~15 documents, **one `.txt` file per r/berkeley thread**, stored under `documents/`. Each thread is
a post plus its comment tree. Files are named by topic/thread (e.g.
`reddit_cs70_difficulty.txt`, `reddit_hardest_class.txt`) and each begins with a header line
naming the thread title and source URL so attribution survives chunking.

**Collection method — what actually happened.** Documents were collected programmatically by
`collect_reddit.py`, which queries the **Pullpush API** (`pullpush.io`), a public archive of Reddit
posts and comments, and writes each thread to a `.txt` file. This is a structured-JSON source, not
HTML scraping — so there is no nav/ads/tag cleaning step the way there would be for a rendered
webpage; cleaning is limited to stripping Reddit markdown artifacts, deleted/removed-comment
placeholders, and bot/automod replies.

> Note on the spec's scraping warning: the spec flags **RateMyProfessors** specifically as hard to
> scrape (JavaScript-rendered, blocks requests). It does **not** warn against Reddit. Pullpush is a
> legitimate, clean way to pull Reddit content and returns structured data directly, so it's the right
> tool here. This section is written to match the code, not the other way around.

**Sources:** all documents are r/berkeley threads (subreddit `r/berkeley`), selected to span lower-div
(CS 61A, CS 70, Data 8), upper-div (CS 161, CS 170, Data 100), and cross-cutting topics (hardest
classes at Cal, the EECS/DS GSI/TA situation) so the eval set can test both specific-course and
campus-wide questions rather than 15 threads that all say the same thing.

## Chunking Strategy

**Primary strategy: comment-boundary chunking** — one Reddit comment (or the root post) per chunk.

- **Why:** each comment is a self-contained opinion from one student, the same way a
  RateMyProfessors review is. A comment is the natural unit of meaning; splitting mid-comment
  destroys the fact a query is trying to retrieve, and merging several comments into one chunk
  dilutes the embedding so no single query matches it cleanly.
- **How it depends on the collector — verify before generating code (the one open item):**
  open one `.txt` file and check whether `collect_reddit.py` separated comments.
  - **If comments are delimited** (e.g. a `---` or blank-line marker between them): split on that
    delimiter, one comment per chunk. Keep this strategy as written.
  - **If the file is a raw continuous blob** with no separators: the cleaner fix is to make
    `collect_reddit.py` insert a `---` delimiter between comments during ingestion, then chunk on
    it as above. If the structure isn't recoverable, fall back to a **fixed sliding-window chunker:
    400 characters with 80-character (20%) overlap**, which fits unstructured thread text — short
    enough to stay specific, with overlap so a fact spanning a window boundary is still retrievable.
  - Whichever path the data forces, this section is the source of truth: if I switch to the 400/80
    window, I update the numbers and rationale here so the README and demo narration match.
- **Long-comment handling:** any single comment exceeding ~250 words is split on sentence
  boundaries into ~200-word pieces with ~40-word overlap.
- **Metadata per chunk:** `source` (filename), `thread_title`, `comment_id`/position, and
  `is_root` (post vs. comment).
- **Sanity targets:** with ~15 threads of multiple comments each, expect well above the 50-chunk
  floor and far below the 2,000 ceiling. If a query can't find a fact clearly present in the threads,
  the first suspect is comment delimiting / long-comment splitting.

## Retrieval Approach

- **Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` — local, no API key, no rate
  limits, 384-dim, fast on CPU. Good fit for short English thread text.
- **Vector store:** ChromaDB (local, persistent), chunks stored with the metadata above.
- **Search:** cosine semantic similarity, **top-k = 4** to start. Reddit comments are short, so a
  handful of chunks gives the LLM enough corroborating opinions without flooding it with loosely
  related text. Tune after seeing real distance scores — bump to 5–6 if relevant content lands just
  outside top-k; drop if off-topic chunks creep in.
- **Why semantic over keyword:** students phrase the same complaint a dozen ways ("brutal,"
  "weeder," "artificially hard," "exams don't match lecture"); embeddings match on meaning, so a
  query about why a course feels hard retrieves comments with zero shared keywords.
- **Production tradeoffs (for the README reflection):** MiniLM is cheap and private but English-only
  with a small context window; a production deployment weighing larger/API models would trade
  cost and data-egress for better recall on long or multilingual text.

## Evaluation Plan

> Five test questions. Q1–Q4 are grounded in the threads I collected; **fill each expected answer
> directly from the relevant `.txt` file** — these must be checkable against the documents, not
> invented. Q5 is deliberately out-of-scope to surface the refusal/failure case the rubric requires.
> (The **EECS/DS GSI/TA petition** thread is a ready swap-in for any of Q1–Q4 whose expected
> answer turns out not to be verifiable in the docs.)

1. **CS 61A instructor:** "What do r/berkeley posters say about taking CS 61A with John DeNero —
   is he the recommended instructor for the course?"
   *Expected (fill from `reddit_cs61a_denero.txt`):* the consensus the threads actually state, with
   the reasons given.
2. **CS 70 difficulty:** "Why do students say CS 70 feels artificially hard — harder than its content
   warrants?"
   *Expected (fill from `reddit_cs70_difficulty.txt`):* the specific causes posters cite (e.g. exam
   style vs. lecture, pacing, staffing) — answer must list the reasons present in the docs.
3. **Stat 134 Spring 2025 midterm:** "What do students say about the Stat 134 Spring 2025
   midterm?"
   *Expected (fill from the Stat 134 thread):* the specific complaint/observation in that thread.
4. **Hardest class at Cal:** "According to r/berkeley, which courses get named the hardest at
   Berkeley?"
   *Expected (fill from `reddit_hardest_class.txt`):* the courses the threads actually name —
   checkable as a list.
5. **Out-of-scope (refusal/failure case):** "What is the average starting salary for CS 161
   graduates?"
   *Expected:* the system **declines** — "I don't have enough information on that." No thread
   contains salary data, so any specific number is a grounding failure to document, not hide.

If all five come back clean, the questions are too easy — tighten until at least one genuinely strains
retrieval or generation.

## Anticipated Challenges

- **Thread noise:** Reddit markdown, deleted/removed comments, automod/bot replies, off-topic
  tangents within a thread. Cleaning must strip these; chunking must not turn `[deleted]` into a chunk.
- **Comment delimiting (the live risk):** if `collect_reddit.py` didn't separate comments, "one
  comment per chunk" is undefined — this is the open item flagged in Chunking Strategy above.
- **Contradictory opinions:** students disagree about the same course/instructor. Retrieval will
  surface conflicting chunks; the generation prompt must be allowed to report disagreement rather
  than forcing one answer.
- **Sparse coverage:** some topics have far more comments than others; thin-coverage questions
  risk weak retrieval (high distance scores) — Q5 exploits this honestly.
- **Stale opinions:** a comment about a past offering or an instructor no longer teaching the course.
  Metadata keeps the source visible so the answer can be attributed and the user can judge recency.
- **Grounding leakage:** the LLM "knows" generic things about CS courses and may answer from
  training data instead of retrieved text — the system prompt must enforce context-only.

## AI Tool Plan

Using Claude Code as the implementation assistant, prompted *from this spec*, against the lab's
`app.py` / `config.py` template, for these stages:

- **Ingestion + chunking** (Milestone 3): generate the loader, the Reddit-markdown cleaner, the
  metadata extraction, and the comment-boundary splitter (or the 400/80 fallback, per the data
  check) — then I verify against printed sample chunks and correct anything that doesn't match
  this spec.
- **Embedding + ChromaDB + retrieval** (Milestone 4): generate the embed/store/query code; I'll
  ask it to explain any Chroma API I don't recognize before keeping it.
- **Grounded generation + Gradio interface** (Milestone 5): generate the prompt template
  (context-only + refusal clause) and the UI wiring; I'll personally verify the system prompt
  *enforces* grounding and that source attribution is appended programmatically, not left to the model.

I am **not** using AI to write this `planning.md`, choose the domain, or write the evaluation
ground-truth answers — those are my decisions and my verified data.

## Architecture

```
  ┌──────────────────────┐     ┌────────────────────┐     ┌──────────────────────────┐
  │ Document Ingestion    │     │ Chunking            │     │ Embedding + Vector Store  │
  │ collect_reddit.py     │ ──▶ │ comment-boundary    │ ──▶ │ all-MiniLM-L6-v2          │
  │  → Pullpush API        │     │ (1 comment/chunk;   │     │  → ChromaDB (+metadata)   │
  │  → r/berkeley .txt      │     │  400/80 fallback)   │     │                           │
  └──────────────────────┘     └────────────────────┘     └─────────────┬────────────┘
                                                                          │
                                                                          ▼
                      ┌────────────────────────┐     ┌──────────────────────────┐
                      │ Generation              │     │ Retrieval                 │
                      │ Groq                    │ ◀── │ semantic similarity       │
                      │ llama-3.3-70b-versatile  │     │ top-k = 4 + source meta   │
                      │ context-only + cite      │     │                           │
                      └────────────────────────┘     └──────────────────────────┘

  Ingestion → Chunking → Embedding+Store → Retrieval → Generation
```

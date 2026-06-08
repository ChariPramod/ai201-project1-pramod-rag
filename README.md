# The Unofficial Guide — Project 1

A retrieval-augmented question-answering system over **r/berkeley** discussions of UC Berkeley CS / EECS / Data Science courses and instructors. Built on ChromaDB + `all-MiniLM-L6-v2` embeddings + Groq `llama-3.3-70b-versatile`. Source attribution is programmatic (drawn from chunk metadata, not model output). The system refuses **before calling the LLM** when no retrieved chunk meets the cosine-distance threshold, so out-of-scope questions cannot produce hallucinated answers.

---

## Domain

This system makes **student-generated discussion about UC Berkeley CS, EECS, and Data Science courses and instructors** searchable and answerable. The target knowledge is the practical, opinion-based stuff students trade with each other — how hard a course actually is and why, whether a particular instructor or offering is worth taking, what a specific exam was like, which courses get named the hardest on campus — not the official catalog descriptions.

This knowledge is hard to find through normal channels because it lives scattered across hundreds of r/berkeley threads, each partial and written in the moment. No single place gives a student a grounded, cited answer to "is CS 70 as artificially hard as people say, and why?" Collapsing that fragmentation into a single answerable interface is exactly what a retrieval system is good at.

---

## Document Sources

**Collection method.** All 15 documents were collected programmatically by `collect_reddit.py`, which queries the **Pullpush.io archive** (a public mirror of Reddit posts and comments — no auth required) and writes one thread per `.txt` file under `documents/`. Because Pullpush returns structured JSON, cleaning is narrow: strip `[deleted]` / `[removed]` placeholders, drop empty bodies, and dedupe edit-history duplicates with a `(author, body[:200])` fingerprint. There is no HTML/nav/ad stripping because the source is not rendered HTML.

| # | Thread title | Type | URL / file path |
|---|---|---|---|
| 1 | Bio 1A | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/112jlkk/bio_1a/) · `documents/112jlkk-bio-1a.txt` |
| 2 | Your TAs need your help: sign the petition to save EECS/DS | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/11ytb7x/your_tas_need_your_help_sign_the_petition_to_save/) · `documents/11ytb7x-your-tas-need-your-help-sign-the-petition-to-save-eecs-ds.txt` |
| 3 | 61a John DeNero | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/1fjkvcx/61a_john_denero/) · `documents/1fjkvcx-61a-john-denero.txt` |
| 4 | How do professors generally feel about students who turn things around? | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/1giscdd/how_do_professors_generally_feel_about_students/) · `documents/1giscdd-how-do-professors-generally-feel-about-students-who-turn-thi.txt` |
| 5 | Hardest class at Cal | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/1ho1bra/hardest_class_at_cal/) · `documents/1ho1bra-hardest-class-at-cal.txt` |
| 6 | stat 134 midterm being thrown out — grading structure discussion | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/1jw9g48/stat_134_midterm_being_thrown_out_grading/) · `documents/1jw9g48-stat-134-midterm-being-thrown-out-grading-structure-discussi.txt` |
| 7 | Math 53 prof rotation? | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/1kjttzn/math_53_prof_rotation/) · `documents/1kjttzn-math-53-prof-rotation.txt` |
| 8 | After your last final at Berkeley | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/1ko8q2z/after_your_last_final_at_berkeley/) · `documents/1ko8q2z-after-your-last-final-at-berkeley.txt` |
| 9 | What are the worst, most painful classes at Berkeley? | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/u9k613/what_are_the_worst_most_painful_classes_at/) · `documents/u9k613-what-are-the-worst-most-painful-classes-at-berkeley.txt` |
| 10 | [Critique] CS70 is made artificially harder. Here's how it can improve: | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/ura55a/critique_cs70_is_made_artificially_harder_heres/) · `documents/ura55a-critique-cs70-is-made-artificially-harder-here-s-how-it-can-.txt` |
| 11 | Why Money Alone Won't Fix the CS Department | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/uv46rh/why_money_alone_wont_fix_the_cs_department/) · `documents/uv46rh-why-money-alone-won-t-fix-the-cs-department.txt` |
| 12 | Those Freshmans and Transfers who wanna take 20+ units for CoE | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/vqz7qi/those_freshmans_and_transfers_who_wanna_take_20/) · `documents/vqz7qi-those-freshmans-and-transfers-who-wanna-take-20-units-for-co.txt` |
| 13 | Question for students who believe CS/EECS students don't shower | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/w3y61j/question_for_students_who_believe_cseecs_students/) · `documents/w3y61j-question-for-students-who-believe-cs-eecs-students-don-t-sho.txt` |
| 14 | What is the most difficult class in your major or you have ever taken? | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/y37aya/what_is_the_most_difficult_class_in_your_major_or/) · `documents/y37aya-what-is-the-most-difficult-class-in-your-major-or-you-have-e.txt` |
| 15 | When does 61A check for anti plagiarism | Reddit r/berkeley | [reddit](https://www.reddit.com/r/berkeley/comments/za7vb6/when_does_61a_check_for_anti_plagiarism/) · `documents/za7vb6-when-does-61a-check-for-anti-plagiarism.txt` |

The 15 threads span lower-div (CS 61A, Bio 1A, Math 53), upper-div (CS 70, Stat 134), and cross-cutting topics (hardest classes at Cal, the EECS/DS GSI petition, the CS department reform debate) — so the eval set can test both specific-course and campus-wide questions.

---

## Chunking Strategy

**Primary strategy: comment-boundary chunking** — one Reddit comment (or the root post) per chunk. Split on the `^\[u/[^\]]+\]\s*$` regex, which matches the comment-header markers `collect_reddit.py` writes between every comment.

**Long-comment handling.** Any single comment exceeding **250 words** is sentence-aware split into **~200-word pieces with 40-word overlap** by `split_long_passage()` in `ingest.py`. Before this step, the longest single chunk was 864 words; after, the largest chunk is 247 words.

**Fallback for files without the comment marker.** Sentence-aware sliding window, **400 characters with 80-character (20%) overlap**, backing up to a sentence boundary then a word boundary so a chunk never splits mid-word. Not used in practice — every collected file matched the primary path — but kept for malformed inputs.

**Preprocessing before chunking.** Already performed at collection time, not at chunking time:

- `[deleted]` / `[removed]` comments dropped
- Empty bodies dropped
- Duplicate edit-history comments deduped by `(author, body[:200])` fingerprint
- Bot/automod replies skipped (not separately removed in the parser; none survived the score-sorted top-20 cut)

At chunking time, any chunk shorter than `MIN_CHUNK_LEN = 30` characters is dropped — this catches one-word comments like "lol" or "this" that have no retrievable content.

**Metadata per chunk:** `source` (filename), `topic` (thread title), `chunk_index` (position in the file), `is_root` (post vs. comment).

**Why these choices fit the documents.** A Reddit comment is the natural unit of meaning — each is a self-contained opinion from one student, the same way a RateMyProfessors review is. Splitting mid-comment destroys the fact a query is trying to retrieve; merging multiple comments into one chunk dilutes the embedding so no single query matches it cleanly. The 250-word long-comment cap keeps any one chunk small enough to embed cleanly while still preserving a full argument when a comment is essay-length.

**Final chunk count:** **325 chunks** across 15 files (67 marked `is_root=True`, 258 marked `is_root=False`). Largest chunk after long-comment splitting: 247 words.

### Sample chunks (5, labeled)

| # | Source | `is_root` | `chunk_index` | Text |
|---|---|---|---|---|
| 1 | `u9k613-what-are-the-worst-most-painful-classes-at-berkeley.txt` | False | 5 | "He's teaching CS182 next fall and will design the curriculum to be masochist oriented." |
| 2 | `1giscdd-how-do-professors-generally-feel-about-students-who-turn-thi.txt` | False | 14 | "Let me tell you about one of my students. We'll call him Bill, because that's not his name. I was teaching a class with two midterms and a final. He generally had a hard time, and he was pretty slow with a lot of the math. On the first midterm, he got an F. OK, fine. Doesn't make…" (truncated) |
| 3 | `ura55a-critique-cs70-is-made-artificially-harder-here-s-how-it-can-.txt` | False | 26 | "LMFAOOOOOOOOO YOU SPENT 900 HOURS ON THIS CLASS????? thats probably triple the amount of time I spent for my entire 4 years at berkeley on classes combined" |
| 4 | `11ytb7x-your-tas-need-your-help-sign-the-petition-to-save-eecs-ds.txt` | False | 4 | "Why wasn't this bargaining completed during the strike last semester…" |
| 5 | `w3y61j-question-for-students-who-believe-cs-eecs-students-don-t-sho.txt` | True | 0 | "I'm going to start my Berkeley EECS program starting this Fall semester and I noticed too many people on this subreddit saying (or at least implying) that current CS/EECS students don't shower. My question is: What makes you think ALL CS/EECS students don't shower? Is it possib…" (truncated) |

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. Local, no API key, no rate limits, 384-dimensional vectors, fast on CPU. The model is attached directly to the ChromaDB collection as its `embedding_function`, so `add()` and `query()` automatically embed with the same model — text is never embedded manually in the codebase.

**Vector store:** ChromaDB (local, persistent at `./chroma_db`), HNSW index, cosine similarity (`metadata={"hnsw:space": "cosine"}`).

**Search:** top-k = 4. Reddit comments are short, so 4 chunks give the LLM enough corroborating opinions without flooding it with loosely related text.

**Production tradeoff reflection.**

MiniLM-L6-v2 is cheap and private but English-only with a small (~256-token) input window. If this were deployed for real users and cost were not a constraint, I'd weigh:

- **Larger / API-hosted embeddings** (e.g. OpenAI `text-embedding-3-large`, Cohere `embed-v3`, Voyage `voyage-3`): better recall on longer text and on phrasings that don't reuse the document's vocabulary, at the cost of per-query latency and data egress. Worth it here because r/berkeley posts use a lot of campus-specific slang ("weeder," "GSI," "bcourses," "Tele-bears") that a small general-purpose model may underweight.
- **Multilingual support.** Not relevant to r/berkeley itself, but would block any expansion to international student forums.
- **Domain fine-tuning.** A model fine-tuned on student-review text would likely beat any general-purpose model — but only worth the data-collection and ops cost above a few thousand weekly queries.
- **Latency vs. recall.** A re-ranker stage (e.g. a cross-encoder on top-20) would tighten precision on broad queries like "hardest classes" — but doubles the per-query cost.

### Retrieval tests (3 real queries)

Each test shows the top-4 chunks returned by `retriever.retrieve()` along with the cosine distance and the source filename. The first ~200 characters of the chunk are shown.

**Query 1 — narrow, single-source:** *"Why do students say CS 70 feels artificially hard — harder than its content warrants?"*

| Rank | Distance | Source | Chunk preview |
|---|---|---|---|
| 1 | 0.3991 | `ura55a-critique-cs70-…txt` | "When you have difficult topics like Discrete Math and Probability Theory, all the learning comes through struggling and developing your own problem solving techniques. Seeing the answer just simply re…" |
| 2 | 0.4223 | `ura55a-critique-cs70-…txt` | "Agreed but cs70 is on a whole other level. I had an advance math backgroud. One of those kids who declared math their first semeseter. I noticed how poorly supported cs70 was but said nothing to anyon…" |
| 3 | 0.4238 | `ura55a-critique-cs70-…txt` | "I think you bring up a lot of good points, but the teaching issues are harder to fix than you realize. I got an A+ in the class and was a reader for a semester of 70, but the honest truth is I still d…" |
| 4 | 0.4249 | `ura55a-critique-cs70-…txt` | "Former CS70 GSI here. These are all superficial problems. The real problem with CS70 is that it's a monstrosity spawned by intra-departmental politics. They are trying to cram too much material becaus…" |

All four chunks land in the one thread that's explicitly about this topic. Distances cluster tightly (0.40–0.43).

**Query 2 — broad, multi-source:** *"According to r/berkeley, which courses get named the hardest at Berkeley?"*

| Rank | Distance | Source | Chunk preview |
|---|---|---|---|
| 1 | 0.3614 | `1ho1bra-hardest-class-at-cal.txt` | "What's the hardest/most unreasonable course you've ever taken at Cal?? The type of class that made you reconsider your major or question whether there was genuinely something wrong with you…" |
| 2 | 0.3914 | `y37aya-what-is-the-most-difficult-class-…txt` | "I am civil engineering major. My toughest class was civeng 126 Dynamics and Vibration. It was a mixture of Calc 3, DE and Mathlab. First homework was so fucking tough I ended up saying this class has …" |
| 3 | 0.4298 | `ura55a-critique-cs70-…txt` | "Excellent write up! I've had similar feelings regarding classes with artificially inflated difficulty at this school, and you have great suggestions to help prevent this. Classes like CS 70 really sou…" |
| 4 | 0.4512 | `uv46rh-why-money-alone-won-t-fix-the-cs-department.txt` | "All this just sounds like you're complaining about Berkeley's rigor. It's the best CS school in the world, nothing like high school. Just own up to it." |

Retrieval correctly spans the two general "hardest class" threads plus two CS-specific threads — the breadth a "which courses" question warrants.

**Query 3 — narrow, missing fact (Spring 2025):** *"What do students say about the Stat 134 Spring 2025 midterm?"*

| Rank | Distance | Source | Chunk preview |
|---|---|---|---|
| 1 | 0.5232 | `1ho1bra-hardest-class-at-cal.txt` | "Stat 210A. The content isn't that hard but the homeworks are significantly more difficult than what's taught in lecture." |
| 2 | 0.5265 | `1kjttzn-math-53-prof-rotation.txt` | "I had a friend who told me horror stories about Math 53 in the fall semester. I'm taking the class rn in the spring and her practice exams are almost identical to the midterms. Averages are also quite…" |
| 3 | 0.5569 | `1kjttzn-math-53-prof-rotation.txt` | "Fall 2023 the final exam was curved to a C 💀💀" |
| 4 | 0.5576 | `1jw9g48-stat-134-midterm-being-thrown-out-grading-structure-discussi.txt` | "saddened to hear stat 134 still sucks" |

Distances jump up to 0.52–0.56 and the top-ranked chunk is a *different* class (Stat 210A). Only one Stat-134 chunk surfaces, and even it lacks "Spring 2025" specifics. This is the natural failure case the eval exposes.

---

## Grounded Generation

**System prompt grounding instruction** (verbatim from `generator.py:18-24`):

> You are a study guide assistant. Answer the question using ONLY the context provided below. If the context does not contain enough information to answer, respond exactly: 'I don't have enough information on that.' Do not use any outside knowledge. If reviewers disagree, summarize the range of opinions rather than picking one.

**Structural enforcement of grounding** — the prompt is necessary but not sufficient. Three structural mechanisms close the gap:

1. **Distance-threshold pre-filter (`generator.py:48`).** Every retrieved chunk whose cosine distance exceeds `DISTANCE_THRESHOLD = 0.6` is dropped. If no chunk survives, the system returns the refusal string **without ever calling the LLM** — so an out-of-scope question cannot leak training-data knowledge.
2. **Source-labeled context block (`generator.py:31-33`).** Each surviving chunk is wrapped as `[Source: {filename}]\n{text}` before being concatenated into the user message. The model sees which document each fact came from.
3. **Programmatic source attribution (`generator.py:36-41`, `66`).** The "Retrieved from" list shown to the user is built by `_dedupe_sources()` from chunk metadata — the model never writes the source list, so it cannot hallucinate citations or attribute facts to documents that didn't surface in retrieval.

### Example responses

**Grounded example 1 — CS 70 difficulty** (single high-relevance source)

> **Q:** Why do students say CS 70 feels artificially hard — harder than its content warrants?
>
> **A:** According to the sources, students say CS 70 feels artificially hard for several reasons, including:
>
> 1. Poor support for the class, with some students feeling that the teaching is inadequate and that they have to rely on outside resources to understand the material.
> 2. The class is conceptually very difficult, with some students feeling that they don't fully understand the material despite doing well in the class.
> 3. The class tries to cram too much material, due to conflicting requirements from the EE and CS departments, which can make it hard for students to keep up.
> 4. The class does not provide rigorous definitions of concepts, and jumps straight into complex proofs without fully exploring the properties of the concepts, leaving students in a "mathematical haze".
>
> Overall, the sources suggest that the difficulty of CS 70 is not just due to the inherent complexity of the material, but also due to the way the class is structured and taught.
>
> **Retrieved from:** `ura55a-critique-cs70-is-made-artificially-harder-here-s-how-it-can-.txt`

**Grounded example 2 — Hardest classes at Cal** (multi-source, opinion synthesis)

> **Q:** According to r/berkeley, which courses get named the hardest at Berkeley?
>
> **A:** According to the provided context, the following courses are mentioned as being particularly hard at Berkeley:
>
> 1. Civeng 126 Dynamics and Vibration (Civil Engineering major)
> 2. CE 191 Optimization (Civil Engineering major)
> 3. CS 70 (Computer Science major)
>
> Note that opinions may vary, and not all reviewers may agree on the difficulty of these courses, with one user suggesting that some classes, like CS 70, are made "artificially harder". Another user dismisses complaints about difficulty, stating that Berkeley's rigor is a reflection of its status as a top CS school.
>
> **Retrieved from:** `1ho1bra-hardest-class-at-cal.txt`, `y37aya-what-is-the-most-difficult-class-in-your-major-or-you-have-e.txt`, `ura55a-critique-cs70-is-made-artificially-harder-here-s-how-it-can-.txt`, `uv46rh-why-money-alone-won-t-fix-the-cs-department.txt`

**Refusal example — out-of-scope question**

> **Q:** What is the average starting salary for CS 161 graduates?
>
> **A:** I don't have enough information on that.
>
> **Retrieved from:** (none returned to the user — chunks were retrieved but the LLM honored the refusal directive because no chunk contains salary data.)

---

## Query Interface

A local **Gradio Blocks** web app at `http://127.0.0.1:7860` (launched by `python app.py`). Layout: a two-line question textbox with placeholder example, an **Ask** button, an Answer box, and a "Retrieved from" box listing the source filenames that contributed to the answer. Submitting via Enter or the Ask button calls `handle_query()`, which strips the question, calls `generate_response()`, formats the dedup'd source list with bullets, and returns both to the UI.

### Example transcript

```
> Your question:
  Why do students say CS 70 feels artificially hard — harder than its content warrants?

> Answer:
  According to the sources, students say CS 70 feels artificially hard for several reasons,
  including:
  1. Poor support for the class, with some students feeling that the teaching is inadequate…
  2. The class is conceptually very difficult, with some students feeling that they don't
     fully understand the material despite doing well in the class.
  3. The class tries to cram too much material, due to conflicting requirements from the EE
     and CS departments, which can make it hard for students to keep up.
  4. The class does not provide rigorous definitions of concepts, and jumps straight into
     complex proofs without fully exploring the properties of the concepts, leaving students
     in a "mathematical haze".

> Retrieved from:
  • ura55a-critique-cs70-is-made-artificially-harder-here-s-how-it-can-.txt
```

---

## Evaluation Report

Generated by `python evaluate.py` against the 5 questions from `planning.md` (Evaluation Plan). The "Expected answer" cells reproduce the placeholder descriptions in `planning.md` — final ground-truth answers will be filled in by the reviewer after cross-checking each `.txt` source.

| # | Question | Expected answer (from planning.md) | System response | Retrieval quality | Response accuracy |
|---|---|---|---|---|---|
| 1 | What do r/berkeley posters say about taking CS 61A with John DeNero — is he the recommended instructor for the course? | (planning.md Q1 — fill from `1fjkvcx-61a-john-denero.txt`) The consensus the threads actually state, with the reasons given. | "The provided context only mentions John DeNero in relation to CS 61A in one source. According to that source, the poster encourages and supports students taking CS 61A, stating it's an 'amazing course' and that the assignments are 'pretty much hand-holding'. However, it does not explicitly compare John DeNero to other instructors or state that he is the recommended instructor for the course. It simply offers encouragement and advice for students taking the course with him."<br><br>Retrieved: `u9k613-…-painful-classes…txt` (0.4844), `uv46rh-…-cs-department.txt` (0.4965), `vqz7qi-…-20-units-for-co.txt` (0.5064), `1fjkvcx-61a-john-denero.txt` (0.5229) | <!-- TODO: mine --> | <!-- TODO: mine --> |
| 2 | Why do students say CS 70 feels artificially hard — harder than its content warrants? | (planning.md Q2 — fill from `ura55a-…-cs70-…txt`) The specific causes posters cite (e.g. exam style vs. lecture, pacing, staffing) — answer must list the reasons present in the docs. | "According to the sources, students say CS 70 feels artificially hard for several reasons: (1) poor support for the class … (2) the class is conceptually very difficult … (3) the class tries to cram too much material, due to conflicting requirements from the EE and CS departments … (4) the class does not provide rigorous definitions of concepts, jumps straight into complex proofs … leaving students in a 'mathematical haze'."<br><br>Retrieved: 4× `ura55a-…-cs70-…txt` (0.3991, 0.4223, 0.4238, 0.4249) | <!-- TODO: mine --> | <!-- TODO: mine --> |
| 3 | What do students say about the Stat 134 Spring 2025 midterm? | (planning.md Q3 — fill from the Stat 134 thread) The specific complaint/observation in that thread. | "I don't have enough information on that."<br><br>Retrieved: `1ho1bra-hardest-class-at-cal.txt` (0.5232), `1kjttzn-math-53-prof-rotation.txt` (0.5265), `1kjttzn-math-53-prof-rotation.txt` (0.5569), `1jw9g48-stat-134-midterm-being-thrown-out-grading-structure-discussi.txt` (0.5576) | <!-- TODO: mine --> | <!-- TODO: mine --> |
| 4 | According to r/berkeley, which courses get named the hardest at Berkeley? | (planning.md Q4 — fill from `1ho1bra-hardest-class-at-cal.txt`) The courses the threads actually name — checkable as a list. | "According to the provided context, the following courses are mentioned as being particularly hard at Berkeley: (1) Civeng 126 Dynamics and Vibration … (2) CE 191 Optimization … (3) CS 70 … Note that opinions may vary … one user suggesting that some classes, like CS 70, are made 'artificially harder'. Another user dismisses complaints about difficulty…"<br><br>Retrieved: `1ho1bra-hardest-class-at-cal.txt` (0.3614), `y37aya-…-difficult-class…txt` (0.3914), `ura55a-…-cs70-…txt` (0.4298), `uv46rh-…-cs-department.txt` (0.4512) | <!-- TODO: mine --> | <!-- TODO: mine --> |
| 5 | What is the average starting salary for CS 161 graduates? | (planning.md Q5, out-of-scope) The system declines: "I don't have enough information on that." | "I don't have enough information on that."<br><br>Retrieved: `11ytb7x-…-petition-to-save-eecs-ds.txt` (0.5085), `u9k613-…-painful-classes…txt` (0.5277), `u9k613-…-painful-classes…txt` (0.5689), `u9k613-…-painful-classes…txt` (0.5973) | <!-- TODO: mine --> | <!-- TODO: mine --> |

**Retrieval quality scale:** Relevant / Partially relevant / Off-target
**Response accuracy scale:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** What do students say about the Stat 134 Spring 2025 midterm?

**What the system returned:** "I don't have enough information on that."

**Root cause (tied to a specific pipeline stage):** The thread is literally about the midterm being thrown out, so the answer is in the corpus but retrieval surfaced a contentless one liner ("saddened to hear stat 134 still sucks") plus three chunks from unrelated classes (Stat 210A, Math 53) that scored closer to the query than the substantive Stat 134 content. "Spring 2025" appears in no chunk, so the temporal qualifier can't anchor the match. With nothing usable in the top 4, the model correctly refused rather than guess.

**What you would change to fix it:** hybrid BM25 search (the literal token "stat 134" would boost the right thread), as hybrid search is what the hybrid stretch feature fixes or a re ranker over a larger top k, or metadata/topic filtering to bias toward the Stat 134 thread.

---

## Spec Reflection

**One way the spec helped you during implementation:** writing the comment-boundary chunking decision in planning.md before coding gave a precise brief, so ingestion matched the Reddit-thread structure on the first pass instead of defaulting to naive fixed-size splitting.

**One way your implementation diverged from the spec, and why:** planning.md specified course/instructor metadata (RMP-style), but the data turned out to be Reddit threads, not per-instructor reviews so the implementation uses topic (thread title) + is_root (post vs. comment) instead.

---

## AI Usage

**Instance 1**
I directed Claude Code to implement generate_response() with the grounding system prompt included verbatim, and verified the "ONLY"/refusal directive wasn't softened during code generation (the known failure mode where AI rephrases prompt instructions).

**Instance 2**
The 0.6 distance threshold, I kept it at 0.6 rather than dropping it to ~0.5, specifically so it wouldn't mask the Q3 retrieval failure. That's a genuine override of an AI suggestion.

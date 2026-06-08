"""
Load and chunk r/berkeley thread documents.

Structure produced by collect_reddit.py:

    # <Thread title>
    Subreddit: r/berkeley
    Posted by: u/<author>
    Score: <N>    Comments: <N>
    URL: ...

    ## Original post
    <post body>

    ## Top comments

    [u/<author> — score <N>]
    <comment body>

    [u/<author> — score <N>]
    <comment body>
    ...

Primary chunking strategy: one comment (or the root post) per chunk, split on the
`[u/<author> — score <N>]` delimiter. If a file lacks that pattern we fall back to
a sentence-aware sliding window using CHUNK_SIZE / CHUNK_OVERLAP from config.
"""

import glob
import os
import random
import re

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

COMMENT_HEADER_RE = re.compile(r"^\[u/[^\]]+\]\s*$", re.MULTILINE)
POST_HEADER_RE = re.compile(r"^##\s*Original post\s*$", re.MULTILINE)
COMMENTS_HEADER_RE = re.compile(r"^##\s*Top comments\s*$", re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MIN_CHUNK_LEN = 30
LONG_COMMENT_WORDS = 250
SPLIT_TARGET_WORDS = 200
SPLIT_OVERLAP_WORDS = 40


def parse_topic(text: str) -> str | None:
    """Return the first single-hash heading, stripped, or None."""
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#") and not s.startswith("##"):
            return s.lstrip("#").strip()
    return None


def split_by_comment_delimiter(text: str) -> list[dict] | None:
    """Split into [{text, is_root=True (post)}, {text, is_root=False (comment)}, ...].
    Return None if no delimiter present."""
    matches = list(COMMENT_HEADER_RE.finditer(text))
    if len(matches) < 2:
        return None

    pieces: list[dict] = []

    post_section = POST_HEADER_RE.search(text)
    if post_section:
        post_start = post_section.end()
        post_end = matches[0].start()
        marker = COMMENTS_HEADER_RE.search(text, post_start, post_end)
        if marker:
            post_end = marker.start()
        post_text = text[post_start:post_end].strip()
        if post_text:
            pieces.append({"text": post_text, "is_root": True})

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            pieces.append({"text": body, "is_root": False})

    return pieces


def split_long_passage(text: str, target_words: int, overlap_words: int) -> list[str]:
    """Sentence-aware split for any single comment > LONG_COMMENT_WORDS. Each
    output piece runs ~target_words with overlap_words carried from the prior piece."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    if not sentences:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for sent in sentences:
        sw = len(sent.split())
        if current and current_words + sw > target_words:
            pieces.append(" ".join(current))
            overlap_buf: list[str] = []
            buf_words = 0
            for s in reversed(current):
                w = len(s.split())
                if buf_words + w > overlap_words:
                    break
                overlap_buf.insert(0, s)
                buf_words += w
            current = overlap_buf
            current_words = buf_words
        current.append(sent)
        current_words += sw

    if current:
        pieces.append(" ".join(current))

    return pieces


def sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware sliding window. Backs up to a sentence end, then a word
    boundary, before splitting — never splits mid-word."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + size, n)
        if end < n:
            search_from = i + size // 2
            sentence_end = max(
                text.rfind(". ", search_from, end),
                text.rfind("! ", search_from, end),
                text.rfind("? ", search_from, end),
                text.rfind(".\n", search_from, end),
                text.rfind("!\n", search_from, end),
                text.rfind("?\n", search_from, end),
            )
            if sentence_end != -1:
                end = sentence_end + 1
            else:
                ws = text.rfind(" ", search_from, end)
                if ws != -1:
                    end = ws

        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        next_i = end - overlap
        if next_i <= i:
            next_i = end
        i = next_i

    return chunks


def load_and_chunk() -> list[dict]:
    """Load every .txt in DOCS_DIR, chunk it, and return chunks with metadata."""
    all_chunks: list[dict] = []
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))

    for path in paths:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        topic = parse_topic(text) or filename

        delimited = split_by_comment_delimiter(text)
        if delimited is None:
            raw_pieces = [
                {"text": t, "is_root": False}
                for t in sliding_window(text, CHUNK_SIZE, CHUNK_OVERLAP)
            ]
        else:
            raw_pieces = []
            for p in delimited:
                if len(p["text"].split()) > LONG_COMMENT_WORDS:
                    for sub in split_long_passage(
                        p["text"], SPLIT_TARGET_WORDS, SPLIT_OVERLAP_WORDS
                    ):
                        raw_pieces.append({"text": sub, "is_root": p["is_root"]})
                else:
                    raw_pieces.append(p)

        chunk_index = 0
        for piece in raw_pieces:
            if len(piece["text"]) < MIN_CHUNK_LEN:
                continue
            all_chunks.append(
                {
                    "text": piece["text"],
                    "metadata": {
                        "source": filename,
                        "topic": topic,
                        "chunk_index": chunk_index,
                        "is_root": piece["is_root"],
                    },
                }
            )
            chunk_index += 1

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Total chunks: {len(chunks)}\n")

    sample = random.sample(chunks, min(5, len(chunks)))
    for i, c in enumerate(sample, 1):
        print(f"--- Sample {i} ---")
        print(f"metadata: {c['metadata']}")
        print(f"text ({len(c['text'])} chars):")
        print(c["text"])
        print()

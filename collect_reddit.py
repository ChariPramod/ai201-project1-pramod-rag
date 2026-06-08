"""
Pull r/berkeley threads about professors / courses via the Pullpush.io archive.
Each surviving thread is written to documents/<slug>.txt as plain text.

Pullpush is a community-run Reddit archive — no auth required. We throttle requests.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

SUBREDDIT = "berkeley"
QUERIES = [
    "professor review",
    "best professor",
    "worst professor",
    "easy A",
    "hardest class",
    "professor recommendation",
    "61A professor",
    "EECS professor",
    "math professor",
    "good professor",
]
MIN_COMMENTS = 10
MAX_DOCS = 15
PER_QUERY = 25
OUTPUT_DIR = "documents"
USER_AGENT = "ai201-unofficial-guide-collector/0.1"
SLEEP = 1.0


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_threads(query: str) -> list[dict]:
    q = urllib.parse.quote(query)
    url = (
        f"https://api.pullpush.io/reddit/search/submission/"
        f"?subreddit={SUBREDDIT}&q={q}&size={PER_QUERY}"
        f"&sort=desc&sort_type=num_comments"
    )
    return fetch_json(url).get("data", [])


def fetch_comments(thread_id: str) -> list[dict]:
    url = (
        f"https://api.pullpush.io/reddit/search/comment/"
        f"?link_id={thread_id}&size=100&sort=desc&sort_type=score"
    )
    return fetch_json(url).get("data", [])


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:60] or "thread"


def render(post: dict, comments: list[dict]) -> str:
    lines = [
        f"# {(post.get('title') or '').strip()}",
        f"Subreddit: r/{SUBREDDIT}",
        f"Posted by: u/{post.get('author', '[deleted]')}",
        f"Score: {post.get('score', 0)}    Comments: {post.get('num_comments', 0)}",
        f"URL: https://www.reddit.com{post.get('permalink', '')}",
        "",
        "## Original post",
        ((post.get("selftext") or "").strip() or "(link post, no body text)"),
        "",
        "## Top comments",
    ]
    kept = 0
    seen_bodies: set[str] = set()
    for c in comments:
        body = (c.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        fingerprint = (c.get("author", ""), body[:200])
        if fingerprint in seen_bodies:
            continue
        seen_bodies.add(fingerprint)
        lines.append(f"\n[u/{c.get('author', '[deleted]')} — score {c.get('score', 0)}]")
        lines.append(body)
        kept += 1
        if kept >= 20:
            break
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for q in QUERIES:
        print(f"  search: {q!r}")
        try:
            results = search_threads(q)
        except Exception as exc:
            print(f"    error: {exc}")
            time.sleep(SLEEP)
            continue
        for r in results:
            tid = r.get("id")
            if not tid or tid in seen_ids:
                continue
            if r.get("num_comments", 0) < MIN_COMMENTS:
                continue
            if not (r.get("selftext") or "").strip():
                # Skip pure link posts; we want text-rich docs
                continue
            seen_ids.add(tid)
            candidates.append(r)
        time.sleep(SLEEP)

    print(f"\n{len(candidates)} candidate threads after filtering.")
    candidates.sort(key=lambda r: r.get("num_comments", 0), reverse=True)
    picked = candidates[:MAX_DOCS]

    saved = 0
    for r in picked:
        tid = r["id"]
        slug = slugify(r.get("title", tid))
        path = os.path.join(OUTPUT_DIR, f"{tid}-{slug}.txt")
        try:
            comments = fetch_comments(tid)
        except Exception as exc:
            print(f"  skip {tid}: {exc}")
            time.sleep(SLEEP)
            continue
        text = render(r, comments)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        saved += 1
        print(f"  wrote {path} ({len(text)} chars, {len(comments)} comments)")
        time.sleep(SLEEP)

    print(f"\nDone. {saved} documents in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

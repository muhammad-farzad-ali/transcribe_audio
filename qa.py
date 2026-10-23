#!/usr/bin/env python3
"""Sanity checks for generated transcripts.

Usage: .venv/bin/python qa.py transcript/data_engineering_complete.json
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def srt_stats(path: Path) -> dict[str, float | int]:
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    cue_re = re.compile(
        r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)"
    )

    def secs(groups: tuple[str, ...]) -> float:
        h, m, s, ms = map(int, groups)
        return h * 3600 + m * 60 + s + ms / 1000

    times: list[tuple[float, float]] = []
    chars: list[int] = []
    for block in blocks:
        lines = block.split("\n")
        match = cue_re.search(lines[1]) if len(lines) > 1 else None
        if not match:
            continue
        times.append((secs(match.groups()[:4]), secs(match.groups()[4:])))
        chars.append(len(" ".join(lines[2:])))
    return {
        "cues": len(times),
        "chars_median": statistics.median(chars) if chars else 0,
        "chars_max": max(chars) if chars else 0,
        "monotonic": all(b[0] >= a[0] for a, b in zip(times, times[1:])),
        "overlaps": sum(1 for a, b in zip(times, times[1:]) if a[1] > b[0] + 1e-6),
        "short_cues": sum(1 for a, b in times if b - a < 0.6),
        "last_end": times[-1][1] if times else 0.0,
    }


def main(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data["segments"]
    text = " ".join(s["text"] for s in segments)
    words = len(text.split())
    stops = len(re.findall(r"[.!?](?=\s|$)", text))

    overlaps = sum(1 for a, b in zip(segments, segments[1:]) if a["end"] > b["start"] + 1e-6)
    out_of_order = sum(1 for a, b in zip(segments, segments[1:]) if b["start"] < a["start"])
    dup_text = Counter(s["text"].strip().lower() for s in segments)
    repeats = [(t, n) for t, n in dup_text.most_common(3) if n > 5]
    bigram_dups = re.findall(r"\b(\w+\s+\w+\s+\w+)\s+\1\b", text)

    duration = data["meta"]["duration_seconds"]
    audio = probe_duration(path.with_suffix(".m4a"))
    print(f"segments            {len(segments):,}")
    print(f"words               {words:,} ({words / max(duration, 1) * 60:.0f} wpm)")
    print(f"stops per 1k words  {stops / max(words, 1) * 1000:.1f}")
    print(f"timeline overlaps   {overlaps}")
    print(f"out of order        {out_of_order}")
    print(f"duplicate segments  {repeats or 'none'}")
    print(f"triple-word repeats {len(bigram_dups)}")
    print(f"coverage            {segments[-1]['end'] / duration * 100:.2f}% of {duration / 3600:.2f} h")
    if audio:
        print(f"audio duration      {audio:.1f}s (probed)")

    srt = path.with_suffix(".srt")
    if srt.exists():
        print("srt                 " + json.dumps(srt_stats(srt)))

    ok = overlaps == 0 and out_of_order == 0 and not repeats
    print("\nresult:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "transcript/transcript.json")))

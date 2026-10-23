#!/usr/bin/env python3
"""Chunked, resumable long-form transcription with faster-whisper.

Designed for multi-hour recordings on a single consumer GPU:

  * splits audio into ~20 minute chunks at the quietest nearby point
  * writes one JSON per chunk, so a crash never costs more than one chunk
  * merges chunks back into a single timeline, removing overlap duplicates
  * emits txt / srt / vtt / json / timestamped markdown
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
import wave
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# CUDA shared libraries shipped as pip wheels
# --------------------------------------------------------------------------- #
def _ensure_cuda_libs() -> None:
    """Make pip-installed cuBLAS/cuDNN discoverable by CTranslate2.

    Wheels live under site-packages/nvidia/*/lib, which the dynamic loader
    does not search. We add them to LD_LIBRARY_PATH and preload them with
    RTLD_GLOBAL, so CTranslate2's own dlopen() calls resolve.
    """
    if os.environ.get("_FW_CUDA_LIBS") == "1":
        return
    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    lib_dirs = [d for d in (site / "cublas" / "lib", site / "cudnn" / "lib") if d.is_dir()]
    if not lib_dirs:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(d) for d in lib_dirs] + [os.environ.get("LD_LIBRARY_PATH", "")]
    ).strip(os.pathsep)

    import ctypes

    # Order matters a little: base libs before the cudnn sub-libraries.
    patterns = ("libcublasLt.so*", "libcublas.so*", "libcudnn.so*", "libcudnn_*.so*")
    loaded: set[str] = set()
    for pattern in patterns:
        for lib_dir in lib_dirs:
            for so in sorted(lib_dir.glob(pattern)):
                if so.name in loaded:
                    continue
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    loaded.add(so.name)
                except OSError:
                    pass
    os.environ["_FW_CUDA_LIBS"] = "1"


_ensure_cuda_libs()
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))

import numpy as np  # noqa: E402
from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402

SILENCE_RE = re.compile(r"silence_(start|end):\s*([0-9.]+)")


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #
def run(cmd: list[str], quiet: bool = True) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc.stderr if quiet else proc.stdout


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def find_quiet_cut(src: Path, target: float, window: float, threshold_db: float, min_silence: float) -> float:
    """Return the centre of the longest silence near `target`, else `target`."""
    start = max(0.0, target - window)
    span = window * 2
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
        "-i", str(src), "-vn",
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    log = run(cmd)
    starts = [float(m.group(2)) for m in SILENCE_RE.finditer(log) if m.group(1) == "start"]
    ends = [float(m.group(2)) for m in SILENCE_RE.finditer(log) if m.group(1) == "end"]
    best: tuple[float, float] | None = None
    for s, e in zip(starts, ends):
        if e > s and (best is None or (e - s) > (best[1] - best[0])):
            best = (s, e)
    if best is None:
        return target
    return start + (best[0] + best[1]) / 2


def plan_chunks(
    src: Path,
    duration: float,
    chunk_len: float,
    overlap: float,
    min_chunk: float,
    cut_window: float,
    cut_noise: float,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    idx = 0
    start = 0.0
    while start < duration - 1.0:
        nominal = start + chunk_len
        if duration - nominal < min_chunk:
            end = duration
        else:
            end = find_quiet_cut(src, nominal, cut_window, cut_noise, 0.6)
            if end <= start + min_chunk:
                end = min(duration, nominal)
        end = min(end, duration)
        chunks.append(
            {
                "index": idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            }
        )
        if verbose:
            print(
                f"  chunk {idx:02d}: {fmt_ts(start)} -> {fmt_ts(end)}  "
                f"({(end - start) / 60:.1f} min)",
                flush=True,
            )
        idx += 1
        if end >= duration - 0.5:
            break
        start = max(end - overlap, start + min_chunk)
    return chunks


def extract_wav(src: Path, dst: Path, start: float, end: float) -> None:
    duration = max(0.05, end - start)
    run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
        ]
    )


# --------------------------------------------------------------------------- #
# formatting / text helpers
# --------------------------------------------------------------------------- #
def fmt_ts(seconds: float, sep: str = ",") -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_clock(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, seconds))))


def fmt_hms(seconds: float) -> str:
    return fmt_ts(seconds, sep=".")[:-4]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


# Conservative normalisation of domain words the ASR hears but often writes in
# lowercase. Only applied to standalone tokens, never inside existing words.
PROPER_TERMS = {
    "clean tracks": "CleanTracks",
    "adrian": "Adrian",
    "sql": "SQL",
    "aws": "AWS",
    "s3": "S3",
    "iam": "IAM",
    "acl": "ACL",
    "api": "API",
    "apis": "APIs",
    "json": "JSON",
    "csv": "CSV",
    "python": "Python",
    "numpy": "NumPy",
    "pyspark": "PySpark",
    "spark": "Spark",
    "hadoop": "Hadoop",
    "kafka": "Kafka",
    "airflow": "Airflow",
    "snowflake": "Snowflake",
    "bigquery": "BigQuery",
    "redshift": "Redshift",
    "databricks": "Databricks",
    "parquet": "Parquet",
    "avro": "Avro",
    "mongodb": "MongoDB",
    "pymongo": "PyMongo",
    "sqlalchemy": "SQLAlchemy",
    "pytest": "PyTest",
    "jupyter": "Jupyter",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "github": "GitHub",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "sqlite": "SQLite",
}


def polish_text(text: str, capitalize: bool = False) -> str:
    """Cosmetic cleanup: sentence capitals, pronoun I, canonical tech nouns."""
    if capitalize and text:
        text = text[0].upper() + text[1:]
    text = re.sub(r"\bi'(m|ve|ll|d|re)\b", lambda m: "I'" + m.group(1), text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(
        r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
    )
    for src, dst in PROPER_TERMS.items():
        text = re.sub(r"\b" + re.escape(src) + r"\b", dst, text, flags=re.IGNORECASE)
    return text


def polish_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    capitalize = True
    for seg in segments:
        seg["text"] = polish_text(seg["text"], capitalize)
        capitalize = seg["text"].rstrip()[-1:] in ".!?"
    return segments


def similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity, cheap and good enough for dedupe."""
    ta, tb = set(norm_text(a).split()), set(norm_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #
def load_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    t0 = time.time()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        num_workers=1,
        cpu_threads=os.cpu_count() or 4,
    )
    print(f"[model] {model_name} on {device}/{compute_type} loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def load_wav_mono16k(path: Path) -> np.ndarray:
    """Read our own 16 kHz mono PCM16 wav into float32 [-1, 1].

    Done directly instead of via PyAV so the pipeline does not depend on
    PyAV/FFmpeg API churn (faster-whisper's decoder needs a newer/older av).
    """
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getframerate(), w.getsampwidth()) != (1, 16000, 2):
            raise ValueError(f"unexpected wav format in {path}")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_wav(
    pipe: BatchedInferencePipeline | WhisperModel,
    wav: Path,
    language: str,
    prompt: str | None,
    beam_size: int,
    batch_size: int,
    vad: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Transcribe one wav file, retrying with a smaller batch on CUDA OOM."""
    audio = load_wav_mono16k(wav)
    batch = batch_size
    for attempt in range(4):
        t0 = time.time()
        try:
            segments, _info = pipe.transcribe(
                audio,
                language=language,
                task="transcribe",
                beam_size=beam_size,
                batch_size=batch,
                vad_filter=vad,
                vad_parameters=None,  # library defaults: keeps more context per utterance
                condition_on_previous_text=False,
                initial_prompt=prompt,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=False,
            )
            out = []
            for seg in segments:
                text = clean_text(seg.text)
                if not text:
                    continue
                out.append({"start": seg.start, "end": seg.end, "text": text})
            return out, time.time() - t0
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch <= 1:
                raise
            batch = max(1, batch // 2)
            gc.collect()
            print(f"[warn] CUDA OOM, retrying with batch_size={batch}", flush=True)
    raise RuntimeError("transcription failed after OOM retries")


def chunk_path(workdir: Path, index: int, suffix: str) -> Path:
    return workdir / "chunks" / f"chunk_{index:04d}.{suffix}"


def load_done_chunk(path: Path, chunk: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data.get("segments"), list):
        return None
    if abs(float(data.get("start", -1)) - chunk["start"]) > 0.01:
        return None
    if abs(float(data.get("end", -1)) - chunk["end"]) > 0.01:
        return None
    return data


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    out = []
    for tok in text.split():
        tok = re.sub(r"^[^\w']+|[^\w']+$", "", tok).lower()
        if tok:
            out.append(tok)
    return out


def strip_duplicated_prefix(text: str, prev_text: str, max_words: int = 80) -> tuple[str, float]:
    """Remove words at the start of `text` that repeat the tail of `prev_text`.

    Chunk overlap means the new chunk usually re-transcribes the last few
    seconds of the previous one; only the genuinely new tail is kept.
    Tolerates small ASR differences ("BMS" vs "RDBMS") via token similarity.
    Returns the remaining text and the fraction of the segment it covers.
    """
    new = _tokens(text)
    prev = _tokens(prev_text)[-max_words:]
    if len(new) < 4 or len(prev) < 4:
        return text, 1.0
    words = text.split()
    best_k, best_sim = 0, 0.0
    for k in range(min(len(new), len(prev)), 3, -1):
        sim = similarity(" ".join(new[:k]), " ".join(prev[-k:]))
        if sim >= 0.75 and (sim > best_sim or (sim == best_sim and k > best_k)):
            best_k, best_sim = k, sim
    if not best_k:
        return text, 1.0
    return " ".join(words[best_k:]).strip(), max(0.05, 1.0 - best_k / len(new))


def common_suffix_prefix(a: str, b: str, max_words: int = 80) -> int:
    """Number of trailing words of `a` that also open `b` (fuzzy)."""
    aw = _tokens(a)[-max_words:]
    bw = _tokens(b)[:max_words]
    best_k, best_sim = 0, 0.0
    for k in range(min(len(aw), len(bw)), 1, -1):
        sim = similarity(" ".join(aw[-k:]), " ".join(bw[:k]))
        if sim >= 0.75 and (sim > best_sim or (sim == best_sim and k > best_k)):
            best_k, best_sim = k, sim
    return best_k


def merge_chunks(chunks: list[dict[str, Any]], overlap: float, verbose: bool = True) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for ci, chunk in enumerate(chunks):
        cs = float(chunk["start"])
        segs = chunk["segments"]
        # The previous chunk was cut mid-audio at its boundary, so its trailing
        # segments (inside this chunk's overlap) can be truncated mid-sentence.
        # The new chunk re-transcribes that audio from a clean window, so let it
        # own the overlap and drop the previous tail.
        if ci > 0 and merged and segs and float(segs[0]["start"]) <= cs + 2.0:
            new_head = clean_text(segs[0]["text"])
            last = merged[-1]
            if last["end"] > cs:
                k = common_suffix_prefix(last["text"], new_head)
                if k:
                    words = last["text"].split()
                    kept = words[:-k] if k < len(words) else []
                    if kept:
                        last["text"] = " ".join(kept)
                        last["end"] = cs
                    else:
                        merged.pop()
                else:
                    last["end"] = max(last["start"] + 0.2, cs)
            merged = [s for s in merged if s["start"] < cs - 0.05]
        prev_tail = " ".join(s["text"] for s in merged[-6:])
        for seg in segs:
            s = float(seg["start"])
            e = float(seg["end"])
            text = clean_text(seg["text"])
            if not text:
                continue
            # A segment starting before this chunk's start straddles the overlap;
            # strip the words already covered by the previous chunk.
            if ci > 0 and s < cs - 0.05:
                stripped, frac = strip_duplicated_prefix(text, prev_tail)
                if stripped:
                    text = stripped
                    s = e - (e - s) * frac
                elif similarity(text, prev_tail) >= 0.6:
                    continue
            # collapse immediate repeats (classic Whisper loop)
            if merged:
                last = merged[-1]
                if norm_text(last["text"]) and norm_text(last["text"]) == norm_text(text):
                    last["end"] = max(last["end"], e)
                    continue
                if s - last["end"] < 2.0 and similarity(last["text"], text) > 0.85:
                    last["end"] = max(last["end"], e)
                    continue
            merged.append({"start": s, "end": e, "text": text, "chunk": ci})
        if verbose:
            print(
                f"  merged chunk {ci:02d}  +{len(segs):4d} segs -> {len(merged):5d} total",
                flush=True,
            )
    merged.sort(key=lambda x: x["start"])
    return merged


def stitch_broken(segments: list[dict[str, Any]]) -> None:
    """Join segments that were cut mid-sentence across chunk boundaries."""
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        if cur["start"] - prev["end"] > 1.5:
            continue
        if prev["text"] and prev["text"][-1] in ".!?;:\"'”’":
            continue
        if not cur["text"] or not cur["text"][0].islower():
            continue
        if len(prev["text"]) > 60 or len(cur["text"]) > 60:
            continue
        prev["text"] = prev["text"].rstrip() + " " + cur["text"].lstrip()
        prev["end"] = cur["end"]
        cur["text"] = ""


def fix_overlaps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clamp cue ends so no two entries overlap in the subtitle timeline."""
    for a, b in zip(segments, segments[1:]):
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    return [s for s in segments if s["end"] > s["start"]]


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def pack_cues(text: str, max_chars: int) -> list[str]:
    """Greedy word packing into subtitle-sized cues."""
    out: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            out.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        out.append(" ".join(current))
    return out


def split_for_subtitles(segments: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Split paragraph-length ASR segments into readable cues.

    Cue times are interpolated by character share inside the source segment,
    which keeps subtitles aligned without needing word-level timestamps.
    """
    cues: list[dict[str, Any]] = []
    for seg in segments:
        pieces = pack_cues(seg["text"], max_chars)
        if not pieces:
            continue
        total = sum(len(p) for p in pieces)
        span = max(0.0, seg["end"] - seg["start"])
        cursor = seg["start"]
        for i, piece in enumerate(pieces):
            share = len(piece) / total if total else 1 / len(pieces)
            end = seg["end"] if i == len(pieces) - 1 else cursor + span * share
            cues.append({"start": round(cursor, 3), "end": round(max(end, cursor + 0.35), 3), "text": piece})
            cursor = end
    # avoid cumulative rounding drift pushing a cue past the next one
    for i in range(len(cues) - 1):
        if cues[i]["end"] > cues[i + 1]["start"]:
            cues[i]["end"] = cues[i + 1]["start"]
    return [c for c in cues if c["end"] > c["start"]]


def write_outputs(
    segments: list[dict[str, Any]],
    outdir: Path,
    stem: str = "transcript",
    subtitle_max_chars: int = 84,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    segments = [s for s in segments if s["text"]]
    cues = split_for_subtitles(segments, subtitle_max_chars)

    srt = []
    for i, s in enumerate(cues, 1):
        srt.append(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n")
    (outdir / f"{stem}.srt").write_text("\n".join(srt), encoding="utf-8")

    vtt = ["WEBVTT", ""]
    for i, s in enumerate(cues, 1):
        vtt.append(f"{i}\n{fmt_ts(s['start'], '.')} --> {fmt_ts(s['end'], '.')}\n{s['text']}\n")
    (outdir / f"{stem}.vtt").write_text("\n".join(vtt), encoding="utf-8")

    (outdir / f"{stem}.txt").write_text(" ".join(s["text"] for s in segments).strip() + "\n", encoding="utf-8")

    payload = {
        "segments": [
            {"id": i, "start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"]}
            for i, s in enumerate(segments)
        ],
        "meta": {
            "segment_count": len(segments),
            "duration_seconds": round(segments[-1]["end"], 2) if segments else 0,
            "word_count": sum(len(s["text"].split()) for s in segments),
        },
    }
    (outdir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # Paragraph-per-~2-minutes markdown: easiest format to skim/read.
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    block_start = segments[0]["start"] if segments else 0.0
    for s in segments:
        if current and (s["start"] - block_start >= 120 or s["start"] - current[-1]["end"] > 2.5):
            blocks.append(current)
            current = []
            block_start = s["start"]
        current.append(s)
    if current:
        blocks.append(current)

    md = [f"# Transcript — {stem}", "", f"Duration: {fmt_clock(payload['meta']['duration_seconds'])} · "
          f"{payload['meta']['word_count']:,} words", ""]
    for block in blocks:
        md.append(f"**[{fmt_hms(block[0]['start'])}]** " + " ".join(x["text"] for x in block))
        md.append("")
    (outdir / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")

    return {ext: outdir / f"{stem}.{ext}" for ext in ("txt", "srt", "vtt", "json", "md")}


def quality_report(segments: list[dict[str, Any]], duration: float) -> str:
    words = sum(len(s["text"].split()) for s in segments)
    speech = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    repeats = 0
    for i in range(3, len(segments)):
        if norm_text(segments[i]["text"]) and len(
            {norm_text(segments[j]["text"]) for j in range(i - 3, i + 1)}
        ) == 1:
            repeats += 1
    empty_runs = 0
    covered = speech / duration if duration else 0
    return (
        f"segments={len(segments):,}  words={words:,}  "
        f"speech_coverage={covered * 100:.1f}%  repeat_loops={repeats}  empty_runs={empty_runs}"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="audio/video file (m4a, mp3, wav, mp4, ...)")
    p.add_argument("--outdir", type=Path, default=Path("transcript"))
    p.add_argument("--workdir", type=Path, default=Path("work"))
    p.add_argument("--stem", default="transcript", help="output file basename")
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="int8_float16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-batch", action="store_true", help="use sequential WhisperModel instead of batched pipeline")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for ctranslate2")
    p.add_argument("--chunk-minutes", type=float, default=20.0)
    p.add_argument("--overlap-seconds", type=float, default=10.0)
    p.add_argument("--min-chunk-minutes", type=float, default=3.0)
    p.add_argument("--cut-window-seconds", type=float, default=45.0)
    p.add_argument("--cut-noise-db", type=float, default=-32.0)
    p.add_argument("--subtitle-max-chars", type=int, default=84, help="max characters per subtitle cue")
    p.add_argument("--prompt", default=None, help="initial prompt / glossary")
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--max-chunks", type=int, default=0)
    p.add_argument("--keep-wav", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore cached chunk results")
    p.add_argument("--merge-only", action="store_true", help="rebuild outputs from cached chunk JSONs")
    p.add_argument("--no-polish", action="store_true", help="skip cosmetic text cleanup (capitals, tech nouns)")
    p.add_argument("--no-vad", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    src: Path = args.input.expanduser().resolve()
    if not src.exists():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "chunks").mkdir(exist_ok=True)
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    duration = probe_duration(src)
    print(f"[input] {src.name}  size={src.stat().st_size / 1e9:.2f} GB  duration={fmt_clock(duration)}", flush=True)

    state_file = args.workdir / "state.json"
    state: dict[str, Any] = {}
    if state_file.exists() and not args.force:
        try:
            state = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            state = {}

    plan_key = {
        "file": str(src),
        "duration": round(duration, 2),
        "chunk_minutes": args.chunk_minutes,
        "overlap": args.overlap_seconds,
        "cut_noise": args.cut_noise_db,
    }
    if state.get("plan") != plan_key or args.force:
        print("[plan] computing chunk boundaries ...", flush=True)
        t0 = time.time()
        chunks = plan_chunks(
            src, duration,
            args.chunk_minutes * 60, args.overlap_seconds, args.min_chunk_minutes * 60,
            args.cut_window_seconds, args.cut_noise_db,
        )
        print(f"[plan] {len(chunks)} chunks in {time.time() - t0:.1f}s", flush=True)
        state = {"plan": plan_key, "chunks": chunks, "runs": [], "model": args.model}
        write_json_atomic(state_file, state)
    else:
        chunks = state["chunks"]
        print(f"[plan] resumed existing plan: {len(chunks)} chunks", flush=True)

    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    if args.merge_only:
        done = []
        for chunk in chunks:
            data = load_done_chunk(chunk_path(args.workdir, chunk["index"], "json"), chunk)
            if data is None:
                print(f"error: missing chunk {chunk['index']}; run transcription first", file=sys.stderr)
                return 2
            done.append(data)
        segments = merge_chunks(done, args.overlap_seconds)
        stitch_broken(segments)
        segments = [s for s in segments if s["text"].strip()]
        if not args.no_polish:
            segments = polish_segments(segments)
        segments = fix_overlaps(segments)
        paths = write_outputs(segments, args.outdir, args.stem, args.subtitle_max_chars)
        print(f"[done] {quality_report(segments, duration)}", flush=True)
        for ext, path in paths.items():
            print(f"  {ext:5s} {path}  ({path.stat().st_size / 1024:.0f} KB)", flush=True)
        return 0

    model = load_model(args.model, args.device, args.compute_type)
    pipe = model if args.no_batch else BatchedInferencePipeline(model=model)

    done: list[dict[str, Any]] = []
    session_start = time.time()
    audio_done = 0.0
    for pos, chunk in enumerate(chunks):
        jpath = chunk_path(args.workdir, chunk["index"], "json")
        cached = None if args.force else load_done_chunk(jpath, chunk)
        if cached is not None:
            done.append(cached)
            audio_done += chunk["duration"]
            print(f"[{pos + 1}/{len(chunks)}] chunk {chunk['index']:04d} cached "
                  f"({fmt_clock(chunk['start'])} -> {fmt_clock(chunk['end'])})", flush=True)
            continue

        wav = chunk_path(args.workdir, chunk["index"], "wav")
        extract_wav(src, wav, chunk["start"], chunk["end"])
        segments, elapsed = transcribe_wav(
            pipe, wav, args.language, prompt, args.beam_size, args.batch_size, not args.no_vad
        )
        for seg in segments:
            seg["start"] = round(seg["start"] + chunk["start"], 3)
            seg["end"] = round(seg["end"] + chunk["start"], 3)

        payload = {
            "index": chunk["index"],
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": chunk["duration"],
            "model": args.model,
            "elapsed": round(elapsed, 2),
            "realtime_factor": round(chunk["duration"] / elapsed, 2) if elapsed else None,
            "segments": segments,
            "text": " ".join(s["text"] for s in segments),
        }
        write_json_atomic(jpath, payload)
        done.append(payload)
        audio_done += chunk["duration"]

        run_info = {
            "index": chunk["index"],
            "elapsed": round(elapsed, 2),
            "duration": chunk["duration"],
            "rtf": payload["realtime_factor"],
            "model": args.model,
            "batch_size": args.batch_size,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        state.setdefault("runs", []).append(run_info)
        write_json_atomic(state_file, state)

        session_elapsed = time.time() - session_start
        remaining = duration - audio_done
        speed = audio_done / session_elapsed if session_elapsed > 0 else 0
        eta = remaining / speed if speed > 0 else 0
        print(
            f"[{pos + 1}/{len(chunks)}] chunk {chunk['index']:04d} "
            f"{fmt_clock(chunk['start'])} -> {fmt_clock(chunk['end'])}  "
            f"{elapsed / 60:.1f} min  {payload['realtime_factor']}x  "
            f"segs={len(segments)}  eta={fmt_clock(eta)}",
            flush=True,
        )
        if not args.keep_wav:
            wav.unlink(missing_ok=True)

    print("[merge] combining chunks ...", flush=True)
    segments = merge_chunks(done, args.overlap_seconds)
    stitch_broken(segments)
    segments = [s for s in segments if s["text"].strip()]
    if not args.no_polish:
        segments = polish_segments(segments)
    segments = fix_overlaps(segments)

    paths = write_outputs(segments, args.outdir, args.stem, args.subtitle_max_chars)
    print(f"[done] {quality_report(segments, duration)}", flush=True)
    for ext, path in paths.items():
        print(f"  {ext:5s} {path}  ({path.stat().st_size / 1024:.0f} KB)", flush=True)

    state["last_run"] = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - session_start, 1),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    write_json_atomic(state_file, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())

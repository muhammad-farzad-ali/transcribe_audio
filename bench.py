#!/usr/bin/env python3
"""Benchmark candidate Whisper models on slices of a real recording.

Runs each model over the same short windows, measures realtime factor and
VRAM, and writes bench/REPORT.md + bench/results.json. `large-v3` output is
used as a pseudo-reference for a relative WER comparison.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

from transcribe import (
    BatchedInferencePipeline,
    WhisperModel,
    clean_text,
    extract_wav,
    load_wav_mono16k,
    probe_duration,
    transcribe_wav,
)

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
AUDIO = ROOT / "data_engineering_complete.m4a"

CANDIDATES = [
    # (model, batch_size)
    ("large-v3-turbo", 8),
    ("distil-large-v3.5", 8),
    ("large-v3", 4),
]
SLICES = [(600, 300), (33000, 300), (63000, 300)]
GLOSSARY = (
    "Data engineering course. Terms: data lake, data warehouse, database, RDBMS, SQL, "
    "ETL, ELT, pipeline, ingestion, batch, streaming, Kafka, Spark, Hadoop, Airflow, "
    "dbt, Snowflake, BigQuery, Redshift, Databricks, Parquet, Avro, ORC, partitioning, "
    "OLAP, OLTP, cloud, AWS, GCP, Azure, scheduling, orchestration, data model, schema."
)


class VramMonitor(threading.Thread):
    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=5)
        return self.peak


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)] / len(r)


def main() -> int:
    BENCH.mkdir(exist_ok=True)
    (BENCH / "slices").mkdir(exist_ok=True)
    duration = probe_duration(AUDIO)
    print(f"[bench] audio {duration / 3600:.2f} h, slices={SLICES}", flush=True)

    wavs: list[Path] = []
    for i, (start, length) in enumerate(SLICES):
        wav = BENCH / "slices" / f"slice_{i}.wav"
        if not wav.exists():
            print(f"[bench] extracting slice {i}: {start}s +{length}s", flush=True)
            extract_wav(AUDIO, wav, start, start + length)
        wavs.append(wav)

    results: dict[str, dict] = {}
    for model_name, batch in CANDIDATES:
        print(f"\n[bench] === {model_name} (batch={batch}) ===", flush=True)
        t0 = time.time()
        model = WhisperModel(model_name, device="cuda", compute_type="int8_float16", num_workers=1)
        load_s = time.time() - t0
        pipe = BatchedInferencePipeline(model=model)
        print(f"[bench] loaded in {load_s:.1f}s", flush=True)

        mon = VramMonitor()
        mon.start()
        per_slice = []
        for i, wav in enumerate(wavs):
            segs, elapsed = transcribe_wav(pipe, wav, "en", GLOSSARY, 5, batch, True)
            text = " ".join(s["text"] for s in segs)
            per_slice.append(
                {
                    "slice": i,
                    "offset": SLICES[i][0],
                    "audio_seconds": SLICES[i][1],
                    "elapsed": round(elapsed, 2),
                    "rtf": round(SLICES[i][1] / elapsed, 2),
                    "seconds_per_audio_hour": round(elapsed / (SLICES[i][1] / 3600), 1),
                    "words": len(text.split()),
                    "text": text,
                }
            )
            print(
                f"[bench] slice {i}: {elapsed:.1f}s for {SLICES[i][1]}s audio "
                f"({per_slice[-1]['rtf']}x) {per_slice[-1]['words']} words",
                flush=True,
            )
        peak = mon.stop()
        rtfs = [s["rtf"] for s in per_slice]
        results[model_name] = {
            "batch_size": batch,
            "load_seconds": round(load_s, 1),
            "peak_vram_mib": peak,
            "rtf_median": round(statistics.median(rtfs), 2),
            "rtf_min": round(min(rtfs), 2),
            "projected_hours": round(duration / statistics.median(rtfs) / 3600, 2),
            "slices": per_slice,
        }
        print(
            f"[bench] {model_name}: median {results[model_name]['rtf_median']}x -> "
            f"{results[model_name]['projected_hours']} h for the full file, peak VRAM {peak} MiB",
            flush=True,
        )
        del pipe, model
        import gc

        gc.collect()
        time.sleep(2)

    ref = results.get("large-v3")
    if ref:
        for name, res in results.items():
            if name == "large-v3":
                res["wer_vs_large_v3"] = 0.0
                continue
            wers = [
                wer(clean_text(a["text"]), clean_text(b["text"]))
                for a, b in zip(ref["slices"], res["slices"])
            ]
            res["wer_vs_large_v3"] = round(statistics.mean(wers) * 100, 2)

    (BENCH / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# Model benchmark — data_engineering_complete.m4a",
        "",
        f"Audio: {duration / 3600:.2f} h · GPU: RTX 4050 Laptop 6 GB · compute `int8_float16` · "
        f"language `en` · 3 × 5 min slices",
        "",
        "| Model | Batch | Load | Speed (median) | VRAM peak | Full 18.6 h | WER vs large-v3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["projected_hours"]):
        lines.append(
            f"| `{name}` | {r['batch_size']} | {r['load_seconds']}s | {r['rtf_median']}x | "
            f"{r['peak_vram_mib']} MiB | **{r['projected_hours']} h** | "
            f"{r.get('wer_vs_large_v3', '-')}% |"
        )
    lines += ["", "WER is measured against `large-v3` output as a pseudo-reference (not human truth).", ""]
    for name, r in results.items():
        lines.append(f"## {name}")
        lines.append("")
        for s in r["slices"]:
            lines.append(f"**slice {s['slice']}** ({s['offset']}s, {s['rtf']}x): {s['text'][:700]}")
            lines.append("")
    (BENCH / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n[bench] wrote bench/REPORT.md and bench/results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

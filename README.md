# transcribe_audio

Local, resumable transcription of long recordings with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Built for
multi-hour lectures on a single consumer GPU (tested on an RTX 4050 Laptop,
6 GB VRAM).

## Why it is chunked

An 18-hour file in one pass is fragile: a crash loses everything, and Whisper
degrades on very long inputs. This pipeline cuts the audio into ~20 minute
chunks at the quietest point near each boundary, writes one JSON per chunk
(`work/chunks/chunk_XXXX.json`), and merges them back into a single timeline.
Re-running the same command resumes where it stopped.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e .
```

Python 3.12 is used because CUDA wheels for 3.14 are not available yet.
cuBLAS/cuDNN come from pip; `transcribe.py` puts them on `LD_LIBRARY_PATH`
itself (it re-execs once), so no system CUDA toolkit is required.

## Usage

```bash
.venv/bin/python transcribe.py recording.m4a --model large-v3-turbo
```

Outputs land in `transcript/`: `transcript.txt`, `.srt`, `.vtt`, `.json`,
`.md` (timestamped, readable). Working data lives in `work/`.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `large-v3-turbo` | any faster-whisper model (`large-v3`, `distil-large-v3.5`, ...) |
| `--language` | `en` | source language |
| `--chunk-minutes` | `20` | target chunk length |
| `--overlap-seconds` | `10` | overlap between chunks, de-duplicated at merge |
| `--batch-size` | `8` | batched inference; lowered automatically on CUDA OOM |
| `--prompt` / `--prompt-file` | – | glossary / initial prompt to bias spelling |
| `--merge-only` | off | rebuild outputs from cached chunks without re-transcribing |
| `--max-chunks` | 0 (all) | transcribe only the first N chunks (smoke test) |
| `--force` | off | ignore cached chunks and start over |

For an unattended run:

```bash
systemd-inhibit --what=sleep:idle --why="transcribing" \
  .venv/bin/python transcribe.py recording.m4a --model large-v3-turbo
```

## Benchmark

Measured on this machine (RTX 4050 Laptop 6 GB, `int8_float16`, 3 × 5 min slices
of `data_engineering_complete.m4a`):

| Model | Batch | Speed | VRAM peak | Projected full file | WER vs large-v3 |
|---|---|---|---|---|---|
| `distil-large-v3.5` | 8 | 74x | 2861 MiB | 0.25 h | 5.36% |
| `large-v3-turbo` | 8 | 68x | 3030 MiB | 0.27 h | 2.99% |
| `large-v3` | 4 | 26x | 4776 MiB | 0.70 h | reference |

`large-v3-turbo` was chosen: near-reference accuracy at ~2.6× the speed.
The full 18 h 35 m file ran in **~14 minutes** (76–80× realtime sustained,
~100% GPU utilisation, no throttling), producing 184,772 words.

## Output of this recording

`transcript/data_engineering_complete.{txt,srt,vtt,json,md}` — 18:35:12,
2,460 segments, 184,772 words (166 wpm), 14,117 subtitle cues (≤84 chars each).
`qa.py` reports no timeline overlaps, no duplicates and full coverage:

```bash
.venv/bin/python qa.py transcript/data_engineering_complete.json
```

## Notes

- `.m4a` is decoded by `ffmpeg`; any audio/video format ffmpeg understands works.
- Chunk WAVs are 16 kHz mono PCM and deleted after each chunk unless `--keep-wav`.
- Overlapping audio is de-duplicated at merge time with a token-similarity
  match; the new chunk owns the overlap, and the previous chunk's truncated
  tail is trimmed rather than duplicated.
- `condition_on_previous_text=False` and `temperature=[0, 0.2, 0.4]` are used
  to suppress the repetition loops Whisper falls into on long inputs.
- Avoid `--prompt` for chunked runs unless you verify it: on this recording an
  `initial_prompt` glossary made the model silently skip the first seconds of
  several chunks (audio starts mid-sentence). The skip is model-independent and
  content-dependent, not length-dependent. The transcript here was produced
  without a prompt.
- Punctuation is Whisper's own (with sentence capitals and canonical tech nouns
  normalised by the polish pass); occasional run-on sentences at chunk starts
  are expected because each chunk begins mid-sentence.


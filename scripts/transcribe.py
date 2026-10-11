"""Transcribe all train+test audio files with faster-whisper (small.en, int8 CPU).

Writes incremental progress to data/transcripts.csv so the job can be
monitored and resumed if interrupted. Also records word-level timing stats
(total words, total pause time, longest pause) for prosodic features later.
"""
import csv
import json
import os
import time

from faster_whisper import WhisperModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Dataset_Final")
OUT_CSV = os.path.join(ROOT, "data", "transcripts.csv")

FIELDS = [
    "split", "filename", "duration_sec", "text",
    "n_words", "n_segments", "total_pause_sec", "max_pause_sec",
    "avg_word_conf", "lang_prob",
]


def iter_files():
    for split in ("train", "test"):
        d = os.path.join(DATA, split)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".wav"):
                yield split, os.path.join(d, fn), fn


def already_done():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((row["split"], row["filename"]))
    return done


def main():
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    done = already_done()
    write_header = not os.path.exists(OUT_CSV)

    files = list(iter_files())
    total = len(files)
    print(f"Total files: {total}, already done: {len(done)}", flush=True)

    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        processed = 0
        t_start = time.time()
        for split, path, fn in files:
            if (split, fn) in done:
                continue
            t0 = time.time()
            segments, info = model.transcribe(
                path, beam_size=5, word_timestamps=True, vad_filter=True,
            )
            segments = list(segments)
            words = []
            for seg in segments:
                if seg.words:
                    words.extend(seg.words)

            text = " ".join(s.text.strip() for s in segments).strip()
            n_words = len(words)
            pauses = []
            for i in range(1, len(words)):
                gap = words[i].start - words[i - 1].end
                if gap > 0:
                    pauses.append(gap)
            total_pause = sum(pauses)
            max_pause = max(pauses) if pauses else 0.0
            avg_conf = (
                sum(w.probability for w in words) / len(words) if words else 0.0
            )

            # duration via soundfile-free approach: use info.duration from whisper
            duration = info.duration

            row = {
                "split": split,
                "filename": fn,
                "duration_sec": round(duration, 3),
                "text": text,
                "n_words": n_words,
                "n_segments": len(segments),
                "total_pause_sec": round(total_pause, 3),
                "max_pause_sec": round(max_pause, 3),
                "avg_word_conf": round(avg_conf, 4),
                "lang_prob": round(info.language_probability, 4),
            }
            writer.writerow(row)
            f.flush()
            processed += 1
            elapsed = time.time() - t_start
            rate = elapsed / processed
            remaining = total - len(done) - processed
            eta_min = rate * remaining / 60
            print(
                f"[{len(done)+processed}/{total}] {split}/{fn} "
                f"({time.time()-t0:.1f}s) ETA {eta_min:.1f} min",
                flush=True,
            )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

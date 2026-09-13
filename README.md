# FootfallCam AI Engineer Assessment — Staff Identification & Localization

A computer-vision pipeline that watches an overhead CCTV video, figures out which
person is the tagged staff member (the one wearing a name tag), and reports which
frames they appear in — plus, as a bonus, their (x, y) location in each of those
frames.

Built for the FootfallCam AI Engineer take-home assessment: identify staff in
`sample.mp4`, and be ready to run the same pipeline live, unmodified, on a new,
unseen test video.

## How it works, in one sentence

Detect and track every person in the video (YOLOv8-seg + ByteTrack), match each
tracked person's clothing color against one reference photo picked by a human at
the start (CIE-Lab color distance), keep only the tracks that are both
color-matched and genuinely walking, automatically resolve conflicts and stitch
together broken tracks, and — for anything the pipeline can't confidently decide
on its own (like an outfit change) — ask a human to confirm with a single
keypress rather than guess.

```
INPUT (video)
   │
   ▼
Pick one reference photo of the staff member  (one-time, interactive)
   │
   ▼
Detect + track every person → score clothing color → keep walking + color matches
   │
   ▼
Resolve conflicts, bridge broken tracks, flag anything ambiguous for a human check
   │
   ▼
OUTPUT: per-frame (x, y) table + an annotated video + an audit trail of anything flagged
```

See `pipeline_diagram.png` for the full step-by-step diagram.

## Repo contents

| File / folder | What it is |
|---|---|
| `src/staff_id.py` | The pipeline itself — the only script you run |
| `sample.mp4` | The video provided for this assessment |
| `yolov8n-seg.pt` | Pretrained detection weights (bundled so it also works offline) |
| `requirements.txt` | Python dependencies |
| `DOCUMENTATION_FINAL.md` | **Start here** — the short write-up: assumptions, methodology, results, and how to use it (a `.docx`/PDF copy can be generated from this on request) |
| `DOCUMENTATION_FULL.md` | The unabridged version of the same write-up, with every assumption/challenge/limitation kept in, for anyone who wants the full detail |
| `pipeline_diagram.png` | Standalone copy of the pipeline diagram |
| `KNOWLEDGE.md`, `LOG.md`, `SOLUTION_PLAN.md` | Internal engineering notes — design reasoning, a chronological build log, and the original planning doc. Not required reading; kept for anyone curious how the pipeline reached its current form |
| `AI Evaluation Test.pdf` | The original task brief |

## Setup

Requires Python 3.12 and a webcam-free desktop environment (the interactive steps
below pop up a small preview window).

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # macOS/Linux
```

`yolov8n-seg.pt` is already included in this repo, so the first run doesn't need
an internet connection to fetch it.

## Quick start

For a first-time or live-demo run — no command-line flags needed:

1. Put the video you want to test into this project folder.
2. Open a terminal here and run:
   ```bash
   .venv\Scripts\python.exe src\staff_id.py
   ```
3. Pick the video from the list it shows you.
4. A window pops up: scrub to a frame where the staff member is clearly visible,
   drag a box around them, and press Enter. This is the only manual step — it
   tells the pipeline who to look for.
5. Wait while it processes the whole video automatically (a few minutes for a
   ~1 minute clip).
6. If it spots someone it isn't fully sure about (e.g. after an outfit change),
   a photo pops up asking *"Is this the staff member? (Y/N/Skip)"* — answer
   with one keypress.
7. Open the `output/` folder when it's done:
   - `annotated.mp4` — the video with the staff member boxed in green, for a
     quick visual check
   - `staff_detections.csv` — the frame-by-frame (x, y) location table
   - `possible_staff_review.csv` + `auto_rejected_conflicts.csv` — only
     created if something needed review; an audit trail of what was flagged
     and how it was resolved

## Repeatable / scripted runs

For re-running against a known video without the interactive prompts (useful
for development, or once you already know the reference frame/box):

```bash
.venv\Scripts\python.exe src\staff_id.py --video sample.mp4 --output-dir output --ref-frame 402 --ref-box 476,488,106,173
```

Add `--skip-review` to skip the Y/N/S popups entirely (leaves anything
ambiguous unresolved in `possible_staff_review.csv` instead). Run
`python src/staff_id.py --help` for the full list of tunable thresholds.

## Read next

- **`DOCUMENTATION_FINAL.md`** — assumptions, methodology, evaluation results,
  and known limitations, condensed to the essentials.
- **`DOCUMENTATION_FULL.md`** — the same document, unabridged.

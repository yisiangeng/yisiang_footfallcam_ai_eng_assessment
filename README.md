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
| `pipeline_diagram.png` | Standalone copy of the pipeline diagram |
| `KNOWLEDGE.md` | Design reasoning: why the pipeline is built the way it is, and its known limitations. Not required reading, but the most useful of the internal notes if you only read one |
| `AI Evaluation Test.pdf` | The original task brief |
| `dev_notes/` | Personal working history (chronological build log, the original planning doc, the unabridged documentation draft, this document's own outline/brief, and archived debug images from an earlier discarded approach). Not required reading — kept for the author's own record, not for a reviewer |

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
7. Open the `output/` folder when it's done — see **Output folder** below for
   what's in it.

## Output folder

Everything lands in `--output-dir` (`output/` by default):

| File | What it is |
|---|---|
| `staff_detections.csv` | Frame-by-frame (x, y) location table — the main bonus-task output |
| `annotated.mp4` | Full video, staff boxed green, ambiguous events yellow, everyone else orange |
| `staff_highlight_clip.mp4` | Same as above, trimmed to just the windows where staff is present — for a quick demo |
| `staff_trajectory.png` | Plot of the staff member's (x, y) path over time |
| `reference_crop.jpg` | The reference photo actually used to match against |
| `possible_staff_review.csv` + `review_event_*.jpg` | Only created if something was flagged as ambiguous (e.g. an outfit change) — audit trail of what was reviewed and the outcome |
| `auto_rejected_conflicts.csv` | Only created if two tracks looked like staff at the same time — audit trail of which one was auto-rejected and why |

**`staff_detections.csv` columns:**
- `frame` / `timestamp_s` — which frame, and its time in seconds
- `staff_present` — True/False, was staff detected in this frame
- `track_id` — the tracker's ID for that detection (blank if `interpolated`)
- `x`, `y` — the person's on-screen position (footpoint)
- `match_score` — how closely their clothing matched the reference (blank if `interpolated`)
- `interpolated` — True if this row is a gap-fill estimate (no real detection that frame), not an actual observation

**`possible_staff_review.csv` columns:**
- `event` — ID for one flagged occurrence (a group of nearby track fragments)
- `start_frame`/`end_frame`, `start_time_s`/`end_time_s` — when it happened
- `n_track_fragments` — how many separate tracker IDs were merged into this event
- `representative_crop` — filename of the photo shown during review
- `resolution` — `confirmed STAFF` / `confirmed not staff` / `needs manual review`
- `color_outlier` — True if flagged because part of the event's clothing color diverged from the rest (possibly a different person mid-event)
- `position_jump` — True if flagged because of an implausible position jump partway through (possibly a tracker ID-switch)

**`auto_rejected_conflicts.csv` columns:**
- `track_id` — the track that was auto-rejected
- `start_frame`/`end_frame`, `start_time_s`/`end_time_s` — when it happened
- `median_score` — its own color-match score
- `conflicts_with_track_id` / `conflicts_with_median_score` — the higher-scoring track it lost to, and that track's score

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
- **`dev_notes/DOCUMENTATION_FULL.md`** — the same document, unabridged, kept
  for personal reference rather than as part of the deliverable.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

Claude Code is acting as the AI Engineer on this task (the user's own role for this take-home).
Work in that capacity: own the technical design end-to-end, push back on plan details that
conflict with evidence already gathered (see `KNOWLEDGE.md` / `dev_notes/LOG.md`), not just
implement whatever is drafted.

## Project

FootfallCam AI Engineer take-home assessment: detect which frames of `sample.mp4` (a static,
ceiling-mounted, fisheye overhead people-counting camera) show the "staff" member — the one
person wearing a name tag — and, as a bonus, that person's (x, y) location per frame. See
`AI Evaluation Test.pdf` for the original brief and `KNOWLEDGE.md` for the full history of
what's been tried, why, and current known limitations — read `KNOWLEDGE.md` before making
further changes to the detection/matching logic, since several earlier approaches (stock
YOLO alone, whole-crop HSV histograms, IoU-based person verification) were tried and
discarded for documented reasons.

## Environment

Use the project's `.venv` (`.venv/Scripts/python.exe` on Windows) to run `staff_id.py`,
not the system Python. The system Python has both `opencv-python` and `opencv-python-headless`
installed (the latter pulled in by an unrelated package, `easyocr`), and they conflict in the
shared `cv2` namespace — GUI functions like `cv2.namedWindow` silently resolve to the headless
build's stub and raise `cv2.error: ... The function is not implemented` even though a
GUI-capable `opencv-python` is also installed. `.venv` has a clean, GUI-capable `opencv-python`
only, with no such conflict — confirmed working (`cv2.namedWindow` succeeds). See `dev_notes/LOG.md`,
"Round 2" for how this was diagnosed. Set up via:
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install opencv-python numpy scipy torch ultralytics transformers lap
```
(`lap` is `ultralytics`' Hungarian-matching dependency for ByteTrack. It auto-installs itself
on first `.track()` call if missing, but that install doesn't take effect until the *next*
process start — the run that triggers the auto-install silently gets no track IDs at all. Install
it up front to avoid a wasted multi-minute run; see `dev_notes/LOG.md`, "Round 2" for how this was found.)

## Commands

Current pipeline (round 2). Guided mode — asks which video and output folder interactively,
then walks through picking the staff reference on-screen step by step (this is the intended
live-demo flow for an unseen test video, see `dev_notes/SOLUTION_PLAN.md` 2.8). Recommended
live-demo invocation adds `--max-review-events` on top of the otherwise flag-less guided flow
(guided mode only replaces the `--video`/`--output-dir` prompts — every other flag, including
this one, still applies normally regardless):

```bash
.venv/Scripts/python.exe src/staff_id.py --max-review-events 6
```

Why 6: measured directly on a real run of `sample.mp4` (17 possible-staff events: 4 from the
original unmatched-color category, 2 of which were real staff; 13 from the newer `stationary`
category, none of which were) — see `dev_notes/UNEXPECTED.md`, risk #4 and #7. Since
`stationary` events are always appended after the original category, a cap around 6 shows the
historically higher-hit-rate events live (all 4 original ones, plus a little headroom) and
defers the rest to `possible_staff_review.csv` rather than spending live demo time on a
category that happened to have a 0% hit rate on this video. Treat 6 as a starting point, not a
hard rule — check the printed event count on the actual test video and raise it if there's
time to spare. Omit the flag entirely (plain
`python src/staff_id.py`) for the uncapped "real" behavior — e.g. if asked how thorough the
review step actually is, the full uncapped `possible_staff_review.csv` from a run without the
cap is the more convincing answer than the capped one.

Scripted/repeated runs (e.g. this headless dev environment, or automated re-runs on a known
video) skip the prompts by passing flags directly:

```bash
.venv/Scripts/python.exe src/staff_id.py --video sample.mp4 --output-dir output --ref-frame 402 --ref-box 476,488,106,173
```
(`--ref-frame`/`--ref-box` are the calibrated values for `sample.mp4` — re-pick for any other
video, either by dropping them to trigger the interactive picker, or by finding new coordinates.)

Key flags (see `python src/staff_id.py --help`):
- `--staff-threshold` (default 0.5): minimum median Lab-based match score for a track to
  count as staff. Tuned for `sample.mp4`; re-check the per-track diagnostic printout (median
  score, speed, range for every track) against a new video before trusting the default.
- `--min-walk-speed` / `--min-walk-range` (default 6px/frame, 40px): a color-qualifying track
  only counts as staff if it's also moving fast/far enough to be a genuine walking event, not
  a seated/stationary match — see "Motion-based track filter" below.
- `--use-clip`: blend in CLIP cosine similarity (15% weight). Off by default — measured
  anti-correlated with the true match on this footage (see `dev_notes/LOG.md`, "Round 2"); kept as an
  opt-in experiment, not because it's expected to help.
- `--skip-review`: don't pop up the Y/N/S confirmation window for flagged possible-staff
  events (see "Possible-staff flagging" below) — leaves them unresolved in
  `possible_staff_review.csv` instead. Needed for scripted/headless runs (no display to pop a
  window on); optional otherwise.
- `--max-review-events N`: cap how many possible-staff events are shown interactively (e.g. to
  keep a time-boxed live demo moving) — any beyond N are left as `needs manual review` in
  `possible_staff_review.csv` without a popup, same as `--skip-review` but only past the cap.
  Added as part of the demo-risk mitigations in `dev_notes/UNEXPECTED.md`.

Before any of the above runs, a short pre-flight check sequence (see `dev_notes/UNEXPECTED.md`
for the full reasoning): validates the video actually opens and reads (fails fast with a clear
message on a corrupt/unsupported file instead of failing deep inside an interactive step),
checks `cv2` GUI windows actually work on this machine if the run will need one (catches the
`opencv-python`/`opencv-python-headless` conflict above immediately rather than mid-demo), and
probes every output filename for a lock from a leftover open file before Pass 1 starts. Once
the reference is built, a quick multi-second spot-check scores it against a handful of sampled
frames and warns if the best match found is suspiciously low, before the full multi-minute
Pass 1 commits to a possibly-bad reference pick.

Outputs land in `--output-dir` (default `output/`):
- `staff_detections.csv` — `frame, timestamp_s, staff_present, track_id, x, y, match_score,
  interpolated` (row-per-detection; supports multiple simultaneous staff without a schema
  change). `interpolated=True` rows are gap-filled — no detection existed for that frame; `x,
  y` are linearly interpolated between the nearest real staff sightings, `track_id` and
  `match_score` are blank — see "Fragment bridging + gap interpolation" below.
- `annotated.mp4` — green boxes for identified staff tracks (including any confirmed via the
  review prompt, and bridged/interpolated ones — the latter drawn as a green dot + label since
  there's no detected box for them), yellow for still-unresolved possible-staff events, orange
  for everyone else
- `reference_crop.jpg` — the mask-restricted appearance template actually used for that run
- `possible_staff_review.csv` + `review_event_*.jpg` — only written if something was flagged
  (see "Possible-staff flagging" below); one row/crop per flagged event, with a `resolution`
  column (`confirmed STAFF` / `confirmed not staff` / `needs manual review`), a `color_outlier`
  column (`True` for a sub-event split out because its color diverged from the rest of a
  larger merged event), a `position_jump` column (`True` for a sub-event split out because
  of an implausible position jump from the previous fragment), and a `stationary` column
  (`True` for a track whose color matched well but wasn't detected walking, and so is flagged
  for human review instead of being silently dropped by the motion-based track filter — see
  `dev_notes/UNEXPECTED.md`) — see "Possible-staff flagging" below for the first two. Each
  `review_event_*.jpg` is a side-by-side pair of generously padded, highlighted context crops
  (start and end of the fragment, not a single tight torso crop) — see
  `padded_highlight_crop()`/`hstack_crops()`.
- `auto_rejected_conflicts.csv` — only written if a simultaneous-staff conflict was found (see
  "Simultaneous-staff conflict resolution" below); one row per auto-rejected track, with which
  higher-scoring track it conflicted with and both their median scores. These are *not* also
  put through the interactive review — a simultaneous higher-scoring track is already strong,
  objective evidence, unlike the subjective color/motion ambiguity the review popup is for.
- `staff_trajectory.png` — a scatter/line plot of every staff-present `(x, y)` point, colored
  by timestamp (`render_staff_trajectory()`). The connecting line deliberately breaks across
  any real absence gap longer than `TRAJECTORY_LINE_BREAK_SECONDS` (1.2s), so it never visually
  implies movement during a period the person wasn't actually detected. Skipped (not written)
  if the video has zero staff-present frames.
- `staff_highlight_clip.mp4` — the same frames as `annotated.mp4`, but only the windows where
  staff is actually present, each padded with `HIGHLIGHT_PAD_SECONDS` (0.5s) of context and
  merged if that padding makes neighboring windows overlap (`compute_highlight_windows()`) —
  a much shorter clip for a quick demo instead of scrubbing through the full-length video.
  Written during the same Pass 2 video read as `annotated.mp4` (no extra pass over the video).
  Skipped if there are no staff-present frames at all.

Dependencies: `opencv-python`, `numpy`, `torch`, `ultralytics` (YOLOv8-seg + ByteTrack),
`scipy` (Savitzky-Golay coordinate smoothing), `matplotlib` (`staff_trajectory.png`) required;
`transformers` (CLIP) only needed if `--use-clip` is passed.

Round 1 (an earlier motion-detection-based pipeline) was deleted once this pipeline fully
superseded it — see `dev_notes/LOG.md`, "Testing 1" for the complete history of what it did, why it was
built that way, and why it was replaced. Nothing there is needed to run or understand the
current pipeline; it's kept purely as narrative history.

## Architecture (`src/staff_id.py`)

Single YOLO pass (no separate rendering pass needed for detection — only raw frame reads):

1. **Reference:** interactive or CLI-specified crop, segmented with the same detector so the
   reference embedding is mask-restricted too (apples-to-apples with candidate crops).
2. **Detect + track + score, streamed frame-by-frame:**
   - `YOLOv8-seg` (`.track(tracker='bytetrack.yaml', persist=True)`) gives person boxes,
     instance masks, and stable track IDs (Kalman + Hungarian matching) in one call — replaces
     round 1's separate MOG2 detector + greedy centroid tracker.
   - Each box's crop has non-mask pixels replaced with a neutral background before scoring, so
     a loose box including desk/floor doesn't dilute the color read (round 1's dominant
     false-positive cause).
   - `mean_lab()` scores each crop against the reference by CIE-Lab distance (primary signal,
     0.85 weight if `--use-clip`, else the whole score) — CLIP cosine similarity is available
     but off by default (see "Commands").
3. **Track-level decision:** a track counts as staff only if (a) it has enough frames to not be
   flicker, (b) its median score clears `--staff-threshold`, and (c) it passes the
   **motion-based track filter (velocity + displacement thresholding)** — average centroid
   speed and positional range, computed from the track's own coordinates, high enough to be
   a genuine walking event rather than a
   seated/stationary match. This last check exists because YOLO (unlike round 1's motion-only
   detector) also matches seated people, and this office has more than one person in similarly
   light-colored clothing — color alone can't disambiguate two people at the same desk, but a
   person actually walking through the open corridor was the one context round 1 visually
   confirmed as unambiguous. `--min-walk-speed` (a raw px/frame threshold) is internally scaled
   by `REFERENCE_FPS / fps` so a different frame rate still requires the same real-world
   walking speed, not the same raw pixel count per frame — see `dev_notes/UNEXPECTED.md`. A
   track that clears the color threshold but fails this motion filter isn't silently dropped
   any more either — it's routed into the possible-staff review below as a `stationary` event
   (same file).
4. **Fragment bridging + gap interpolation (`bridge_track_fragments()`,
   `interpolate_staff_gaps()`):** measured directly on `sample.mp4` (see `dev_notes/LOG.md`, "Round 2"
   evaluation entry) — tracking fragmentation, not the appearance-matching ceiling, turned out
   to be the *main* cause of missed frames: ByteTrack keeps losing and re-acquiring a walking
   person, splitting one continuous walk into short track fragments, some too brief to
   individually clear `--min-track-seconds`/the motion-based track filter, some with zero detection
   at all for a few frames. Both functions apply the same "the person doesn't teleport" logic
   (`BRIDGE_MAX_GAP_SECONDS`/`BRIDGE_MAX_JUMP_PX`): `bridge_track_fragments()` pulls a
   short/marginal fragment into the staff result if it's within that gap+jump of an
   already-confirmed staff fragment (run to a fixed point, so a chain of several short pieces
   all get pulled in, not just the one touching a confirmed fragment directly); a lenient
   `BRIDGE_COLOR_FLOOR` guards against bridging in an unrelated person. `interpolate_staff_gaps()`
   then linearly interpolates `(x, y)` across any remaining true zero-detection gap between two
   staff sightings that are still close in time and space, writing those frames to the CSV with
   `interpolated=True` rather than leaving them as `staff_present=False`.
5. **Simultaneous-staff conflict resolution (`resolve_simultaneous_staff_conflicts()`):** the
   task brief states there's exactly one tagged staff member, so if two or more auto-qualified
   tracks are active at the *same* time, at most one of them can genuinely be staff — each was
   being judged independently against `--staff-threshold` with no awareness of the other. Found
   necessary on real footage: a sustained ~8s, 201-frame track (median score 0.55, just over
   threshold) and a genuine 18-frame staff track (median score 0.92) were both shown as STAFF
   simultaneously in the same frames — the office really does have more than one
   similarly-light-clothed person (see "Known limitations" below), and this particular one
   stayed close enough in color to clear the threshold for a long stretch. For each group of
   temporally-overlapping qualified tracks, keeps only the single highest-median-score one; the
   rest are demoted and **auto-rejected outright, logged to `auto_rejected_conflicts.csv`, not
   routed into the possible-staff review below.** Unlike a color/motion-ambiguous case, a
   demoted track already has strong, objective evidence against it (a higher-scoring track was
   simultaneously active) — sending it through the same subjective Y/N/S review as a genuine
   clothing-change case would flood the reviewer with settled questions for no benefit (found
   directly in testing: routing these into review produced 15 popups in one run, most of them
   not real ambiguity — see `dev_notes/LOG.md`, "Round 2").
6. **Possible-staff flagging + interactive confirmation (`find_possible_staff_events()`,
   `split_event_by_color_outliers()`, `confirm_event_interactive()`):** a walking track whose
   color *doesn't* match the reference could genuinely be someone else, or it could be the
   staff member after an outfit change (e.g. a jacket over the tagged shirt) — color can't tell
   those apart, and neither can this pipeline, honestly. Rather than guessing (full automation
   was deliberately rejected — see below), walking-but-unmatched tracks are grouped into
   "events" (fragments within `EVENT_MERGE_GAP_SECONDS` of each other — one real occurrence
   fragments into several track IDs, same as a confirmed staff track does), then any event
   whose color repeats elsewhere in the video (within `COLOR_REPEAT_LAB_DIST`) is dropped as
   more likely a regular non-staff person seen more than once. **Each surviving event is then
   split at any internal color discontinuity** (`split_event_by_color_outliers()`,
   `FRAGMENT_OUTLIER_LAB_DIST`) **and any internal position jump**
   (`split_event_by_position_jump()`, `MAX_BRIDGE_SPEED_PX_PER_FRAME`) — a merged event still
   only groups fragments by *time* proximity, so it can silently contain a tracker ID-switch
   onto a different person for part of its span (found on real footage, see below); a fragment
   whose own color, or position relative to its neighbor, diverges sharply becomes its own
   separate sub-event (`color_outlier`/`position_jump` flags) instead of riding along inside one
   blanket decision. The position check treats a genuine time *overlap* between two fragments
   specially: "implied speed" isn't meaningful for zero/negative elapsed time, so it instead
   checks whether they're far apart in space while coexisting (direct proof of two people) —
   an earlier version divided by a forced denominator of 1 in that case, which misfired on
   ordinary tracker handoffs of the *same* person and fragmented one continuous walk into six
   separate reviews in one real run (see `dev_notes/LOG.md`, "Round 2"). What's left gets shown to the
   user one at a time — a large popup (window explicitly resized to match its content; a live
   test found the default noticeably too small to read comfortably) with two generously padded,
   highlighted context crops (start and end of the fragment, the actual person outlined in
   gold; `padded_highlight_crop()`) and, when relevant, a visible "Flagged: ..." reason line —
   `Y`/`N`/`S` keypress or click, unless
   `--skip-review` is passed. `Y` promotes that (sub-)event's tracks straight into the confirmed
   staff results, `N` confirms it's not staff, `S` leaves it unresolved. All outcomes are
   written to `possible_staff_review.csv` regardless (including confirmed/rejected ones, for an
   audit trail). The wider crop + flagging reason were added after a live test: the original
   tight torso-only crop with no explanation led the user to press `Y` on a case the
   position-jump split had *already* correctly flagged, because the crop was too tight to judge
   identity and they fell back to clothing color — precisely the confounded signal in that case
   (see `dev_notes/LOG.md`, "Round 2").
   **Also flagged here (not silently dropped): color-matched tracks that fail the
   motion-based track filter** (`stationary` flag on their event) — one per track, no
   time-merging needed since there's no "does this color repeat" ambiguity to resolve first
   the way there is for unmatched color. Added so a genuinely seated/stationary staff member on
   an unseen video (this pipeline's biggest documented blind spot — see `KNOWLEDGE.md`, "Known
   limitations") at least reaches a human instead of vanishing with zero trace. See
   `dev_notes/UNEXPECTED.md`.
   **Why not fully automate this decision:** tested and rejected — on real data from this
   video, a flagged event has turned out to be a genuinely different person, so auto-promoting
   would silently mislabel someone. The cost of a wrong silent auto-label (an incorrect record
   in a system tracking who's staff) outweighs the cost of a human glancing at a photo for a
   few seconds. See `dev_notes/LOG.md`, "Round 2" for the full validation story on `sample.mp4`
   (correctly isolated a real clothing-change event with zero false positives among 60+ other
   tracked people in one test; correctly left a genuinely-different-person event for the user
   to reject in another; caught a real tracker ID-switch during brief physical contact between
   the staff and a second person mid-event), including a real mistuning bug caught and fixed
   during testing (`EVENT_MERGE_GAP_SECONDS` was initially too tight). Note the color-outlier
   split is itself limited by the same color-based reasoning as everything else here: it only
   catches an intruding fragment whose color meaningfully diverges from the majority, and
   measurably missed one real case where the intruding person happened to be wearing similarly
   dark clothing to the staff's jacket — see `KNOWLEDGE.md`, "Known limitations."
7. **Coordinates:** footpoint (bbox bottom-center) per frame, Savitzky-Golay smoothed per
   staff track (bridged/interpolated frames are not re-smoothed — the interpolation is already
   linear, i.e. already smooth by construction).
8. **Render pass:** re-reads the video (no re-detection) to draw boxes and write
   `annotated.mp4`, using the cached per-frame detections from step 2. Also writes
   `staff_highlight_clip.mp4` in the same pass (frames within a staff-present window get
   written to both files, no second read of the video) and, separately (matplotlib, no video
   read needed), `staff_trajectory.png` from the coordinates already gathered while writing
   `staff_detections.csv`.

No test suite or build step — this is a single evaluation script, run directly.

## Project layout

- `src/staff_id.py`, `output/` — the pipeline and its outputs. (Renamed from `staff_id_v2.py`
  once the original round-1 `staff_id.py` was deleted — no more "v1 vs v2" to disambiguate.)
- `yolov8n-seg.pt` — YOLOv8-seg weights used by the pipeline (auto-downloaded by `ultralytics`
  on first run if missing; present in the repo root so it also works offline).
- `AI Evaluation Test.pdf`, `sample.mp4` — original task brief and input video, unmodified.
- `KNOWLEDGE.md` — current-state reference: task requirements, why the pipeline is built the
  way it is, known limitations. Rewritten as understanding changes, not a chronological log.
- `DOCUMENTATION_FINAL.md` / `.docx` — the actual 1-2 page deliverable write-up (condensed:
  only the most impactful assumptions/challenges/limitations are kept). This is what to hand
  to a reviewer.
- `pipeline_diagram.png` — the pipeline diagram embedded in the documentation, generated by a
  one-off matplotlib script not checked into the repo (a documentation tool, not part of the
  pipeline itself).
- `README.md` — the repo's front door: project intro, setup, and usage instructions for
  someone landing on the repo without prior context.
- `dev_notes/` — everything below is personal working history, not required interviewer
  reading (moved out of the root to keep it uncluttered for a reviewer); nothing in it is the
  deliverable, but nothing in it should be treated as disposable either:
  - `dev_notes/LOG.md` — append-only chronological log of work done, in order, including the full
    "Testing 1" history (superseded `TESTING1_NOTES.md`, which no longer exists). Add new
    entries here as work happens rather than starting another notes file.
  - `dev_notes/SOLUTION_PLAN.md` — the active design doc for the current (round 2) approach.
  - `dev_notes/DOCUMENTATION_FULL.md` — the unabridged version of the deliverable write-up,
    with every assumption/challenge/limitation kept in; personal-reference only, not the
    deliverable.
  - `dev_notes/DOCUMENTATION_INSTRUCTION.md` — the outline/brief `DOCUMENTATION_FINAL.md` was
    written against, plus a log of deliberate deviations from it.
  - `dev_notes/UNEXPECTED.md` — live-demo risk assessment: what could go wrong with an unseen
    test video during the actual interview (long video, different resolution/fps, non-walking
    staff, GUI/environment issues, etc.), which of those were judged safe to fix automatically
    vs. requiring a before/after accuracy check first, and the implementation status of each.
  - `debug_archive/round1_crops/` — the stray debug crop images this folder held were later
    deleted (the folder itself remains, empty); `dev_notes/LOG.md`'s "Testing 1" entry still
    references them by name as narrative history, but the actual image files no longer exist
    in the repo.

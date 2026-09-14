# UNEXPECTED SITUATIONS 

Live-demo risk assessment: what could realistically go wrong with an *unseen* test video
during the actual interview, which of those risks were judged safe to fix automatically,
which need a before/after accuracy check before being trusted, and which aren't fixable in
code at all. Written in response to a direct request to think through demo-day contingencies
ahead of time, rather than after something breaks live.

Companion to `KNOWLEDGE.md` (design reasoning / known limitations of the *algorithm*) and
`dev_notes/LOG.md` (chronological history) — this file is specifically about *operational*
risk: things that could go wrong not because the matching logic is imperfect (that's already
covered elsewhere), but because the demo video, demo machine, or demo time budget differs from
what `sample.mp4` and this dev environment were tuned/tested against.

## How risks were triaged

Every risk below was mapped to one of three buckets:

- **Fixed (safe)** — the fix is a diagnostic, a workflow addition, or an equivalent-by-default
  change that provably doesn't alter behavior on `sample.mp4` itself. Implemented.
- **Considered, deferred** — the fix would change the actual detection/appearance-matching
  signal (not just surface a warning or add a review step). That could genuinely improve
  robustness to an unseen video, or it could quietly regress the precision/recall numbers
  already reported in `DOCUMENTATION_FINAL.md`. Not implemented without a real before/after
  comparison against the existing evaluation (the `12s-23s`/`31s-50s` ground-truth windows in
  `KNOWLEDGE.md`, "Evaluation") — that comparison hasn't been run, so these stay unimplemented
  for now.
- **Not fixable in code** — a genuine limitation to be honest about if it comes up live, not
  something a patch resolves in the time available.

Design principle for everything in the "Fixed" bucket: **detect and warn/ask, don't silently
auto-switch behavior.** If the pipeline quietly changed its own thresholds mid-run based on an
inferred condition, "why did it do that?" would have no good answer during a live technical
interview. Every fix below either (a) fails fast with a clear message, (b) prints a warning
and keeps going with a documented fallback, or (c) surfaces something for a human decision
instead of deciding silently.

## Risks and mitigations

### Fixed

| # | Risk | Mitigation | Where |
|---|---|---|---|
| 1 | GUI crashes mid-demo (the known `opencv-python`/`opencv-python-headless` conflict, see `CLAUDE.md` "Environment") | `check_gui_available()` opens+closes a dummy `cv2` window before any real work starts (only when the run will actually need one), and fails immediately with the exact fix instead of a cryptic `cv2.error` deep inside an interactive step | `src/staff_id.py`, `run()` pre-flight section |
| 2 | Bad/corrupt/unsupported video file or codec | `validate_video()` confirms the file opens *and* its first frame actually reads before anything else happens, and prints resolution/fps/frame count immediately so a wrong file is obvious right away | `src/staff_id.py`, `validate_video()` |
| 3 | A previous run's output file still open elsewhere (Excel, a photo/video viewer) locks the write | `check_outputs_writable()` probes every output filename (non-destructively, append-mode) before Pass 1 starts, using the same "close it and press Enter" retry prompt that was already in place for the final write — just moved earlier so it's caught in seconds, not after minutes of processing | `src/staff_id.py`, `check_outputs_writable()` / `_probe_writable()` |
| 4 | Too many possible-staff review popups eat the demo's time budget | `--max-review-events N` caps how many are shown live; anything beyond the cap is left as `needs manual review` in `possible_staff_review.csv`, same outcome as `--skip-review` but only past the cap. **Recommended live-demo value: 5** — see "Real interactive run" below for the data this is based on | `src/staff_id.py`, `parse_args()` / the review loop in `run()` |
| 5 | A different frame rate than `sample.mp4`'s 25fps makes `--min-walk-speed` (a raw px/frame threshold) too strict or too loose | `--min-walk-speed` is scaled internally by `REFERENCE_FPS / fps` so the same *real-world* walking speed is required regardless of frame rate, not the same raw pixel count per frame. **Verified as an exact no-op on `sample.mp4`** — `sample.mp4` is exactly 25.0fps (confirmed via `cap.get(cv2.CAP_PROP_FPS)`), matching `REFERENCE_FPS` exactly, so `effective_min_walk_speed == args.min_walk_speed` on this video with zero rounding error | `src/staff_id.py`, `REFERENCE_FPS` constant + `run()` |
| 6 | Many tracks scored close to `--staff-threshold` (color isn't discriminating cleanly for this video, more review events than usual) | After the score-spread printout, warn if too many eligible tracks (>15%) have a median score within `CLUSTER_BAND` (0.1) of the threshold, so this is visible before it becomes a surprise mid-review | `src/staff_id.py`, `run()`, right after "Color-match score spread" |
| 7 | Staff member rarely or never walks (motion-based track filter would previously silently drop them with zero trace) | Color-matched tracks that fail the walking gate are no longer dropped — they're collected (`stationary_candidates`) and routed into the same human-review flow as unmatched-color events, with a `stationary` flag/reason so a human can catch a genuinely seated/stationary staff member instead of the pipeline reporting zero staff presence with no explanation. This can only *add* review candidates, never auto-promote one to STAFF on its own — precision is still human-gated, not automatic | `src/staff_id.py`, `run()` (track-level decision loop + possible-staff-flagging section) |
| 8 | A bad reference pick (wrong frame/box) wastes the first ~2-3 minutes of a live demo before it's obvious something's wrong | `preview_reference_score()` spot-checks the reference against `PREVIEW_SAMPLE_COUNT` (6) frames spread across the video in a few seconds, right after the reference is built and before Pass 1 starts. If the best score found is below `PREVIEW_LOW_SCORE_WARNING` (0.35), warns and (when not `--skip-review`) asks whether to continue anyway | `src/staff_id.py`, `preview_reference_score()` |

### Considered, deferred (would change matching/detection behavior — needs a before/after accuracy check first)

| # | Risk | Would-be mitigation | Why deferred |
|---|---|---|---|
| 9 | Different camera resolution/distance than `sample.mp4` — every pixel-based threshold (`--min-walk-range`, `BRIDGE_MAX_JUMP_PX`, `MAX_BRIDGE_SPEED_PX_PER_FRAME`, the Lab-distance constants) was tuned on `sample.mp4`'s specific 960x720 geometry | Make motion thresholds scale-invariant — measure relative to the person's own detected box size ("body-heights per second") instead of raw pixels | The box-height denominator itself jitters frame-to-frame from ordinary segmentation noise/pose changes, which could introduce new noise into a currently-clean signal. Needs a full re-run against the existing ground-truth windows to confirm no regression on `sample.mp4` before being trusted |
| 10 | Unusual lighting (dim, overexposed, strong color cast) directly corrupts the CIE-Lab color-matching signal | Match on Lab *a/b* (chromaticity) only, dropping *L* (lightness) — more robust to brightness/exposure differences | Dropping a channel could also *reduce* discrimination on `sample.mp4` itself if brightness happens to carry real separating signal there. Same as above: needs a measured before/after, not implemented on judgment alone |

### Not fixable in code — operational awareness only

- **Panning/zooming camera.** The motion-based track filter computes speed/range in raw pixel coordinates assuming a static camera — a real architectural assumption (see `KNOWLEDGE.md`, "Known limitations"), not a quick patch. Best handled by being honest about it if it comes up, not by attempting a fix under time pressure.
- **Much longer video than `sample.mp4`.** Pass 1 runs at roughly 2.9s of processing per second of footage on CPU (measured: 53.6s of footage took 2m34s). A 5-10 minute test video could mean 15-30+ minutes for Pass 1 alone. Best mitigated by knowing the video length ahead of time and deciding in advance whether to demo on a trimmed clip, not by a code change under time pressure. (`--max-review-events` above helps with the *review* portion of demo time, not Pass 1 itself.)
- **GPU availability on the interview machine.** The pipeline already auto-detects CUDA vs CPU (`args.device`); worth confirming the actual demo machine's setup ahead of time rather than assuming it matches the dev `.venv`.

## Verification

All 8 "Fixed" items were implemented, then the full pipeline was re-run end-to-end on
`sample.mp4` (`--video sample.mp4 --output-dir output_verify --ref-frame 402 --ref-box
476,488,106,173 --skip-review`) and compared against the pre-fix baseline run. Results:

- **Ran cleanly**: exit code 0, no errors from any of the new pre-flight checks.
- **New pre-flight output confirmed working**: printed `Video: 960x720, 25.0fps, 1341 frames
  (53.6s)` immediately on start (validate_video()), and `Quick preview: scoring the reference
  against 6 sample frames... Best color-match score found in the quick scan: 0.83` right after
  building the reference (preview_reference_score()) — well above the 0.35 warning bar, so no
  false-alarm prompt on a known-good reference.
- **Core staff-detection result byte-for-byte identical to the pre-fix baseline**: `RESULT: 4
  of 69 tracked people identified as staff`, `Staff found in: 75 / 1341 frames (5.6%)`,
  `staff_detections.csv` (82 staff rows, 7 interpolated), highlight clip (166 frames, 6.6s),
  1 auto-rejected conflict (#89 vs #88) — every one of these numbers matches the pre-fix run
  exactly, confirming the fps-normalization is truly a no-op on this video (it's exactly
  25.0fps) and the walking-gate change never altered what gets auto-confirmed as staff.
- **New behavior fires correctly**: `possible_staff_review.csv` grew from 4 rows to 12 -- the
  original 4 unchanged, plus 8 new `stationary=True` rows (events 5-12) for tracks that the
  pre-fix baseline's per-track table showed as `not walking (seated?)` with a high median score
  (e.g. #77 at 0.84, #40 at 0.85, #143 at 0.50) and silently excluded with no trace. Each new
  row carries the correct reason text and the CSV's new `stationary` column is populated
  correctly (verified by reading the file directly, not just the console log).

`output_verify/` was deleted after verification -- it was a throwaway comparison run, not a
deliverable output.

## Real interactive run — how many review events is "too many," and the resulting recommendation

The `--skip-review` verification above confirmed the *logic* works, but doesn't test the actual
live-review experience (nothing gets shown when `--skip-review` is set). A real interactive run
was done separately (full guided flow, popups answered by hand): 17 possible-staff events
total, all genuinely reviewed. Breakdown from the resulting `possible_staff_review.csv`:

| Category | Count | Confirmed STAFF | Hit rate |
|---|---|---|---|
| Original (unmatched color) | 4 | 2 | 50% |
| `stationary` (new, item #7) | 13 | 0 | 0% |

Two takeaways:

1. **17 live popups is genuinely too many for a demo** — real, direct feedback: "the output was
   good and accurate, but that's slightly too much."
2. **The volume is entirely explained by the new `stationary` category, and it had a 0% hit
   rate on this video.** That's expected, not a flaw in the check: `stationary` exists purely as
   a safety net for a video where the real staff member happens to be seated part of the time
   (see item #7) — `sample.mp4`'s staff member never was, so this run paid the check's full
   review cost with none of its benefit. On a different unseen video where that scenario does
   occur, the same 13 slots could include the one that matters.

Resolved by capping rather than weakening the check: `possible_events` always lists the
original category first, `stationary` events appended after (see `find_possible_staff_events()`
call site in `run()`), so `--max-review-events 5` naturally shows the historically
higher-hit-rate category live and defers the `stationary` batch to the CSV — still fully
recorded, not dropped, just not spent live-demo time on a category that had zero payoff on the
one video this was actually measured on. Documented as the recommended live-demo invocation in
`CLAUDE.md`, "Commands" and `README.md`, "Quick start". The uncapped run remains the way to show
full thoroughness if asked, via the CSV rather than live popups.

A cleaner follow-up considered but not implemented: sort events by a priority score (e.g.
descending median color score for `stationary` events) before applying the cap, instead of
relying on category-append-order as a proxy for priority. Would make the cap correct even if a
future video's original category has more than ~5 genuine events. Not done yet -- flagged here
as a possible next step, not a promise.

# KNOWLEDGE.md

Working notes for this project: task requirements, assumptions, and the reasoning behind
the current approach. CLAUDE.md is the short operational summary; this file is the longer
record of *why* things are built the way they are, kept up to date as the solution evolves.
For the full chronological history (including an earlier, now-deleted pipeline this one
replaced), see `LOG.md` — this file only describes the current state, not how it got here.

## Task (from `AI Evaluation Test.pdf`)

FootfallCam AI Engineer take-home. Given `sample.mp4` (view from a 3D overhead sensor):

1. Identify which frames contain the "staff" — the one person wearing a staff name tag.
2. **Bonus:** output that person's (x, y) location per frame.
3. Deliverables: a 1-2 page write-up, code/repo, and be ready to run the code live on a
   *different* test video during the interview.

### What "staff" looks like

- The PDF's illustration photo (a green polo with a white embroidered logo) is a generic
  example of what a name tag looks like — it is **not** a frame from `sample.mp4`, and the
  colours don't carry over.
- In `sample.mp4` itself, the tagged person wears a light blue/white top (later a dark
  jacket over it, partway through the clip — see "Known limitations" below). The tag/logo is
  a small patch on the chest; at this camera's resolution and top-down angle it is **not**
  resolvable as text or a logo shape — confirmed directly by zooming 4x into the one photo
  known for certain to show the actual tag (the reference crop itself), which showed no
  discernible pattern at all, just a smooth colour blob. So the tag is only usable as "this
  torso region reads as a distinct colour," never as an OCR'd or template-matched graphic.
- The task states all other people visible are untagged.

### Data availability

Only `sample.mp4` is available for this task — **confirmed directly with the user**, despite
the brief mentioning "a few hundred video samples" might exist. This rules out any model
training or fine-tuning (no dataset to train on); the pipeline is built entirely from
pretrained, off-the-shelf models (`YOLOv8-seg`, trained by Ultralytics on COCO; CLIP, trained
by OpenAI — neither fine-tuned here) plus classical, non-learned color matching. This is a
data constraint, not a design preference — worth stating plainly if asked in the interview.

### Video characteristics

- `sample.mp4`: 960x720, 25 fps, 1341 frames (~53.6s), fisheye ceiling-mounted overhead
  camera, static (does not pan/move). Scene is an office: desks/seated people around the
  edges, an open corridor down the middle where people walk through.
- Because it's fisheye + directly overhead, people are heavily foreshortened and don't look
  like the eye-level pedestrians most public detectors/embeddings are trained on — this
  domain shift has been the recurring theme of this whole project (see `LOG.md`).
- More than one person in this office wears similarly light-colored clothing, and the
  tagged person's own clothing changes partway through (shirt, then a jacket over it) —
  color alone cannot fully disambiguate every case; see "Known limitations."

## Assumptions

Consolidated in one place since this is exactly the kind of question worth having a clean
answer for in the interview. Each links to where it's discussed in more depth.

1. **A human bootstraps identity; the pipeline never discovers "who is staff" on its own.**
   Nothing here reads a name tag or recognizes a badge — a human picks one reference crop, and
   everything downstream is "does this look/move like that." This is arguably the single
   biggest assumption: the system's job is propagating a human-given example through the video,
   not detecting "staff-ness" from scratch. Reasonable given the tag isn't resolvable at this
   resolution (see "What 'staff' looks like" above) — but worth being upfront that it's not a
   from-scratch identification system.
2. **Exactly one person is genuinely staff at any given moment.** Stated directly in the task
   brief (other people are explicitly untagged) and now mechanically enforced —
   `resolve_simultaneous_staff_conflicts()` keeps only the single highest-scoring track when
   two auto-qualify simultaneously (see "Current approach" and `LOG.md`, "Round 2"). Only
   catches the *simultaneous* case, though — a similarly-dressed person walking through alone
   at a different time, scoring above threshold, would still slip through uncaught. See "Known
   limitations."
3. **Only a *walking* person can be counted as staff.** A deliberate scope restriction, not
   just a filter — a seated/stationary high-color-match is judged too ambiguous to trust (this
   office has more than one similarly-dressed person at desks), while an actually-walking
   person through the open corridor was the one context visually confirmed unambiguous. If the
   real staff member in an unseen test video spends meaningful time seated, this pipeline would
   simply never count them as present during that time — untested against that scenario.
4. **Clothing color is a sufficient, if imperfect, proxy for identity**, because the tag itself
   isn't resolvable at this resolution (see "What 'staff' looks like"). This is the ceiling the
   whole pipeline operates under: whenever two people share a similar color (the "more than one
   similarly-dressed person" and "clothing-change" limitations below), the pipeline is
   fundamentally guessing, mitigated only by the walking gate, conflict resolution, and
   human-in-the-loop review — not solved.
5. **A track ID means the same physical person, unless a jump/color check says otherwise.**
   The pipeline trusts ByteTrack's identity assignment by default, and only distrusts it when a
   fragment's position or color diverges sharply from its neighbors
   (`split_event_by_position_jump()`, `split_event_by_color_outliers()`). A quiet,
   slow, same-colored ID-switch with no detectable jump would go uncaught — the one real
   ID-switch found and fixed (see `LOG.md`, "Round 2") happened to produce both a color-ceiling
   *and* a sharp position jump, which is what made it catchable at all.
6. **A short, spatially-close gap between two sightings means the same person continuing, not
   a different one** ("the person doesn't teleport" — `bridge_track_fragments()`,
   `interpolate_staff_gaps()`). Reasonable in this sparse office scene, where two different
   people can't occupy near-identical space within ~1s of each other except at a literal
   contact/handoff (which is exactly the case the position-jump check exists to catch
   separately) — would need re-examining in a denser crowd, where that assumption is weaker.
7. **A single reference crop, picked once, represents the staff's appearance for as long as
   their outfit doesn't change.** One frame's masked-crop color is treated as the ground truth
   for a whole "outfit era" — reasonable for a short clip, more fragile over a longer video with
   more lighting/pose variation. **Which era gets picked matters a lot, confirmed directly**:
   across three separate interactive picks on the same video, `#107`'s score swung from 0.55 to
   0.86 purely from which reference frame a human chose that time — one run's reference turned
   out to be the dark-jacket era rather than the light-blue-shirt era, which flipped the whole
   video's ranking (the real shirt-period staff frames scored *lower*, 0.39-0.43, than several
   unrelated dark-clothed bystanders, 0.70-0.86 — see "Known limitations"). Dark/generic clothing
   is more confusable with random bystanders than a distinctive light color, so the shirt era is
   the safer pick for `sample.mp4` specifically — worth saying explicitly to whoever runs this
   live, not left implicit.
8. **Static, non-panning camera.** The walking-motion gate computes speed/range directly in raw
   pixel coordinates frame-to-frame; a panning/zooming camera would need motion computed in a
   stabilized or world coordinate frame instead.
9. **The unseen test video will have broadly similar camera geometry to `sample.mp4`**
   (960x720, ~25fps, ceiling-mounted fisheye, similar height/angle). Every pixel-based
   constant (`--min-walk-speed`, `--min-walk-range`, `BRIDGE_MAX_JUMP_PX`,
   `LAB_DIST_SCALE`, etc.) is calibrated to this specific footage and is explicitly *not*
   claimed to transfer — see "Calibration reference points" below, and the pipeline always
   prints a fresh per-track diagnostic table so these can be sanity-checked (or re-tuned)
   against any new video rather than trusted blindly.
10. **No training data exists beyond `sample.mp4` itself** (confirmed with the user — see "Data
    availability" above) — forces every model to be pretrained and unmodified, and rules out
    any approach that would need labeled examples of this specific staff member or camera angle.

## Current approach & why

- **Detection + tracking:** `YOLOv8-seg` (permissive confidence, so it emits a box for
  everyone including partial/foreshortened poses) + ByteTrack (Kalman + Hungarian matching,
  via `ultralytics`' built-in tracker), in one pass. Re-validated directly on this footage
  after an earlier attempt (wrongly) ruled YOLO out based on a single unlucky frame — see
  `LOG.md`, "Round 2." Detection confidence never decides who's staff; appearance matching
  does, on every detected person.
- **ROI extraction is mask-aware:** each box's non-mask pixels are replaced with a neutral
  gray before any color is read, so a loose box that happens to include desk/floor doesn't
  dilute the measurement. This was the single biggest source of false positives in the
  earlier attempt, root-caused and fixed for good here.
- **Appearance matching is mean CIE-Lab color distance** to a one-shot reference crop —
  classical, not learned. CLIP (a pretrained embedding model) was tried as an alternative and
  measured *anti-correlated* with the true match on this footage (the true match scored
  lowest, clearly-wrong tracks scored highest) — kept available but off by default. Lab
  distance alone cleanly separates true matches (~20-30 distance) from everyone else
  (~60-100+).
- **A track only counts as staff if it's also walking** (speed + positional range computed
  from its own tracked coordinates), not just color-matching. This exists because the
  detector (unlike an earlier motion-only approach) also matches seated people, and more than
  one person here wears similarly light clothing — color can't disambiguate two people at the
  same desk, but a person actually walking through the open corridor was the one context
  confirmed unambiguous by direct visual inspection.
- **Ambiguous cases are flagged for a human to confirm, not guessed.** A walking person whose
  color doesn't match could be a different person, or the staff member after a clothing
  change — color genuinely can't tell those apart. Rather than automate this call (tested and
  rejected: on this exact video, a flagged case turned out to be a genuinely different
  person, so auto-labeling would have silently mislabeled someone), the pipeline pops up each
  ambiguous case for a one-keypress Y/N/S confirmation right in the same run. This is a
  deliberate "reject option" design choice, not a gap — see `LOG.md` for the reasoning and the
  real validation results (correctly caught a real clothing-change event; correctly left a
  genuinely-different-person event for the user to reject).
- **Short/marginal track fragments next to a confirmed staff sighting are bridged in
  automatically** (`bridge_track_fragments()`), and true zero-detection gaps between two staff
  sightings are linearly interpolated (`interpolate_staff_gaps()`), both gated by the same
  "the person doesn't teleport" time+distance check (`BRIDGE_MAX_GAP_SECONDS`,
  `BRIDGE_MAX_JUMP_PX`). Added directly in response to the evaluation below finding that
  tracking fragmentation, not appearance-matching, was the dominant cause of missed frames.
- **A merged possible-staff event is split at any internal color discontinuity**
  (`split_event_by_color_outliers()`, `FRAGMENT_OUTLIER_LAB_DIST`) before being shown to the
  user, so a fragment whose own color diverges sharply from the rest of the event (e.g. a
  tracker ID-switch onto a different person mid-event) becomes its own separate confirmation
  instead of riding along inside one blanket yes/no. Added directly in response to the
  evaluation below finding a real ID-switch case; see "Known limitations" for how well it
  actually catches that case (partially, not fully — it's still a color-based check).
- **Only one auto-qualified track can be "staff" at any given moment**
  (`resolve_simultaneous_staff_conflicts()`) — the task brief states there's exactly one tagged
  person, so two tracks both clearing `--staff-threshold` at the *same* time means at most one
  of them is right. Each track was previously judged independently, with no cross-check against
  what else was happening at the same moment. Added after a live run showed a sustained ~8s
  false positive (median score 0.55, a different similarly-clothed person) running green
  "STAFF" at the same time as the genuine staff (median score 0.92) in the same frames — see
  "Known limitations" and the "simultaneous-staff conflict" entry in `LOG.md` for the full
  story. **The loser is auto-rejected and logged to `auto_rejected_conflicts.csv`, not sent
  through the possible-staff review** — a simultaneous higher-scoring track is strong,
  objective evidence, unlike the subjective color/motion ambiguity the review is for, and
  routing it through review anyway was found (in the same live test) to flood the reviewer with
  already-settled questions.
- **A possible-staff event is split at any internal position jump, not just a color
  discontinuity** (`split_event_by_position_jump()`) — and the position check treats a genuine
  time *overlap* between two fragments specially, since "implied speed" isn't meaningful for
  zero/negative elapsed time; it instead checks whether the fragments are far apart in space
  while coexisting, which is direct proof of two people regardless of any speed calculation. An
  earlier version divided by a forced denominator of 1 whenever fragments overlapped, which
  misfired on ordinary tracker handoffs of the *same* person — found directly in testing when it
  fragmented one continuous walk into six separate reviews. See "Known limitations" and
  `LOG.md`, "Round 2."

## Calibration reference points (`sample.mp4`)

- Confirmed walking window (visual ground truth): frames ~380-641.
- Reference appearance crop used throughout testing: frame 402, box `(476, 488, 106, 173)`
  (x, y, w, h) — a clear, mostly-unoccluded view of the tagged person's torso. This (or any
  other reasonable pick — the pipeline isn't sensitive to exactly which frame/box) is passed
  via `--ref-frame`/`--ref-box`, or picked interactively with no flags.
- `LAB_DIST_SCALE = 80.0`, `--staff-threshold` default `0.5`: calibrated against this clip's
  real score distribution (median track score ≈0.2-0.3 globally; true-match tracks ≈0.6-0.9).
- `--min-walk-speed 6.0` / `--min-walk-range 40.0` (px/frame, px): calibrated against measured
  real walking tracks (12-17px/frame speed, 96-250px range) vs. the seated/desk cluster
  (0.6-4.2px/frame, 5-91px range) — a clean separation on this footage.
- `EVENT_MERGE_GAP_SECONDS = 5.0`, `COLOR_REPEAT_LAB_DIST = 20.0`: govern the possible-staff
  flagging logic's event-grouping and repeat-color rejection. The merge gap was initially
  2.0s and found (by testing) to be too tight — even a confirmed real walking track
  fragments with gaps up to ~2.2s, so a 2.0s merge window split one real occurrence into
  several pieces that then falsely "confirmed" each other as a repeating pattern.
- `FRAGMENT_OUTLIER_LAB_DIST = 35.0`: how far a fragment's own color has to diverge from its
  event's majority color to be split out as its own sub-event (see "Current approach"). Not
  independently tuned yet beyond "clearly above typical same-person frame-to-frame variation" —
  worth revisiting with more labeled examples.
- `BRIDGE_MAX_GAP_SECONDS = 1.0`, `BRIDGE_MAX_JUMP_PX = 120.0`, `BRIDGE_COLOR_FLOOR = 0.3`:
  govern fragment-bridging and gap-interpolation (see "Current approach"). Deliberately tighter
  than `EVENT_MERGE_GAP_SECONDS`/`COLOR_REPEAT_LAB_DIST` above — bridging silently pulls a
  fragment into the *confirmed* staff result with no human review at all, so it should only
  fire on short, spatially-obvious continuations, not anything as loose as the possible-staff
  event grouping (which a human still gets to check).
- **All of these are dataset-specific constants, not universal** — re-check the per-track
  diagnostic printout (which the pipeline always prints) against any new test video rather
  than assuming these transfer.

## Evaluation (`sample.mp4`)

Ground truth: the user watched the full video and reported the true staff-visible windows as
**12s-23s** and **31s-50s** (mm:ss from their video player, so +-1s or so, not frame-exact).
Diffing that against one full run's `staff_detections.csv` (`output_ui_test/`, 1343 frames,
25fps) gives:

- **Frame-level presence, naive:** precision **1.000**, recall **0.365**, F1 **0.534**
  (275 predicted-positive frames, all inside a true window -> zero false positives by this
  measure; 479 of the 754 true-positive frames missed).
- **Frame-level presence, identity-aware** (see below for why): precision **0.920**, recall
  **0.336**, F1 **0.492**.
- Recall is low in *both* windows, not just the jacket one: 30.8% in-window during 12s-23s
  (plain shirt, the "easy" period) and 39.7% during 31s-50s (jacket). So the biggest driver of
  missed frames isn't the clothing-change appearance ceiling already documented below — it's
  **tracking fragmentation**: ByteTrack loses and re-acquires the person repeatedly even while
  their appearance matches fine, splitting one continuous walk into many short track-ID
  fragments, some of which individually fail `--min-track-seconds` / the walking-motion gate
  and so never get counted as staff at all, leaving frame-level gaps mid-walk. This was not
  previously measured and is a bigger practical limitation than the clothing-change ceiling.
- **A tracker ID-switch was caught and confirmed by the user watching `annotated.mp4` frame
  by frame**, inside the reviewed/human-confirmed possible-staff event (frames 826-1231,
  33.0s-49.2s, 9 track fragments, promoted to STAFF as one block via the Y/N/S review): around
  **48.2s** a second person (dark shirt, a numbered badge — not the staff's tag) briefly makes
  contact with the staff; the track handoff from fragment `#303` (ends 48.20s) to fragment
  `#324` (starts 48.28s, 22 frames / ~0.9s) happens almost seamlessly, and `#324` is that other
  person, not the staff, per the user's direct visual check. Because the whole 9-fragment event
  was confirmed as one block, those 22 frames got swept in as "STAFF" along with the genuine
  fragments around them. **This is the source of the "identity-aware" precision above** —
  0.920 instead of a naive 1.000, treating those 22 frames as wrong-identity rather than
  correct. It's a real, if narrow, finding: presence-only precision/recall cannot see a
  right-time-window-but-wrong-person error at all; only the bonus (x, y)/identity check (here,
  a human re-watching `annotated.mp4`) caught it. Also notable: after `#324` ends at 49.24s, no
  further track gets marked STAFF through the end of the true window (50.00s) in this run — so
  whatever the user saw as "the box goes back to the staff" afterward, that hand-back either
  didn't happen in this run's actual output, or happened under a track that didn't individually
  clear the score/motion thresholds (unreviewed, since it fell outside the one flagged event).
- Caveats on all the above: one run, one video, ground truth to the nearest second from a human
  watching a video player (not frame-exact), and the "identity-aware" adjustment covers only
  the one mismatch a human happened to catch and report — there could be others unverified.
  Treat these as real, directionally meaningful numbers, not a polished benchmark.

### Re-measured after implementing the fragment-bridging and position-jump fixes

A later full, interactive run (`output/`, user's own reference pick, both fixes enabled, real
Y/N/S answers on all 4 review events) gives **precision 1.000, recall 0.448, F1 0.619** against
the same ground truth — recall up from 0.365 (naive, pre-fix) with precision still perfect.
Two things worth recording about *how* this number was reached, not just the number itself:

- The user's first pass through the review popups **pressed Y on event 4 again** — the exact
  same black-shirt/badge person the position-jump fix was built to isolate, correctly flagged
  this run with a "tracker ID-switch during contact" note. Asked why, the user said the crop was
  too tight to see the face, so they judged by shirt color — the one signal that's actually
  confounded here (see "Evaluation" above: `#303`/`#324`'s colors are nearly identical). Caught
  by re-checking `possible_staff_review.csv` against the known ground truth before trusting the
  run, then corrected by directly patching `staff_detections.csv` (flip that track's 22 rows to
  not-present) and `possible_staff_review.csv`'s resolution, rather than re-running the whole
  pipeline. **This is exactly the scenario the human-in-the-loop design was built to be honest
  about**: even a correctly-flagged, correctly-explained case can still get a wrong answer from
  a rushed or under-informed human — automating it outright would have the same failure mode
  with no chance to catch it afterward. **Fixed:** the review popup now shows two generously
  padded context crops (start and end of the fragment) with the person highlighted, instead of
  one tight torso-only crop, plus a visible "Flagged: ..." line explaining *why* an event was
  flagged (position jump / color divergence) — directly targeting what went wrong here: no
  identity context, and no cue to look for anything beyond clothing color. Not a guarantee
  against every future wrong answer (a human can still misjudge with more context), but a
  direct response to the specific, real failure found. Also enlarged the whole popup (bigger
  photos, bigger text, window explicitly resized to match) after a live test found the default
  noticeably too small to read comfortably.
- Patching the CSV directly (instead of re-running) fixes the data but not `output/annotated.mp4`,
  which still shows a green box on the black-shirt person for that ~1s stretch — a cosmetic
  inconsistency worth knowing about before using that specific video for a demo.

## Known limitations / what to say if asked "what would you improve"

- **Tracking fragmentation was the single biggest driver of missed frames, measured directly**
  (see "Evaluation" above): even during the plain-shirt period, where appearance matching
  works fine, ByteTrack only sustained a "staff" track for ~31% of the frames the person was
  actually visible for, because losing and re-acquiring the person split one walk into many
  short track fragments, some too brief to individually clear the length/motion thresholds.
  **Partially fixed:** `bridge_track_fragments()` pulls a short/marginal fragment into the
  staff result when it's within `BRIDGE_MAX_GAP_SECONDS`/`BRIDGE_MAX_JUMP_PX` of an
  already-confirmed staff fragment, and `interpolate_staff_gaps()` linearly fills true
  zero-detection gaps between two staff sightings under the same constraints. Measured effect
  on `sample.mp4`: in-window recall during the plain-shirt period went from 30.8% to 34.1% in
  one test run (9 frames gap-filled by interpolation; no fragments met the bridging condition
  in that particular run) — a real but modest gain, not a full fix. The remaining shortfall
  looks like it's mostly gaps *longer* than `BRIDGE_MAX_GAP_SECONDS` (1.0s), not many short
  nearby fragments — worth re-measuring the actual gap-length distribution before tuning
  further, rather than just widening the constants blindly.
- **A tracker ID-switch during physical contact between two people can misattribute a few
  frames to the wrong person even inside a mostly-correct, human-confirmed event** — caught
  once, directly, by the user re-watching `annotated.mp4` (see "Evaluation"). **Fixed and
  verified against the real case:** `split_event_by_position_jump()` splits a merged event
  wherever consecutive fragments imply a physically-impossible speed between them
  (`MAX_BRIDGE_SPEED_PX_PER_FRAME`), run *before* `split_event_by_color_outliers()` specifically
  because color alone measurably failed on the real case — the staff's dark jacket and the
  other person's dark shirt had nearly identical mean Lab color (distance ~4, far under
  `FRAGMENT_OUTLIER_LAB_DIST`'s 35), but the position jump between their fragments (~126px in
  ~1 frame, vs. a real max walking speed of ~17px/frame) was unambiguous. Re-running the exact
  same tracking data offline after the fix confirms it isolates precisely the reported
  fragment (`#324`, the black-shirt/badge person) into its own flagged sub-event, cleanly
  separate from the genuine staff fragments around it — the other, larger sub-events are not
  flagged. This is a genuinely narrow fix (one verified case), not a general solved problem:
  it still can't catch an ID-switch that's both same-colored *and* spatially smooth (e.g. two
  people already walking side by side, no contact/jump moment) — that would need a stronger
  signal than either color or position/velocity alone.
- **Color-only matching has a real ceiling, most visibly when the tagged person's own
  clothing changes** (shirt, then a jacket, partway through this clip) — a fixed reference
  crop's color stops matching regardless of tracking. Tested and ruled out: (a) template/logo
  matching (the tag isn't resolvable at this resolution, confirmed directly), (b) full
  clothes-changing person re-identification (a genuinely hard, actively-researched problem,
  and likely to hit the same eye-level-vs-top-down domain shift that's affected every learned
  model tried on this footage). Current mitigation: flag ambiguous cases for a human to
  confirm rather than guess — validated to work correctly on real examples from this video,
  but it is a human-in-the-loop mitigation, not a fully automatic solution.
- **More than one person in this office wears similarly light-colored clothing.** Color alone
  cannot disambiguate two such people if both happen to be walking — the walking-motion gate
  and possible-staff flagging both exist because of this, not as a hypothetical edge case.
  **Confirmed with a real, sustained example**: a track ran green "STAFF" for ~8 seconds (201
  frames, median score 0.55 — a different person, close enough in color to clear
  `--staff-threshold` on its own) at the same time the genuine staff (median score 0.92) was
  also on screen. **Partially mitigated:** `resolve_simultaneous_staff_conflicts()` now catches
  exactly this shape of error — two auto-qualified tracks active simultaneously, at most one
  can be right, keep the higher-scoring one and auto-reject the other (logged, not sent to
  manual review — see "Current approach" for why) — but it only helps when the two people are
  actually on screen *at the same time*. It does nothing for a similarly-dressed person who
  happens to walk through alone, at a different time, with a score that clears the threshold —
  that case would still be silently auto-labeled staff. The underlying limitation (color can't
  reliably tell two similarly-dressed people apart) is not solved, only caught in the one shape
  where it's checkable "for free."
- **No training data beyond `sample.mp4` itself** (confirmed with the user) — every model
  used is pretrained and unmodified; nothing here was fine-tuned or trained specifically for
  this task, camera angle, or this specific staff member.
- Assumes a static, non-panning camera (true for this sensor type) — the walking-motion gate
  computes speed/range in raw pixel coordinates, which would need re-deriving if the camera
  could move.

## Status

Pipeline (`src/staff_id.py`) runs end-to-end, guided (no flags) or scripted, and produces
`output/staff_detections.csv`, `output/annotated.mp4`, `output/reference_crop.jpg`, and (when
anything is flagged) `output/possible_staff_review.csv` + `output/review_event_*.jpg`.
Validated on `sample.mp4` both visually and (see "Evaluation" above) quantitatively against
user-provided ground truth: correctly identifies the confirmed walking windows with zero
naive false positives, correctly flags-and-resolves a real clothing-change event via the
review prompt, and correctly leaves a genuinely-different-person event for the user to reject
rather than guessing wrong — but recall is well below 100% (measured ~35-40%), driven mainly by
tracking fragmentation, not the appearance-matching logic itself.

Since then: fragment-bridging/gap-interpolation, simultaneous-staff conflict resolution,
position-jump event splitting, a redesigned/enlarged review popup, and a reference-crop-choice
tip were all added and individually verified (see `LOG.md`, "Round 2" — several entries). User
confirmed directly, after re-testing live, that the current version is the best-performing one
so far.

## Deliverables still needed

- [ ] 1-2 page write-up (`DOCUMENTATION.md` — currently empty; the "Evaluation" section above
      has the numbers to draw from)
- [x] Precision/recall/F1 evaluation against labeled frames (`SOLUTION_PLAN.md` 2.6) — done,
      see "Evaluation" above; ground truth from the user watching `sample.mp4` directly
- [ ] Test against a second video (only ever tested against `sample.mp4` so far)
- [x] Visual verification of the Savitzky-Golay coordinate smoothing's effect on jitter — done,
      see `LOG.md`; effect is real but subtle (removes small single-frame wiggles, no visible
      lag/overshoot), given the fixed ~5-frame smoothing window the current code always uses
- [x] Implement a fix for tracking fragmentation (gap-bridging/interpolation) and for the
      tracker ID-switch (per-fragment position/color discontinuity splitting) — done, see
      "Known limitations" for what each fix actually achieved (fragmentation: partial,
      measured modest gain; ID-switch: verified fix for the one real case found, still
      narrow — see the "genuinely narrow fix" note there) and `LOG.md` for the implementation
      and validation story
- [x] Re-measure precision/recall/F1 with both fixes enabled end-to-end (interactively
      confirming the review events, not `--skip-review`) — done, see "Evaluation," "Re-measured
      after implementing the fragment-bridging and position-jump fixes"

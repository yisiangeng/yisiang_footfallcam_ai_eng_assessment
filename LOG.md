# LOG.md

Running chronological log of work on this project: what was done, in what order, what broke,
and what to remember going forward. This is the append-only history — `KNOWLEDGE.md` is the
current-state reference (rewritten as understanding changes), `SOLUTION_PLAN.md` is the active
design doc. This file absorbs and replaces `TESTING1_NOTES.md`, whose full content is preserved
below as the "Testing 1" entry.

---

## Testing 1 — first attempt (superseded, kept for reference)

Single-file pipeline (`src/staff_id.py`), MOG2-motion-detection-based. Left in place as a
working reference implementation; not being modified further. Output from its last completed
run lives in `output/`.

**Task:** identify frames of `sample.mp4` (960x720, 25fps, 1341 frames, fisheye ceiling-mounted
static camera, office scene) containing the "staff" person (light blue/white top; the tag
itself isn't legible at this resolution), plus bonus (x,y) localization. Ground-truth window
established by manual inspection: staff visibly walks the corridor frames 380-641.

Two task-brief facts worth keeping precise: the PDF's illustration photo (a green polo with a
white embroidered logo) is a **generic example** of what a name tag looks like, not a frame
from `sample.mp4` — the colors don't carry over, and the actual tagged person in the video
wears the light blue/white top described above. (`SOLUTION_PLAN.md`'s task recap had drifted
into calling the actual staff's shirt "green" — corrected once this was caught during
`KNOWLEDGE.md`'s rewrite.) Separately, the brief states two other people cross the corridor
without the tag — all people in the video besides the one tagged person are untagged.

### What was tried, in order

1. **Stock YOLOv8 (n/m) as primary person detector — abandoned.** Tested on frame 402 only:
   gave the walker just ~0.18-0.41 confidence, similar to or lower than static seated people.
   Concluded top-down fisheye view was too out-of-distribution for a COCO-trained detector.
   *(Note: this conclusion was later re-tested and overturned in Round 2 — see that section.)*
2. **MOG2 background subtraction as primary detector — kept, worked well as the walker-finder.**
   Reliable on this static camera, no model weights needed. Downside: fires on any motion
   (chairs, gestures, shadows) — high recall, poor precision. Also produces *loose* contour
   boxes that can merge a person's silhouette with adjacent furniture via morphological
   closing — this turned out to be the root cause of the final unresolved false positive
   (see "Where work stopped").
3. **YOLOv8n as a secondary verification gate — kept.** Not used to find people, only to
   reject motion blobs that aren't person-shaped (chairs, papers), at a low confidence bar
   (0.08). First version used IoU (min 0.03) between blob and YOLO box — too permissive, let
   a chair-shaped blob slip through via a nearby unrelated person's box. Fixed by switching to
   containment ratio (intersection/blob-area >= 0.4), which correctly rejected it.
4. **Per-frame decision, not per-track — switched after track-based approach failed.** A
   simple greedy centroid tracker was built, and an initial design picked the single track
   with the highest average appearance score as "the staff." This picked the *wrong* person: a
   short 35-frame track edged out the real 506-frame walker track by a razor-thin margin
   (0.499 vs 0.489), because the tracker was silently swapping identities between people who
   passed close together (confirmed by a track's centroid jumping incoherently, e.g. y=86 to
   y=643 within ~50 frames). Fixed by making every frame independently pick its own
   best-scoring candidate, with the tracker kept only for smoother visualization.
5. **Appearance/tag matching — three iterations, ending on Lab color distance.**
   - v1 (HSV histogram, H+S only): failed — a black hoodie scored ~0.6-0.8 similar to the
     light-blue reference, because both a dark garment and the beige floor background are
     low-saturation, and crops are background-dominated.
   - v2 (added V channel): better (one measured false positive dropped 0.836 -> 0.457) but
     still not reliable enough — a whole-crop histogram stays ~50%+ background/hair pixels.
   - v3 (final, kept): mean CIE-Lab distance over a trimmed "torso window" (central ~65%
     width x 55% height of the box), plus a minor 15%-weighted ORB keypoint term. Real matches
     landed ~20-30 Lab distance vs. >100 for unrelated dark clothing in controlled tests; real
     video scores were noisier (best-per-frame percentiles p50~0.38-0.46, p90~0.54-0.58) with
     real overlap between "true match, bad angle" and "unrelated match, lucky lighting."
6. **Presence-decision smoothing — three iterations, ending on hysteresis.**
   - v1 (flat threshold): fragmented one continuous walk-through into many tiny sub-second runs.
   - v2 (+ minimum-event-duration filter): over-corrected — since v1's fragments were already
     shorter than the duration cutoff, the true window shrank to a single 15-frame remnant.
   - v3 (final, kept): ~0.3s rolling average feeding hysteresis (high enter threshold = the
     staff-threshold, lower exit threshold = 0.6x, ~0.5s patience window). Kept one real walk
     as one event while letting brief false matches lapse. `--staff-threshold` settled at 0.55
     after testing 0.35 (far too permissive), 0.5, 0.6 (cut too much real recall).
7. **Spatial-coherence gate — added, didn't fix the last known bug.** Rejected *weak*
   (sub-threshold) matches that jumped too far from the recent trajectory. Made no difference
   to the one remaining false-positive cluster (frames 917-934) because that match scored
   *strongly* (0.62-0.64, above the enter threshold on its own merit) — the gate only filters
   weak candidates, so the real bug was one level deeper, in the matcher itself.

### Where work stopped

Root-caused the frame 917-934 false positive (dark-clothed, untagged person, scoring
0.62-0.64) to bounding box `(555, 297, 135, 228)` — a loose motion-detection box where the
person's actual silhouette is a small fraction of the box, the rest being desk/floor
background. The proportional torso-window trim wasn't enough to exclude that much background
when the box itself was oversized.

An in-progress fix was left incomplete: sample Lab color only from the actual foreground-mask
pixels within the box (from MOG2), not a fixed proportional crop. `MotionDetector.detect()`
was updated to expose `self.last_mask`, but `AppearanceMatcher` and the `run()` pass-1 loop
were never updated to use it — so the saved `output/` results are from the pre-fix run and
still contain this one known false positive (frames 917-934; real detections at 365-403,
448-485, 526-573 are correct).

### Things to note going forward

1. **Bounding-box looseness was the recurring root cause**, not the color-matching formula —
   every matcher iteration (histogram -> +V -> Lab torso-window) improved things only until it
   hit a case where the *box*, not the pixels sampled from it, was the problem. Worth solving
   this class of bug once (mask-aware sampling / tighter box source) rather than continuing to
   patch the color metric.
2. **Containment ratio beats IoU** when box sizes differ a lot (motion blob vs. detector box).
3. **A simple greedy centroid tracker's identity cannot be trusted** across a busy scene;
   decisions need to be per-frame, or use a real tracker (Kalman + Hungarian matching).
4. **Hysteresis + rolling average smooths noise but can't fix a matcher that's sometimes
   confidently wrong** — flat thresholds fragment continuous events, but temporal smoothing is
   not a substitute for a reliable per-frame signal.
5. **Color-only re-identification has a real ceiling** at this resolution/lighting. A learned
   appearance embedding (e.g. CLIP few-shot) was identified as the natural next step.
6. Stray debug images from ad-hoc calibration during this round (`crop665_5.jpg`, `dbg_918.jpg`,
   `fp_crop.jpg`, `frame900_check.jpg`, `ref_crop_nopad.jpg`) are archived in
   `debug_archive/round1_crops/`, not deleted, since they're referenced above.

---

## Round 2 — design restart

Per user direction: keep Testing 1's code/output as reference (not modified further), start a
fresh design discussion before writing new code, and produce a proper plan doc first.

- Wrote `SOLUTION_PLAN.md` — a new cascaded detect -> track -> ROI-extract -> classify ->
  smooth pipeline design, with a row-per-detection CSV schema (fixes Testing 1's schema, which
  could only represent one staff member per frame) and a live-demo interactive reference
  calibration workflow (`cv2.selectROI` + frame-scrub trackbar, since a new/unseen test video
  will be used at the interview and there's no way to pre-pick a reference crop for it ahead of
  time).
- Reviewed the initial plan draft and flagged three issues before implementation: (a) it
  proposed HSV as a lead tag-matching method, which Testing 1 already disproved; (b) it didn't
  explicitly require mask-aware ROI extraction, despite that being Testing 1's dominant root
  cause; (c) it assumed YOLO would work as the primary detector without re-checking Testing 1's
  negative result first. Plan updated: HSV dropped, CLIP few-shot promoted to primary matcher
  (Lab distance kept as fallback), mask-aware ROI extraction made an explicit requirement.
- **Re-validated the YOLO-as-primary-detector question with real data, rather than trusting
  Testing 1's frame-402-only test.** Result: overturned the earlier conclusion. Across all 65
  frames with a known walker position in the confirmed walker window (380-641), YOLOv8n placed
  a person-box within 80px of the true walker position in **65/65 frames (100%)** — frame 402
  alone was an outlier (the walker is caught mid-turn there, mostly foreshortened
  head/shoulders), not representative of the clip. `SOLUTION_PLAN.md` updated: YOLOv8 is now
  the primary detector, MOG2 demoted to an offline/overlap-disambiguation fallback. Key
  correction versus Testing 1's implicit assumption: don't rely on "highest-confidence box =
  the walker" — detect all person boxes at a permissive threshold and let the separate
  appearance-matching stage decide who's staff.
- Tidied the project root: moved Testing 1's stray debug crop images into
  `debug_archive/round1_crops/`, removed `src/__pycache__` (compiled bytecode, safe to
  regenerate). Left `yolov8n.pt`, `output/`, `src/staff_id.py` untouched.
- Created `Documentation.md` (intended for the deliverable write-up) and this file.

### Implementation (`src/staff_id_v2.py`)

Built as a single new file alongside `src/staff_id.py` (kept untouched as reference), writing
to `output_v2/`. Validated each new component individually before combining them (YOLOv8-seg +
ByteTrack via `model.track(tracker='bytetrack.yaml')`, and CLIP via `transformers`) — worth
noting one library quirk hit along the way: this environment's `transformers` (5.14.1)
refactored `CLIPModel.get_image_features()` to return a `BaseModelOutputWithPooling` wrapper
whose `.pooler_output` holds the actual projected embedding, not a plain tensor as older
docs/examples assume.

**Two significant findings changed the design after real data disagreed with the plan:**

1. **CLIP was empirically anti-correlated with the true match, not just weak — reversed
   from primary to optional/off.** The plan (per the design-review pass above) had promoted
   CLIP few-shot cosine similarity to the primary tag-matcher on the theory that a learned
   embedding would generalize better than hand-crafted Lab distance. Tested side-by-side on
   the same tracked detections: Lab distance cleanly separated the true-match tracks (~21-27)
   from every other track (~60-98) — a clean re-confirmation of Testing 1's finding. CLIP
   cosine similarity, on the exact same crops, gave the true-match tracks the *lowest* scores
   (0.69-0.71) and the clearly-wrong tracks the *highest* (0.78-0.79) — backwards. Most likely
   cause: CLIP ViT-B/32's pretraining (natural eye-level photos) doesn't transfer its
   clothing-color discrimination to these small, blurry, top-down fisheye crops — the same
   domain-shift problem that originally (and wrongly, per the earlier re-validation) ruled out
   YOLO, but here it hits embedding quality rather than localization. Reverted: Lab distance is
   the primary signal (0.85 weight), CLIP kept only as an opt-in `--use-clip` flag, off by
   default. Also dropping CLIP by default cut pass-1 runtime roughly 3x (44s -> 14s on a
   100-frame test clip), which matters for the live-demo speed requirement.
2. **Running the corrected pipeline on the full video surfaced a real ambiguity the plan
   hadn't anticipated: more than one person in this office wears a similarly light-colored
   top.** Round 1's motion-only detector could never see this, since it only ever looked at
   the one walking event. YOLOv8-seg detects people whether or not they're moving, so the
   full-video run flagged several additional high-scoring windows outside round 1's confirmed
   walker window (380-641) — some visually confirmed as the same target now seated at a desk
   (plausible), others showing two people in similar light-blue tops standing close together
   (genuinely ambiguous — color alone can't tell them apart). Flagged this to the user with
   the visual evidence rather than picking a resolution unilaterally. **User decision:
   restrict "staff present" to confirmed walking events.** Implemented as a motion gate
   computed directly from each track's own tracked coordinates (no separate motion detector
   needed): average centroid speed >=6px/frame AND positional range >=40px. Calibrated against
   real data — genuine walking tracks measured 12-17px/frame speed with 96-252px range; the
   seated/desk cluster measured 0.6-4.2px/frame with 5-91px range. Range threshold was
   initially set to 100px by analogy to the speed number, then lowered to 40px after it was
   found to wrongly exclude the first ~10 frames of a genuine walking event (high speed, but
   not yet enough accumulated range since the person had just entered frame) — speed turned
   out to be the more reliable of the two signals, range is just a jitter floor.

**Result after both fixes:** on the full `sample.mp4`, presence runs are 364-390, 445-473,
523-560 — matching Testing 1's actual confirmed true positives (365-403, 448-485, 526-573)
closely, while correctly excluding Testing 1's one known false positive (917-934, never
surfaced here) and all of the seated-lookalike ambiguity above. Visually spot-checked
(`output_v2/annotated.mp4` frame 380): tight, correct green box on the true walker.

**Not yet done:** final threshold/parameter sensitivity pass on a second video (only tested
against `sample.mp4` so far), the 1-2 page write-up (`DOCUMENTATION.md`), precision/recall/F1
evaluation against labeled frames (2.6), Savitzky-Golay coordinate smoothing is implemented but
not yet visually verified for jitter reduction.

### Interactive-picker smoke test

Attempted to smoke-test the `cv2.selectROI` reference-picker path (previously untested — only
`--ref-frame`/`--ref-box` had been exercised). Running it under the system Python crashed with
`cv2.error: ... The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or
Cocoa support` at the first `cv2.namedWindow()` call (not a hang — it just took ~30-45s of
import/model-load time before reaching that line, which looked like a hang under a short
timeout at first). Root cause: the system Python has both `opencv-python` (GUI-capable) and
`opencv-python-headless` installed side by side — the latter pulled in by an unrelated package,
`easyocr` — and they conflict in the shared `cv2` import namespace, with the headless build's
GUI stubs winning. Confirmed by isolating a bare `cv2.namedWindow()` call, which reproduced the
same error instantly.

User confirmed they'll demo on this same machine and don't use `easyocr` elsewhere, but chose
the zero-risk fix anyway: a project-scoped virtual environment (`.venv/`) with a clean
`opencv-python` install (no headless variant), rather than uninstalling
`opencv-python-headless` from the system Python. Verified `cv2.namedWindow()` succeeds in
`.venv` — the underlying bug is fixed. The actual click-and-drag interaction still needs a
human (can't be driven from this session), so the full interactive flow itself remains
user-verified-pending, not yet confirmed end-to-end.

**Update:** user ran the real interactive picker end-to-end successfully (frame 556, box
502,69,109,109 — a different frame/box than any previously tested) and it correctly identified
the same 5 staff tracks as every prior run, confirming the pipeline isn't fragile to exactly
where the reference is picked. Hit one more environment gotcha along the way: `ultralytics`
auto-installed a missing dependency (`lap`) mid-run, which (like the earlier tracking smoke
test) doesn't take effect until the *next* process start, so that first run got no track IDs at
all. Fixed by adding `lap` to the venv setup command in `CLAUDE.md` up front.

### CLI UX overhaul

User asked for a friendlier interface: (1) a guided setup asking which video/output-folder
interactively instead of requiring `--video`/`--output-dir` flags, (2) clearer on-screen
instructions in the interactive picker (when to press ENTER vs. scrub vs. drag), (3) more
readable console output including total time taken. Implemented in `src/staff_id_v2.py`:
`guided_setup()` (auto-lists video files found in the working directory, prompts for an output
folder name, defaults to `output_v2`), an on-video instruction banner (semi-transparent strip
with numbered-step text) plus matching console prints at each picker stage, a `format_duration()`
helper, and a restructured per-track table that computes the staff decision *first* then prints
one aligned table with a plain-English verdict column (`STAFF` / `color doesn't match` / `not
walking (seated?)` / `too short to trust`) instead of a raw numeric dump followed by a separate
count. Added a final `DONE` summary block (video length, staff-frame percentage, total time
broken into detect+track+score vs. render, and every output file's path). All changes are
UI/formatting only — no scoring/matching logic touched. Verified via `py_compile` + an
end-to-end run on the short test clip and the guided-setup prompts with simulated stdin.

### Clothing-change failure mode + possible-staff flagging

User observed a real failure watching the annotated video: the staff member wore a shirt
initially (correctly identified), then later put on a jacket (also with a logo) and stopped
being detected. Root cause: matching is against one fixed reference color, so a genuine
clothing change breaks the color signal regardless of tracking — this is distinct from (and a
worse problem than) the earlier track-fragmentation issue.

User proposed logo/tag matching (recognize the tag itself, invariant to what garment it's on)
as a potential fix. **Tested feasibility directly rather than assuming**: zoomed 4x into
several "walking but zero color match" tracks found later in the video (826-1231, clustered
temporally like a single fragmented event) — no logo or pattern visible, just blurry dark
fabric, even though the raw boxes were a generous 90-200px, larger than our own reference crop.
Confirmed conclusively by zooming into **our own reference crop** (the one photo known for
certain to show the actual tag) the same way — it *also* shows no discernible logo, just a
smooth color blob. **Conclusion: logo/tag template matching is not viable at this camera's
actual resolution/quality**, settling something `KNOWLEDGE.md` had only suspected before.

User pushed back on the practical fallback (asking for a second reference photo) since it
breaks the "point once" live-demo experience the interviewer likely expects. Discussed
"clothes-changing person re-identification" as the technically-correct research answer, but
flagged it as high-risk given this session's repeated pattern of eye-level-trained models not
transferring to this top-down fisheye view (YOLO detection, then CLIP matching, likely a third
case). Agreed on a middle path instead: **flag ambiguous cases for human review rather than
auto-labeling or silently dropping them.**

**Implemented in `src/staff_id_v2.py`:** `find_possible_staff_events()` groups
walking-but-color-unmatched tracks into "events" (merging fragments separated by less than
`EVENT_MERGE_GAP_SECONDS`, since one real occurrence fragments into many track IDs just like
the confirmed staff track did), then drops any event whose mean color repeats in another event
elsewhere in the video (`COLOR_REPEAT_LAB_DIST` threshold) — a repeating color is better
explained by a regular non-staff person than a one-off clothing change. What survives gets
written to `possible_staff_review.csv` (one row per event, with a representative zoomed crop
image) and shown as a yellow box in `annotated.mp4`, distinct from green (confirmed staff) and
orange (other people).

**First test run found a real bug**, not just validated the concept: `EVENT_MERGE_GAP_SECONDS =
2.0` was too tight — the actual gaps between the black-clothing cluster's 9 track fragments
were 49-72 frames (1.96-2.88s), just over threshold, splitting one real occurrence into 3
separate "events" that then falsely confirmed *each other* as a repeating color and all got
dropped (result: 0 flagged, silently wrong). Root cause matched a pattern already seen with the
confirmed real staff track, which also fragmented with gaps up to ~2.2s — so 2.0s was simply
too tight a merge window in general, not a one-off fluke. Fixed by widening to 5.0s. Re-ran:
correctly merged all 9 fragments into one event (frames 826-1231, 33.0s-49.2s) and flagged it,
with zero false positives among the other 60+ tracked people. **User visually confirmed the
flagged event's representative crop is genuinely the staff member in the jacket** — the
mechanism is validated against real ground truth, not just internally consistent.

**Second independent test, run by the user themselves (not this session) with a different
reference pick — the jacket, not the shirt** (`reference_crop.jpg` shows dark clothing):
flagged 2 events this time. Event 1 (frames 364-560, 14.6-22.4s, 5 fragments) was the
*original shirt-wearing walk* — with a jacket-colored reference, that window no longer scores
as a direct color match, so it correctly fell through to the flagging safety net instead of
being lost entirely. User confirmed: yes, staff. Event 2 (frames 863-883, 34.5-35.3s, single
fragment) sat *between* two segments the same run had already confirmed directly as staff
(844-856 and 911-919) — this session hypothesized it was likely the same person mid-stride
with one poorly-lit/blurred fragment, and proposed merging flagged fragments by position
continuity with neighboring confirmed segments, not just time, as a possible improvement.
**User checked the continuous video (not just the still crop) and confirmed it's genuinely a
different person** — the hypothesis was wrong. Important catch: had that position-continuity
merge been implemented on the hypothesis alone, it would have been a real bug (wrongly
attributing a different person's presence to the staff member). Lesson banked: verify a
"looks like it should be the same person" instinct against ground truth before building on it,
even when the reasoning sounds plausible.

**Net result across both tests: 2/2 flagged events where a human confirmed "yes, staff" were
correct, and the flagging mechanism correctly avoided asserting a confident answer either way
on the one genuinely ambiguous different-person case** — instead of guessing wrong in either
direction, it surfaced exactly the two clips worth 20-30 seconds of a human's attention out of
a 53.6s video. This is the honest value proposition of `possible_staff_review.csv`: not full
automation, but a large reduction in what a human needs to check by hand.

### Windows file-lock crash, caught by the user

Re-running the pipeline while `possible_staff_review.csv` from the previous run was still open
in Excel crashed with `PermissionError: [Errno 13] Permission denied` right at the very end —
after the full ~3 minute detect/track/score pass had already completed, the worst possible
place to lose a run. Fixed by adding `open_for_write_retrying()` / `imwrite_retrying()` /
`open_video_writer_retrying()` wrappers around every output write (both CSVs, the reference
crop, the per-event review crops, and the annotated video's `VideoWriter`) — on a locked file
they print a clear message ("close it, then press Enter to retry") and retry instead of
crashing and discarding the completed processing work. Verified no regression on the normal
(unlocked) path; the retry path itself is a straightforward try/except around the exact
exception type seen in the real crash, not separately simulated under a live lock.

### Interactive confirmation for flagged events (human-in-the-loop, not full automation)

User asked whether the jacket-changed-staff case (flagged for review) could be resolved
automatically instead of requiring manual follow-up, and specifically whether a human should
stay in the loop at all. Discussed directly: recommended keeping a human in the loop, not
because full automation is impossible in principle, but because (a) real data from this same
video already proved a flagged event can genuinely be a different person (the earlier "event
2" case), so auto-promoting on color/uniqueness alone would sometimes silently mislabel
someone with no visible sign of error; (b) the underlying uncertainty is fundamental (tag
unreadable, clothes-changing re-ID unlikely to transfer to this camera angle), not a gap
better engineering would close; (c) the cost is asymmetric — a flagged case costs a human a
few seconds, a wrong silent auto-label costs an incorrect record in a system meant to track
who's staff. Framed as a "reject option" / selective-prediction design choice rather than a
limitation to apologize for.

Built accordingly: `confirm_event_interactive()` pops up each flagged event's representative
crop with a Y/N/S prompt (`Y` = confirm staff, promotes the event's tracks straight into the
result; `N` = confirm not staff; `S` = skip, stays unresolved) right after processing, instead
of requiring a separate manual Excel/photo-viewer investigation like the previous two test
rounds needed. `--skip-review` bypasses this for scripted/headless runs. `RESULT`/`DONE`
summary lines moved to after this step so the printed staff count reflects any
review-confirmed additions. Verified no regression on the automated (`--skip-review`) path;
the actual Y/N/S keypress interaction itself needs the user to test, same limitation as the
original reference-picker smoke test (this session has no way to send keypresses to a popped-up
window).

### Data availability, confirmed with the user

Asked directly ("where do you get the data to train and test the ML model?") and answered
plainly: neither model in this pipeline was trained here. `YOLOv8-seg` is pretrained by
Ultralytics on COCO; CLIP (when enabled) is pretrained by OpenAI — both used entirely
off-the-shelf, no fine-tuning. The brief mentions "a few hundred video samples" might be
available, which would open up real options (fine-tuning YOLO for this camera angle, training
a small staff/not-staff classifier). **User confirmed `sample.mp4` is the only video they
actually have** — so those options are off the table for this task, not by choice but by data
availability. This is *why* the pipeline leans on pretrained models plus classical color
matching rather than anything learned specifically for this problem, and it's worth being able
to say plainly in the interview if asked why no training happened.

### Folder cleanup + rename

User asked to tidy the project once round 2 was fully validated: deleted `output_v2_flagtest`
/ `output_v2_flagtest2` (test scaffolding, findings already captured above as prose), deleted
`src/staff_id.py` (round 1) + its `output/` + `yolov8n.pt` (only used by round 1's
`PersonVerifier`) since round 2 fully supersedes it and this file's "Testing 1" section already
preserves the full history, renamed `output_v2/` to `output/`, then renamed `staff_id_v2.py` to
`staff_id.py` now that there's no more "v1 vs v2" to disambiguate (updated every internal
self-reference and `CLAUDE.md` cross-reference accordingly). Before rewriting `KNOWLEDGE.md`
into a current-state doc (it had drifted into being entirely about round 1's now-deleted
design), checked it against this file for anything not already preserved elsewhere — found and
fixed two gaps, now captured above: the PDF-illustration-is-generic clarification, and this
data-availability confirmation.

### UI restyle (Japandi + glassmorphism) for the interactive windows

User asked for a visual overhaul of the interactive `cv2` windows (frame-scrub/box-drag
picker, Y/N/S review popup), specifically "Japandi style and glassmorphism." Since these are
raw `cv2` drawing calls, not a real UI toolkit, glass was simulated: `draw_glass_panel()`
blurs whatever is behind a panel, tints it with a warm neutral color, composites through a
rounded-corner mask, and adds a soft drop shadow + thin gold border. Palette: cream fill, warm
near-black ink text, muted sage (affirmative), muted terracotta (negative), muted sand
(secondary) — see `src/staff_id.py`'s `INK`/`CREAM`/`SAGE`/`TERRACOTTA`/`SAND`/`GOLD_LINE`
constants. Rebuilt three pieces: `_draw_instruction_banner()` (was a flat black bar), a new
`select_box_interactive()` replacing `cv2.selectROI` (adds a translucent selection fill, gold
corner handles, a live "W x H px" readout pill — `cv2.selectROI` is a black-box OpenCV widget
with its own chrome, impossible to restyle from the outside, so it had to be replaced outright
to get any control over its look), and `confirm_event_interactive()` (now a standalone Japandi
card with a frosted photo frame and three pill buttons that take either a keypress or a mouse
click). Verified by rendering each piece to a static image and inspecting it directly (no
display in this environment), plus `py_compile`.

Two real bugs found from the user's own live test run (not caught by the static renders,
since neither problem showed up on synthetic/random test images):
1. **Instruction banner text overflowed its own panel.** The panel's width was computed by
   measuring every line with `FONT_HERSHEY_SIMPLEX`, but the header line (index 0) is actually
   *drawn* in `FONT_HERSHEY_DUPLEX`, which is noticeably wider at the same scale — so the
   rendered text ran past the card's right edge. Fixed by measuring each line with the font
   it's actually drawn in.
2. **Review-popup crop looked blocky/low-resolution.** The crop is a small bounding box, always
   upscaled for display; the original code used `INTER_NEAREST` (kept pixels "exact" but reads
   as visible square blocks at review-card size). Switched to `INTER_CUBIC` upscaling + a light
   unsharp-mask pass to counter cubic's softening, and capped the display width at 640px so an
   unusually wide box doesn't blow up the whole window. Confirmed with a side-by-side render on
   a real `sample.mp4` frame — visibly smoother with no loss of detail.

### First quantitative evaluation, against user-provided ground truth

User watched `sample.mp4` directly and reported the true staff-visible windows: **12s-23s**
and **31s-50s**. Diffed against one full run's `output_ui_test/staff_detections.csv` (see
`KNOWLEDGE.md`, "Evaluation" for the full numbers and reasoning) — naive frame-level
precision/recall/F1 = 1.000 / 0.365 / 0.534. Two findings worth calling out specifically:

- **Tracking fragmentation, not the appearance-matching ceiling, is the main cause of missed
  frames** — recall is low (~31-40%) in *both* the plain-shirt and jacket windows, meaning even
  when color-matching works fine, ByteTrack losing/re-acquiring the person mid-walk splits one
  continuous walk into short fragments, some of which don't individually clear the
  length/motion thresholds. This wasn't previously measured or suspected to be this large a
  factor — the clothing-change ceiling (already documented) turned out not to be the dominant
  failure mode after all.
- **A genuine tracker ID-switch, caught by the user re-watching `annotated.mp4` frame by
  frame.** Inside the one flagged/human-confirmed possible-staff event (frames 826-1231,
  33.0s-49.2s, 9 track fragments confirmed as STAFF as one block via the Y/N/S review — see the
  "Interactive confirmation" entry above), a second person (dark shirt, a numbered badge, *not*
  the staff's tag) makes brief physical contact with the staff around 48.2s; the track handoff
  from fragment `#303` to fragment `#324` happens almost seamlessly at that exact moment, and
  `#324` (22 frames, 48.28s-49.24s) is that other person, not the staff, confirmed by the user
  watching the actual footage. Initially the user described this as "the model captured someone
  else" without the mechanism; asked directly whether the earlier Y-press on that event was a
  genuine judgment or just a UI-test click, they confirmed it was genuine at the time, then
  independently identified the *real* mechanism (contact-triggered ID switch, not a bad
  appearance match) once asked to look more closely. This is exactly the scenario the "possible
  staff" flagging design was meant to guard against, except one layer deeper than anticipated:
  the flagging is per *event* (a group of fragments merged by time proximity), so a human
  confirming "yes, this event is the staff" is currently a blanket yes over every fragment in
  it, not a per-fragment check — it can't catch a same-event, wrong-person ID switch that a
  full per-event review otherwise correctly approves. Documented as a real, measured limitation
  in `KNOWLEDGE.md`, with a possible fix noted (per-fragment confirmation, or a position/
  velocity continuity check between fragments before merging) but not yet implemented.

Recomputing precision/recall treating those 22 frames as wrong-identity rather than correct
("identity-aware," vs. the naive frame-presence-only version above) gives 0.920 / 0.336 / 0.492
— the gap between naive and identity-aware precision is the whole point: presence-only metrics
cannot see a right-time-window-but-wrong-person error at all; only a human checking the actual
(x, y)/identity, not just whether *someone* was flagged present, caught it. `KNOWLEDGE.md`'s
"Deliverables still needed" checklist item for this evaluation is now checked off, with a new
"Consider implementing a fix for tracking fragmentation... and/or per-fragment... review" item
added in its place.

### Implementing the two evaluation findings as real fixes

User asked to make sure everything from the evaluation above was properly recorded, then to
actually implement fixes for both findings rather than leaving them as documented limitations.

**Fix 1 — tracking fragmentation (`bridge_track_fragments()`, `interpolate_staff_gaps()`).**
Both apply the same "the person doesn't teleport" reasoning, gated by new constants
`BRIDGE_MAX_GAP_SECONDS = 1.0`, `BRIDGE_MAX_JUMP_PX = 120.0`, `BRIDGE_COLOR_FLOOR = 0.3` (kept
deliberately tighter than the possible-staff event-grouping constants, since bridging commits a
fragment straight into the confirmed staff result with no human review at all — see
`KNOWLEDGE.md`, "Calibration reference points"). `bridge_track_fragments()` runs right after
the initial per-track staff decision, before the possible-staff flagging step, and pulls in any
short/marginal fragment within that gap+jump of an already-confirmed staff fragment, iterating
to a fixed point so a chain of several short pieces gets pulled in together, not just the one
touching a confirmed fragment directly. `interpolate_staff_gaps()` then linearly fills any
*remaining* true zero-detection gap (no track at all for a few frames) between two staff
sightings, marking those rows `interpolated=True` in a new CSV column (schema documented in
`CLAUDE.md`) rather than leaving them as `staff_present=False`. Pass 2 draws these as a small
green dot + "STAFF (interpolated)" label, since there's no detected box to draw for them.

Measured effect on `sample.mp4`, in-window recall during the plain-shirt period (12s-23s, no
appearance-matching ambiguity to confound the comparison): 30.8% (baseline) -> 34.1% (with the
fix), 9 frames gap-filled entirely by interpolation — no fragments actually met the bridging
condition in that run. Real, but modest — a genuine improvement, not a full fix. The remaining
shortfall looks like it's driven by gaps *longer* than `BRIDGE_MAX_GAP_SECONDS`, not many short
fragments sitting right next to a confirmed one; worth re-measuring the real gap-length
distribution before just widening the constant.

**Fix 2 — tracker ID-switch inside a merged possible-staff event
(`split_event_by_position_jump()`, `split_event_by_color_outliers()`).** The color-outlier
splitter (added in the previous session, see "Clothing-change failure mode" above) was tested
first against the real 33.0s-49.2s reviewed event from `output_ui_test`, by re-running Pass 1
standalone (dumping `track_scores`/`track_coords`/`track_lab_colors` to a pickle for offline
analysis rather than re-running the full pipeline+GUI repeatedly) and replaying the exact
decision logic against it. Result: **it missed the real case.** Track `#303` (genuine staff, in
the dark jacket) and track `#324` (the black-shirt/badge man) had mean Lab colors `[16.5, 128,
130]` and `[12.6, 128, 130]` — a distance of ~4, far under `FRAGMENT_OUTLIER_LAB_DIST`'s 35 —
because both happen to be wearing very dark clothing. Color genuinely can't tell them apart
here. But their *positions* gave it away: `#303`'s last known point (555, 276) and `#324`'s
first point (586, 154), essentially the same frame (they briefly overlap, 1207-1208), are
~126px apart — against a real measured max walking speed of ~17px/frame, that's a physically
impossible jump, i.e. almost certainly a different physical person picked up by the tracker at
the exact moment of contact.

Added `split_event_by_position_jump()` (new constant `MAX_BRIDGE_SPEED_PX_PER_FRAME = 35.0`,
generous headroom above the real ~12-17px/frame max but well under a genuine ID-hop), run
*before* the color-outlier split specifically because it catches what color can't. First
implementation had a labeling bug: every sub-event coming out of any split got marked
`position_jump=True`, including the "before the jump" piece that wasn't actually anomalous —
caught by inspecting the offline replay's output directly (event 1, tracks [228, 240], was
being flagged even though nothing about it was suspicious). Fixed by marking only the group(s)
that start *after* a detected jump, not the first group in any split.

Re-validated twice: once against the offline pickle replay (confirmed track `#324` lands in its
own event, cleanly separated from the genuine staff fragments `#264, #279, #293, #294, #303`
around it, with none of the other events wrongly flagged), then once more with a full, real
`--skip-review` pipeline run end-to-end. Final printed output matched exactly:
```
[4] frames 1207-1231 (48.3s-49.2s), 1 track fragment(s) -> review_event_4.jpg  [needs manual review]
    (position jump from the previous fragment -- likely a different person mid-event, e.g. a
    tracker ID-switch during contact)
```
— isolating precisely the reported case, with `possible_staff_review.csv`'s new
`color_outlier`/`position_jump` columns confirming only event 4 is `position_jump=True` and
only event 2 (track `#250`, a separate, smaller color divergence, unrelated to the `#303`/`#324`
case) is `color_outlier=True`. This is a genuinely narrow, verified fix (one real case), not a
general solution — it still can't catch an ID-switch that's both same-colored *and*
spatially smooth (no contact/jump moment to detect), which would need a stronger signal than
either check alone.

Both fixes, their measured effect, and their remaining limits are now documented in
`KNOWLEDGE.md` ("Current approach," "Calibration reference points," "Known limitations,"
"Deliverables still needed") and `CLAUDE.md` (architecture steps 4-5, output schema). Test
output folders (`output_fix_test`) were deleted after validation — not part of the deliverable,
their findings are fully captured here and in `KNOWLEDGE.md`. Note: `output/` (the existing
canonical deliverable folder) predates both fixes and should be regenerated before the write-up
cites its numbers as final.

### Canonical re-run, a wrong Y-press caught before it reached the numbers, and the jitter check

User re-ran the guided setup from scratch (fresh reference crop, `output/` regenerated) to try
the two new fixes live. `possible_staff_review.csv` showed all 4 events confirmed STAFF —
including event 4, the one the position-jump fix specifically flagged with a "tracker
ID-switch during contact" note. That's the black-shirt/badge person from the earlier
evaluation, not the staff. Flagged this to the user before trusting the run; they confirmed
the Y-press was a mistake — the review popup's crop was too tight to see the person's face, so
they judged by shirt color, which is exactly the confounded signal here (`#303`'s and `#324`'s
colors are nearly identical, per the position-jump investigation above). This is a real,
concrete instance of the exact scenario the human-in-the-loop design was meant to be honest
about: a correctly-flagged, correctly-explained case can still get a wrong human answer under
information constraints. Noted as a review-UI finding too (the popup should show more context —
a wider crop or short clip — when the decision hinges on identity rather than clothing color),
not something to fix by just trusting humans more.

Fixed without a full multi-minute re-run: directly patched `output/staff_detections.csv`
(flipped track `#324`'s 22 rows to not-present) and `possible_staff_review.csv` (event 4's
resolution to "confirmed not staff"). Left `output/annotated.mp4` un-patched — it still shows
green on the black-shirt person for that ~1s stretch, a known cosmetic inconsistency, not
worth a full re-render for a data correction. Re-ran the same precision/recall/F1 diff against
the user's ground truth on the corrected data: **1.000 / 0.448 / 0.619** — recall up from the
pre-fix 0.365 with precision still perfect. Documented in `KNOWLEDGE.md`, "Evaluation," under
"Re-measured after implementing the fragment-bridging and position-jump fixes"; that
deliverables-checklist item is now checked off.

Also finished the other open deliverable, visually verifying the Savitzky-Golay smoothing's
effect on jitter: re-ran Pass 1 standalone (same approach as the earlier diagnostic dumps) to
get raw, pre-smoothing per-frame footpoints, then re-ran the pipeline's exact smoothing code
(same `k` selection logic) offline against two real staff tracks (`#108`, 27 frames; `#153`, 18
frames) and plotted raw vs. smoothed x(t)/y(t) with matplotlib (available in `.venv`, not
previously used or documented as a dependency -- worth noting if it needs installing fresh
elsewhere). Result: the effect is real but subtle. Visually, small single-frame wiggles (e.g. a
notch in track `#108`'s x around frames 457-461, a slight dip in its y around 468-470) get
smoothed away while the line still tracks the real walking motion/turns tightly -- no visible
lag or overshoot, which is the main failure mode worth checking for in a rolling-window filter.
Track `#153`'s segment happened to already be naturally smooth, so raw and smoothed nearly
overlap there -- also a good sign (the filter isn't distorting already-clean data). The
frame-to-frame mean-abs-delta metric only drops ~1-2%, but that number is dominated by real
motion speed, not noise, so it isn't the right way to quantify this -- the visual check is what
actually answers the question. `k` (the smoothing window) works out to a fixed 5 frames (0.2s)
for any track of decent length, given the current `min(5, ...)` cap in the code -- a light,
gentle smooth, not an aggressive one, consistent with what was observed. `KNOWLEDGE.md`'s
deliverables checklist item for this is now checked off.

### Fixing the review popup itself, in response to the wrong Y-press

User asked directly for ideas to fix the crop-too-tight/judged-by-color problem, suggesting a
bigger picture with the person highlighted. Agreed and extended it: also show two frames per
event (not one -- a single frame can be an unlucky one, like the near-solid-black silhouette
seen earlier), and surface *why* an event was flagged directly in the popup, not just in the
CSV, since the whole point is to point the reviewer's attention at identity rather than color
when that's specifically what's confounded.

Implemented: `padded_highlight_crop()` (new helper, `src/staff_id.py`) crops a generously
padded region around the person (`REVIEW_CROP_PAD_FRAC = 1.2`, i.e. 120% extra margin on each
side) instead of the tight detector box, with a bright gold rectangle drawn around the actual
person so it's unambiguous who's being asked about despite the wider view. `run()` now samples
two frames per flagged event (the first and last frame the representative track appears in,
falling back to one if the fragment is a single frame) instead of one, and `hstack_crops()`
composites them side by side for the saved `review_event_*.jpg` audit file. `confirm_event_interactive()`
was reworked to accept a list of 1-2 crops (laid out side by side, each in its own glass photo
frame with a timestamp caption) and an optional `flag_note` string, rendered as a visible
terracotta warning line under the header when the event was split out for `position_jump` or
`color_outlier` reasons.

Verified by rendering the popup offline (monkeypatching `cv2.namedWindow`/`imshow`/
`setMouseCallback`/`waitKey`/`destroyWindow` to capture the canvas without a live display) using
real frames from `sample.mp4` and the actual event-4 flag text. Found and fixed one bug from
that render: the flag_note line overflowed the card's right edge -- the card width calculation
only accounted for the photo row's width, not the header text, the same font-width-measurement
bug pattern as the earlier instruction-banner overflow (see "UI restyle" above). Fixed by
measuring the header/flag text width (with the actual fonts they're drawn in) and including it
in the card-width calculation. Re-rendered to confirm the fix.

Documented in `CLAUDE.md` (architecture step 6, output schema) and `KNOWLEDGE.md`'s
"Re-measured after implementing the fragment-bridging and position-jump fixes" note (updated
from "not something to fix by just trusting humans more" to record that it *was* addressed,
directly, in the UI itself).

### Simultaneous-staff conflict resolution -- a new false positive found in live testing

User ran the pipeline again and spotted a real, substantial false positive in a screenshot:
track `#107` labeled "STAFF #107 0.49" at 0:22, next to the genuine staff (`#153`, "STAFF #153
0.93") in the same frame -- both green, both "STAFF," at the same time. Investigated directly
against `output/staff_detections.csv`: `#107` is no brief flicker -- 201 frames, ~8 seconds
(17.8s-25.8s), median score 0.552 (just over the 0.5 `--staff-threshold` default), all 18 of
`#153`'s frames fall inside its span. Not a bug in the recent fixes -- this is the
already-documented "more than one similarly light-clothed person" limitation, just the first
time it showed up as a long, clean, automatically-auto-qualified false positive rather than a
brief or already-flagged one.

The new, previously-missing check: the task brief states there's exactly one tagged staff
member, so two tracks both auto-qualifying *at the same time* means at most one is right --
a constraint the per-track threshold decision never enforced, since each track was judged
independently. Implemented `resolve_simultaneous_staff_conflicts()` (`src/staff_id.py`): finds
groups of temporally-overlapping auto-qualified tracks, keeps only the single highest-median-score
one, and routes the rest back into the possible-staff review pipeline (not a silent drop) --
consistent with the existing human-in-the-loop design elsewhere in the pipeline. Wired in right
after fragment-bridging, before the per-track breakdown printout, so a demoted track shows a
clear "overlaps a stronger staff track -- needs review" verdict instead of silently
disappearing from the table.

Verified directly against the real case (not a synthetic one): reconstructed `track_scores`/
`track_coords` from `output/staff_detections.csv` and ran the new function against the real
`#107`/`#153` data -- confirmed `#107` gets demoted, `#153` is kept. Documented in `CLAUDE.md`
(new architecture step 5, everything after it renumbered), `KNOWLEDGE.md` ("Current approach,"
"Known limitations" -- noted honestly that this only catches the *simultaneous* shape of the
same-color-different-person problem, not a similarly-dressed person walking through alone at a
different time, which would still be silently auto-labeled).

### Popup too small, and 15 events to review -- both from the same live test run

User reported two problems from one screenshot: the review popup ("Staff Review -- 3 of 15")
was too small to read comfortably, and 15 events needing review was a lot more than any
previous run.

**Popup size:** increased `_prep_review_photo()`'s target photo height/width (260/420 ->
420/620), scaled up all margins/fonts/button sizes in `confirm_event_interactive()`
accordingly, and added an explicit `cv2.resizeWindow(win, canvas.shape[1], canvas.shape[0])`
right after `cv2.namedWindow()` -- `WINDOW_NORMAL`'s initial size isn't guaranteed to match the
shown image 1:1 on every Windows/DPI setup, and this forces it to. Verified by rendering the
popup offline (same monkeypatch-cv2 technique as the first popup redesign) -- canvas grew from
~826x580 to ~1103x810, all text legible, no overflow.

**15 events, root-caused:** inspecting `output/possible_staff_review.csv` showed 13 of 15
events flagged `position_jump=True`, including several that split apart what should have been
one continuous jacket-period walk into 6 separate "confirmed STAFF" events. Root cause:
`split_event_by_position_jump()` computed `implied_speed = jump / max(1, gap)`. Whenever two
fragments *overlap* in frame numbers (`gap <= 0` -- common even for the *same* person, when
ByteTrack briefly double-emits during an ordinary handoff, not just at a genuine two-person
contact event), the forced denominator of 1 turned any nonzero spatial distance into a huge
"implied speed," misfiring far more often than intended. It only ever worked correctly for the
one real ID-switch case by coincidence (that case's overlap and its large ~126px distance both
happened to point the same way).

Fixed by giving overlapping fragments their own rule, instead of forcing them through the
speed formula: "implied speed" isn't a coherent idea for zero/negative elapsed time, so for
`gap <= 0` the check now asks directly whether the fragments are far apart in space
(`> BRIDGE_MAX_JUMP_PX`) while genuinely coexisting -- direct proof of two different people,
independent of any speed calculation; a small distance while overlapping is more likely the
tracker briefly double-emitting the same person. Non-overlapping fragments (`gap > 0`) keep the
original speed-based check unchanged.

Separately, also addressed the review-queue load directly: `resolve_simultaneous_staff_conflicts()`
was, until now, routing every demoted (conflict-loser) track into the same interactive
possible-staff queue as genuinely ambiguous color/motion cases. Reasoned that these are
epistemically different -- a demoted track already has strong, objective evidence against it (a
higher-scoring track was simultaneously active), unlike a jacket-period fragment where color
genuinely can't decide. Changed `resolve_simultaneous_staff_conflicts()` to return
`{demoted_tid: winner_tid}` (was a plain set) and stopped adding demoted tracks to
`walking_but_unmatched`; they're now auto-rejected outright and logged to a new
`auto_rejected_conflicts.csv` (track, conflicting track, both median scores, time range) instead
of consuming a review popup.

Verified both fixes together with a full run (`output_fix_verify/`, same calibrated reference
used throughout earlier testing, `--skip-review`): exactly **4** review events (matching the
known-good baseline from before the 15-event run -- events 1-4 with the same
color_outlier/position_jump flags as previously validated, no regression), and track `#89`
(median score 0.67) correctly auto-rejected as conflicting with `#88` (median score 0.74,
overlapping 14.72s-15.08s) via `auto_rejected_conflicts.csv`, with zero impact on the review
count. Deleted `output_fix_verify/` after validation. Documented in `CLAUDE.md` (architecture
step 5 and 6, output list) and `KNOWLEDGE.md` ("Current approach," "Known limitations,"
"Evaluation" section's review-popup note).

### Four more false positives reported -- traced to which "outfit era" got picked as reference

User reported `#107`, `#286`, `#208`, `#292` all wrongly shown as STAFF in `output/annotated.mp4`.
First checked whether `output/` was even a fresh run: its file timestamps matched the earlier
15-events episode exactly, i.e. this was the *pre-fix* run, not a fresh test of the current
code -- worth remembering that a run this messy could partly reflect reviewer fatigue from 15
popups (a real risk of flooding a human reviewer, not just a UX annoyance) rather than a new
regression.

Checked anyway, since `#107`/`#208`/`#292` all appear in `staff_detections.csv` with real
scores: `#107` median 0.858, `#208` median 0.703, `#292` median 0.702 -- all comfortably above
`--staff-threshold`. Compared against the run's own confirmed-genuine staff (`#88`, `#144`,
`#153`, from the well-established plain-shirt period): their medians were *lower* (0.39-0.43)
than all three reported false positives. This rules out a simple threshold fix (any threshold
excluding the false positives would also exclude the genuine staff).

Root cause found by inspecting `output/reference_crop.jpg` directly: it's a near-solid dark
silhouette -- the staff's **dark jacket**, not the light-blue/white shirt used in every
previous calibration. With a dark reference, the real shirt-period staff frames correctly read
as a mismatch (light blue vs. dark -- score low), while `#107`/`#208`/`#292` happen to also
wear dark clothing and score deceptively high by coincidence. Not a new failure mode -- the
same documented clothing-change limitation, but from the other direction: *which* era gets
picked as reference matters a lot, and a common/generic color (dark) is far more confusable
with random bystanders than the shirt's more distinctive light blue.

Fixed two ways: (1) added a `Tip:` block to `select_reference_interactive()`'s on-screen STEP 3
instructions, telling whoever runs this live to prefer the most visually distinctive
appearance available (not black/gray/navy) when a person's outfit changes; (2) documented the
concrete before/after scores in `KNOWLEDGE.md`'s "Assumptions" #7, so it's an explicit,
evidenced warning rather than an implicit gotcha. `#286` couldn't be found anywhere in this
run's `staff_detections.csv` at all -- asked the user for a timestamp to pin down what it
actually refers to (possibly a misread of a yellow "REVIEW?" label, or from a different
viewing).

# FootfallCam AI Evaluation — Staff Identification Solution Plan

**Prepared for:** AI Engineer (task owner)
**Scope:** Technical concepts and pipeline design for the staff identification and localization challenge

---

## 1. Task Recap

- **Input:** `sample.mp4`, a top-down/high-angle 3D sensor video of a corridor.
- **Staff identifier:** a person wearing a name tag, viewed from above. (The task PDF illustrates this with a green polo/white-logo photo, but that's a generic example, not a frame from `sample.mp4` — the actual tagged person in the video wears a light blue/white top; see `LOG.md`, "Testing 1.")
- **Task 1:** Identify which frames contain the staff member.
- **Task 2 (bonus):** Locate the staff member's (x, y) coordinates in frames where present.
- **Constraints:** must be demoable live, on a laptop, against an unseen test video.

---

## 2. Necessary Technical Concepts

### 2.1 Overhead Perspective & Domain Shift
- The sensor is a **top-down/high-angle camera**, not eye-level CCTV.
- Standard detectors (YOLO, Faster R-CNN) are trained on eye-level datasets like COCO — heads/shoulders dominate the frame, legs are foreshortened, faces are rarely visible. This is a **domain shift** problem; pretrained weights should be validated (and possibly fine-tuned) on this viewpoint rather than assumed to work out of the box.
- **Re-validated on this footage (resolved):** an earlier attempt (`LOG.md`, "Testing 1") tested stock YOLOv8n/m on a single frame (402) of `sample.mp4` and measured only ~0.18–0.41 confidence on the walker, on par with seated people, and concluded YOLO wasn't viable as the primary detector. Re-tested more rigorously against 65 frames spanning the confirmed walker window (380–641), using the walker's own tracked (x,y) from round 1 as ground truth: **YOLOv8n placed a person-box within 80px of the true walker position in 65/65 frames (100%)**, and that box was the single highest-confidence detection in the frame in 46/65 (71%, YOLOv8s: 54/65, 83%). Frame 402 in isolation was an outlier — the walker happens to be caught mid-turn there, mostly foreshortened head/shoulders — not representative of the clip. **Conclusion: YOLO is viable as the primary person detector for this footage after all**, provided the pipeline doesn't rely on "highest-confidence box = the walker" (round 1's implicit assumption, which the 402 test technically falsified) but instead detects *all* person boxes at a permissive threshold and disambiguates which one is staff via the separate appearance-matching stage (2.3) — which is what the cascaded design in 2.2 already calls for. See 2.4 for how this changes the detector decision.
- "3D sensor" often implies stereo/ToF depth capability (used elsewhere in FootfallCam's product for people counting) — a potential bonus signal even though this task only supplies RGB.

### 2.2 Two-Stage Cascaded Detection Pipeline
Rather than searching the whole frame for a small visual cue, the standard paradigm is:
1. Detect the **person** bounding box (robust, well-supported by existing detectors).
2. Crop the **upper-torso/chest ROI** from that box.
3. Classify/detect the tag **within that cropped patch**.

This cascade improves signal-to-noise versus a single end-to-end detector searching a full 1080p frame for a small feature.

**ROI extraction must be mask-aware, not a raw rectangle.** The earlier attempt's dominant failure mode wasn't the color/matching formula — it was loose detector/motion boxes that include adjacent desk or floor pixels alongside the person, which dilutes any color statistic computed over the whole box (root-caused and documented in `LOG.md`, "Testing 1": a box padded with background pulled the measured torso lightness far enough toward a light reference to produce a false match). Whichever box source is used (YOLO or motion contour), the torso ROI fed to tag classification should be restricted to actual foreground/person pixels within that box (e.g. a foreground mask from background subtraction, or a segmentation mask), not the full bounding rectangle.

### 2.3 Tag Identification Methods (increasing sophistication)
- ~~**Color-space heuristics (HSV):**~~ already tried and rejected on this footage — H(+S, +V) histograms could not separate a black hoodie from the reference's light shirt, because both a dark garment and the beige floor background are low-saturation, and a torso crop is background-dominated (see `LOG.md`, "Testing 1"). Not worth re-attempting as-is.
- **Lab color-space matching (primary, resolved):** mean CIE-Lab distance over a mask-restricted torso crop, weight 0.85 in the final blended score. Re-confirmed in round 2 on real tracked data: true-match tracks measured Lab distance ~21-27 vs. ~60-98 for every other track — a clean, well-separated signal (see `LOG.md`, "Round 2").
- **Classical feature/template matching (SIFT/ORB + homography):** match the cropped reference patch against each frame's torso crop. Minor/secondary signal at best at this resolution — used this way in the earlier attempt. Not carried into round 2's implementation (Lab alone already separated cleanly; added complexity wasn't justified).
- **Trained classifier on torso crops:** small CNN or fine-tuned MobileNet/ResNet (staff vs. not-staff), trained on labeled crops from the "few hundred video samples" the brief mentions. **Not available for this task** — confirmed with the user that `sample.mp4` is the only video on hand, so there's no dataset to train or fine-tune anything on. Not attempted, both for that reason and because Lab distance alone proved sufficient.
- **CLIP few-shot cosine similarity (tried, demoted to optional/off by default — resolved):** the plan originally prioritized this as the primary matcher on the theory that a learned embedding would generalize better than hand-crafted Lab distance. **Empirically tested in round 2 and found actively counterproductive**: on the same tracked data where Lab distance cleanly separated the true match (21-27) from everyone else (60-98), CLIP (`openai/clip-vit-base-patch32`) cosine similarity was *anti-correlated* with the true match — the true-match tracks scored *lowest* (0.69-0.71) and the clearly-wrong tracks scored *highest* (0.78-0.79). Most likely cause: ViT-B/32 was pretrained on natural eye-level photos and doesn't transfer its clothing-color discrimination to these small, blurry, foreshortened top-down crops — the same domain-shift problem that originally hurt YOLO detection (2.1), but here it degrades embedding quality rather than localization. Kept in the implementation as an opt-in flag (`--use-clip`, off by default) since it's a reasonable thing to have tried and worth being able to discuss, but it is not the primary matcher. See `LOG.md`, "Round 2" for the full numbers.

### 2.4 Multi-Object Tracking (MOT) & Temporal Smoothing
- **Detector choice (resolved by 2.1's re-validation): YOLOv8 (n or s) as the primary person detector.** Run at a permissive confidence threshold (~0.1–0.25, not the default 0.25+NMS-heavy config) so it emits candidate boxes for everyone in frame, including partial/foreshortened poses; let the appearance-matching stage (2.3), not detection confidence, decide who's staff. Keep MOG2 motion detection as a cheap secondary signal — e.g. to help disambiguate when two YOLO boxes overlap heavily, or as an entirely offline fallback if `ultralytics`/weights aren't available on the demo machine — rather than the primary detector, now that YOLO itself is validated as reliable here.
- **Tracking-by-detection** (ByteTrack, DeepSORT, BoT-SORT — built on Kalman filtering + Hungarian matching) links detections into consistent per-person tracklets across frames, so decisions aren't re-made from scratch every frame with no memory. This matching logic (Kalman + Hungarian) is worth adopting regardless of which box source above is chosen — it should replace the earlier attempt's simple greedy nearest-neighbour centroid tracker, whose ID swaps when people cross paths were a real problem there.
- **Track-level temporal voting:** aggregate confidence/majority vote across a track's full lifespan to eliminate single-frame flicker (e.g., a glare that briefly looks tag-like). Note the earlier attempt found *track* identity itself unreliable enough that it made per-frame (not per-track) decisions in the end — if a better tracker (above) fixes ID stability, per-track voting becomes viable again and should be preferred since it's more robust than pure per-frame scoring.
- **Trajectory interpolation:** preserves identity through brief occlusion (e.g., staff member turns, tag hidden for several frames).
- **Walking-motion gate (added in round 2, resolved).** Unlike round 1's motion-only detector, YOLO detects people whether or not they're moving — running the full pipeline surfaced more than one person in this office wearing a similarly light-colored top, some seated at desks. Appearance matching (Lab distance) alone cannot disambiguate two people dressed alike sitting near each other; only the one clearly-confirmed walking event is visually unambiguous. Decision (user-directed): **restrict "staff present" to confirmed walking events** — a color-qualifying track only counts as staff if its average centroid speed is >=6px/frame *and* its total positional range is >=40px (both computed directly from the track's own tracked coordinates, no separate motion detector needed). Calibrated against real tracked data: genuine walking tracks measured 12-17px/frame speed with 96-252px range; the seated/desk cluster measured 0.6-4.2px/frame with 5-91px range — clean separation. See `LOG.md`, "Round 2".

### 2.5 Spatial Localization (Bonus Task)
- Represent position as either the bounding-box **center** or the **bottom-center/footpoint** (ground-plane contact) — the latter is the standard convention in retail footfall analytics.
- Apply light trajectory smoothing (rolling average, Savitzky-Golay) to reduce bounding-box jitter in the (x, y) stream.
- Stretch goal: convert pixel coordinates to real-world floor coordinates via camera calibration/homography, matching FootfallCam's actual product output.

### 2.6 Evaluation & Metrics (done)
- **Precision / Recall / F1** for "staff present in frame" classification — accuracy alone is misleading given class imbalance (one staff track vs. many non-staff people). Done against user-reported ground truth (no labeled dataset existed to compute this against otherwise): naive frame-presence P/R/F1 = 1.000/0.365/0.534; an "identity-aware" variant (see below) = 0.920/0.336/0.492. Full numbers, methodology, and caveats in `KNOWLEDGE.md`, "Evaluation."
- **IoU** for the bonus localization output was not computed as a number — no pixel-level ground-truth boxes exist to score against. Positional accuracy was instead checked qualitatively (does the tracked point stay on the right person) by the user re-watching `annotated.mp4`, which is how a real tracker ID-switch (see below) was actually caught — a check IoU-against-presence-only ground truth wouldn't have surfaced either.
- Evaluation surfaced two concrete findings beyond the headline numbers, both fed back into the pipeline (see `KNOWLEDGE.md`, "Known limitations" and "Current approach"): **tracking fragmentation** (not the clothing-change ceiling) is the dominant cause of missed frames, addressed with fragment-bridging + gap-interpolation; and a **tracker ID-switch** during brief physical contact between two people can misattribute a few frames within an otherwise-correct human-confirmed event, partially addressed with a per-fragment color-outlier split (partial: it's still a color-based check, so it doesn't catch two people who happen to be dressed similarly).

### 2.7 Deployment Constraints (Live Demo Readiness)
The code will be run live on a new, unseen test video, so this isn't purely an offline notebook exercise:
- Favor **lightweight models** (YOLOv8n/s, MobileNet backbones) over heavy multi-stage pipelines that choke on CPU.
- Consider **frame subsampling** (process every 2nd–3rd frame, interpolate the rest) for real-time throughput.
- Pipeline must **generalize** — not be hardcoded to quirks of `sample.mp4`.

### 2.8 Interactive Reference Calibration (Live Demo Workflow)
Running against an *unseen* test video live means there's no pre-picked reference crop available ahead of time — the workflow this plan targets is:

1. **User points the script at the new video** (a CLI arg, e.g. `--video new_test.mp4`) — a plain script, not a notebook, to avoid kernel/state fragility during a live demo.
2. **User selects one reference box interactively** — an OpenCV GUI (`cv2.selectROI` plus a frame-scrub trackbar) opens on the video: scrub to a frame where the staff member is clearly visible, drag a box around them, press Enter. This replaces hand-typed `--ref-frame`/`--ref-box` coordinates (error-prone to get right live) while still keeping those flags available as optional overrides for non-interactive/scripted runs.
3. **The pipeline runs automatically** from there — detection, tracking, mask-aware ROI extraction, matching, temporal smoothing, output writing — with no further manual input.

**What stays fixed vs. what's re-supplied per video** (important to be able to explain clearly, since it's easy to conflate the two): the pretrained model weights (YOLO / CLIP) and all pipeline logic are identical across every video — nothing is retrained or fine-tuned per video. Only two things change per video: (a) the one reference crop from step 2, since "staff appearance" is this-person's-actual-clothing, not a universal category the model can infer on its own — the tag itself isn't resolvable as legible text/logo at this resolution (see 2.3), so a one-shot appearance reference is the honest substitute; and (b) the background model MOG2 (if used, per 2.4's conditional) builds fresh per video automatically, no manual step needed. One caveat to flag rather than gloss over: a few numeric constants (e.g. staff-presence threshold, hysteresis bounds) were tuned by eye against `sample.mp4`'s score distribution in the earlier attempt and aren't guaranteed to transfer perfectly to a new video's lighting/scale — worth a quick sanity check against the new video's score distribution rather than assuming zero-shot transfer, though switching to CLIP cosine similarity (2.3) should make this less brittle than the earlier attempt's raw Lab-distance scale.

---

## 3. Unified Pipeline — Detection, Identification & Localization

Task 1 and Task 2 (bonus) share the same detect → track → classify backbone, so they are implemented as a single pipeline that emits one CSV. Localization (Task 2) simply adds coordinate extraction and smoothing once a track is confirmed as staff.

```
Input: video (--video path/to/video.mp4)
        |
        v
Interactive reference selection  [see 2.8]
Scrub to a frame, drag a box around the staff member (cv2.selectROI);
one-time per video, --ref-frame/--ref-box remain as optional overrides
        |
        v
Detect + segment people (single pass)
YOLOv8-seg (n/s), permissive conf threshold — validated reliable on this
footage (2.1); gives boxes + instance masks together
        |
        v
Track people
ByteTrack (Kalman + Hungarian matching, via ultralytics' built-in tracker),
replaces round 1's greedy centroid tracker
        |
        v
ROI extraction (mask-aware)
Crop torso region per track, non-mask pixels replaced with neutral
background — not the raw rectangle (see 2.2)
        |
        v
Classify tag
Lab color-distance (primary, 0.85 weight) [+ CLIP, off by default — see 2.3]
        |
        v
Track-level decision
Median score >= threshold AND walking-motion gate (speed + range from the
track's own coordinates) — restricts to confirmed walking events (2.4)
        |
        v
Extract & smooth coordinates  [Task 2 / bonus]
Bbox center or footpoint, rolling avg / Savitzky-Golay
        |
        v
Model evaluation
Precision, recall, F1 vs labeled frames; IoU for localization
        |
        v
Output: staff_detections.csv
```

### 3.1 Output schema (`staff_detections.csv`)

One row per **(frame, staff detection)** rather than one row per frame. This is deliberate: a single-row-per-frame schema (with fixed `x`/`y`/`match_score` columns) can only hold one staff member's data per frame and silently breaks if more than one staff member appears in the same frame. The row-per-detection schema scales to any number of simultaneous staff without a format change.

| Column | Description |
|---|---|
| `frame` | Frame index |
| `timestamp_s` | Time in seconds |
| `staff_present` | `True`/`False` — whether this row represents a staff detection |
| `track_id` | ID of the tracked person; required once multiple staff are possible, so rows in the same frame (and across frames) can be told apart and linked into a trajectory. Use a fallback like `unknown_1` if the tracker temporarily loses ID continuity rather than leaving it blank. |
| `x`, `y` | Bbox center or footpoint coordinates (blank if no staff in that row) |
| `match_score` | Tag-match confidence (blank if no staff in that row) |

**Rules:**
- Every processed frame gets **at least one row**. If no staff is detected, emit a single row with `staff_present=False` and blank `track_id`/`x`/`y`/`match_score`.
- If **N staff members** are detected in a frame, emit **N rows** for that frame, each with a distinct `track_id`.

**Example — frame with no staff, then a frame with two staff detected:**

```csv
frame,timestamp_s,staff_present,track_id,x,y,match_score
0,0.00,False,,,,
1,0.04,False,,,,
142,5.68,True,3,512,780,0.91
142,5.68,True,7,890,340,0.88
143,5.72,True,3,515,782,0.90
143,5.72,True,7,895,338,0.87
```

**Downstream use:** group rows by `track_id` to reconstruct each staff member's trajectory; group by `frame` to answer "how many staff were present at time X" — consistent with how footfall analytics are consumed in the actual product.

---

## 4. Deployment Note

Deployment constraints (lightweight model choice, optional frame subsampling, batched OpenCV I/O) apply across every stage above, not as a separate step — the whole pipeline needs to run live on a laptop against an unseen test video during the interview, and must generalize rather than being hardcoded to `sample.mp4`. Concretely, this means a plain CLI script (not a notebook) with the interactive reference-calibration step described in 2.8, since that's the one part of the workflow that has to happen live and can't be pre-baked.

---

## 5. Deliverables Checklist (per task instructions)

- [x] 1–2 page documentation highlighting key solution parts — `DOCUMENTATION_FINAL.md` / `.docx`
- [x] Code script / repository link — `src/staff_id.py`, pushed to GitHub (see `README.md`)
- [x] Annotated output video or other visualization aid — `annotated.mp4`, produced by every run
- [ ] Ability to run live on a new, unseen test video during the interview — the guided/interactive
      flow (2.8) is implemented and used repeatedly on `sample.mp4`, but not yet validated against
      a genuinely different second video (see `KNOWLEDGE.md`, "Deliverables still needed")

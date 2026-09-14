"""
staff_id.py — staff identification pipeline (formerly staff_id_v2.py; renamed after an
earlier motion-detection-based attempt at this file name was deleted, once this pipeline
fully superseded it).

Design rationale: dev_notes/SOLUTION_PLAN.md. Full history, including the earlier attempt and why it
was replaced: dev_notes/LOG.md ("Testing 1" / "Round 2" entries).

Pipeline: YOLOv8-seg (person detection + instance masks) -> ByteTrack (ID-stable tracking,
via ultralytics' built-in tracker) -> mask-aware torso ROI extraction -> CLIP cosine
similarity to an interactively-selected reference crop (+ a minor Lab color-distance term) ->
track-level majority vote for the staff/not-staff decision -> row-per-detection CSV +
annotated video.

Usage:
    # Guided (recommended): asks which video and where to save results, then walks through
    # picking the staff reference on-screen. No flags needed.
    python src/staff_id.py

    # Scripted / repeated runs, skipping all prompts:
    python src/staff_id.py --video sample.mp4 --output-dir output
    python src/staff_id.py --video sample.mp4 --ref-frame 402 --ref-box 476,488,106,173
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe: this pipeline only ever saves plots to a file
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS = 0
NEUTRAL_BG = (128, 128, 128)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Pass-2 annotated-video box colors (BGR, since that's what cv2 draws in). Named here rather
# than left as inline tuples in the render loop, so the legend printed at the end of a run
# ("green=staff, yellow=still needs review, orange=other") is easy to check against the actual
# values -- a real bug was once caused by an inline BGR tuple that was accidentally the RGB
# order instead (see dev_notes/LOG.md, "Round 2"), which a named constant makes harder to repeat.
STAFF_BOX_COLOR = (0, 200, 0)       # green
REVIEW_BOX_COLOR = (0, 210, 255)    # yellow
OTHER_BOX_COLOR = (0, 140, 255)     # orange
LAB_DIST_SCALE = 80.0
# Lab color distance is the primary signal, not CLIP. Measured on this footage (see dev_notes/LOG.md,
# "Round 2"): CLIP similarity was *anti-correlated* with the true match (true-match tracks
# scored 0.69-0.71, clearly-wrong tracks scored 0.78-0.79) -- ViT-B/32, pretrained on natural
# eye-level photos, doesn't transfer its clothing-color discrimination to these small, blurry,
# top-down fisheye crops. Lab distance (round 1's proven metric) cleanly separated the same
# tracks (21-27 true match vs. 60-98 everyone else). CLIP is kept as an optional, off-by-default
# signal (--use-clip) since it's a reasonable thing to have tried and worth being able to show.
CLIP_WEIGHT = 0.15
LAB_WEIGHT = 0.85


def format_duration(seconds):
    """e.g. 192 -> '3m 12s', 45 -> '45s'."""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _wait_for_retry(path):
    print(f"\n  Could not write to {path} -- it's probably still open in another program "
          f"(Excel, a photo/video viewer, etc). Close it there, then press Enter here to "
          f"try again (Ctrl+C to give up).")
    input()


def open_for_write_retrying(path, **kwargs):
    """open(path, 'w', ...) that retries on a locked file instead of crashing and losing a
    multi-minute processing run (hit in practice: user had the previous run's CSV open in
    Excel -- see dev_notes/LOG.md, 'Round 2')."""
    while True:
        try:
            return open(path, "w", **kwargs)
        except PermissionError:
            _wait_for_retry(path)


def imwrite_retrying(path, img):
    while True:
        if cv2.imwrite(str(path), img):
            return
        _wait_for_retry(path)


def open_video_writer_retrying(path, fourcc, fps, size):
    while True:
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer
        _wait_for_retry(path)


# --------------------------------------------------------------------------- #
# Guided setup (asks for the video + output folder, no CLI flags needed)
# --------------------------------------------------------------------------- #
def guided_setup():
    cwd = Path.cwd()
    print()
    print("=" * 64)
    print("  FootfallCam Staff Identification -- Setup")
    print("=" * 64)

    print()
    print("STEP 1: Choose the video to analyze")
    print("-" * 64)
    found = sorted(p.name for p in cwd.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if found:
        print(f"Found these video files already in this folder ({cwd}):")
        for i, name in enumerate(found, 1):
            print(f"  [{i}] {name}")
        print()
        print("If the video you want isn't listed, copy/move it into this folder")
        print("now, then answer below (type its file name).")
    else:
        print(f"No video files found in this folder yet:")
        print(f"  {cwd}")
        print("Copy/move your video file into that folder now, then answer below.")
    print()

    video_path = None
    while video_path is None:
        choice = input("Enter a number from the list above, or type the video's file name: ").strip()
        if not choice:
            continue
        if choice.isdigit() and found and 1 <= int(choice) <= len(found):
            video_path = cwd / found[int(choice) - 1]
        else:
            candidate = Path(choice)
            if not candidate.is_absolute():
                candidate = cwd / choice
            if candidate.exists():
                video_path = candidate
            else:
                print(f"  Couldn't find '{choice}' in {cwd}. Check the spelling, "
                      f"make sure the file's been copied in, and try again.")
    print(f"-> Using video: {video_path.name}")

    print()
    print("STEP 2: Choose where to save the results")
    print("-" * 64)
    default_out = "output"
    out_name = input(f"Enter a name for the output folder [press Enter for '{default_out}']: ").strip()
    if not out_name:
        out_name = default_out
    print(f"-> Results will be saved to: {cwd / out_name}")
    return str(video_path), out_name


# --------------------------------------------------------------------------- #
# UI styling -- Japandi (warm neutrals, muted natural tones, generous
# whitespace, thin accent lines) + glassmorphism (frosted translucent panels,
# soft shadows, rounded corners) for every interactive window. There's no UI
# toolkit here, just cv2 drawing primitives, so "glass" is simulated: blur
# whatever is behind a panel, tint it, and composite through a rounded-corner
# mask -- see draw_glass_panel().
# --------------------------------------------------------------------------- #
INK = (51, 56, 58)          # warm near-black -- primary text
INK_SOFT = (108, 112, 110)  # muted warm gray -- secondary/caption text
CREAM = (228, 237, 244)     # warm off-white -- panel fill / light text on color
SAGE = (110, 142, 122)      # muted green -- affirmative (staff / yes)
TERRACOTTA = (93, 124, 201)  # muted warm red -- negative (no)
SAND = (150, 168, 181)      # muted neutral -- secondary (skip)
GOLD_LINE = (116, 164, 196)  # thin warm-gold accent border


def _rounded_mask(h, w, radius):
    """White-on-black rounded-rectangle mask, h x w, for compositing a panel
    with rounded corners via alpha blending."""
    r = max(0, min(radius, h // 2, w // 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    if r == 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _rounded_rect_pts(x, y, w, h, r, n=8):
    """Polyline points tracing a rounded rectangle outline, for cv2.polylines."""
    r = max(0, min(r, h // 2, w // 2))
    pts = []
    corners = [
        (x + w - r, y + r, 270, 360),
        (x + w - r, y + h - r, 0, 90),
        (x + r, y + h - r, 90, 180),
        (x + r, y + r, 180, 270),
    ]
    for cx, cy, a0, a1 in corners:
        for t in np.linspace(a0, a1, n):
            rad = np.deg2rad(t)
            pts.append((int(cx + r * np.cos(rad)), int(cy + r * np.sin(rad))))
    return np.array(pts, dtype=np.int32)


def draw_glass_panel(img, x, y, w, h, radius=18, tint=CREAM, alpha=0.72, border=GOLD_LINE, shadow=True):
    """Frosted-glass rounded panel over img[y:y+h, x:x+w]: blurs whatever is
    behind the panel, tints it, composites through a rounded mask, then adds a
    soft drop shadow and a thin accent border. Mutates `img` in place."""
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    pw, ph = x2 - x1, y2 - y1

    if shadow:
        sy1, sy2 = min(H, y1 + 5), min(H, y2 + 10)
        sx1, sx2 = max(0, x1 + 2), min(W, x2 + 4)
        if sy2 > sy1 and sx2 > sx1:
            shadow_mask = _rounded_mask(sy2 - sy1, sx2 - sx1, radius)
            region = img[sy1:sy2, sx1:sx2]
            dark = np.zeros_like(region)
            blended = cv2.addWeighted(region, 0.75, dark, 0.25, 0)
            m3 = shadow_mask[..., None] / 255.0
            img[sy1:sy2, sx1:sx2] = (blended * m3 + region * (1 - m3)).astype(np.uint8)

    mask = _rounded_mask(ph, pw, radius)
    roi = img[y1:y2, x1:x2]
    k = max(3, (min(pw, ph) // 6) | 1)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    tint_layer = np.full_like(roi, tint)
    frosted = cv2.addWeighted(blurred, 1 - alpha, tint_layer, alpha, 0)
    m3 = mask[..., None] / 255.0
    img[y1:y2, x1:x2] = (frosted * m3 + roi * (1 - m3)).astype(np.uint8)

    if border is not None:
        cv2.polylines(img, [_rounded_rect_pts(x1, y1, pw, ph, radius)], True, border, 1, cv2.LINE_AA)


def _draw_pill_label(img, text, org, text_color, bg_color, font_scale=0.5):
    """Small rounded glass pill with text, anchored so its bottom-left sits at `org`
    (used for the live width x height readout while dragging a selection box)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad_x, pad_y = 8, 5
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    x, y = org
    x1, y1 = x, y - th - pad_y
    x2, y2 = x + tw + pad_x * 2, y + pad_y
    draw_glass_panel(img, x1, y1, x2 - x1, y2 - y1, radius=(y2 - y1) // 2,
                      tint=bg_color, alpha=0.85, border=None, shadow=False)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y - 2), font, font_scale, text_color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Reference selection
# --------------------------------------------------------------------------- #
def _draw_instruction_banner(img, lines):
    """Frosted-glass instruction card pinned near the top of the frame, so
    on-screen guidance reads as part of the interface rather than a debug
    overlay burned onto the video."""
    pad_x, pad_y, line_h = 16, 12, 24
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Line 0 is drawn in FONT_HERSHEY_DUPLEX (bolder/wider than SIMPLEX at the same
    # scale) -- measure each line with the font it's actually drawn in, or the header
    # overflows past the right edge of the panel (caught visually during a live test run).
    def _line_w(i, line):
        face = cv2.FONT_HERSHEY_DUPLEX if i == 0 else font
        return cv2.getTextSize(line, face, 0.55, 1)[0][0]
    max_w = max(_line_w(i, line) for i, line in enumerate(lines))
    w = min(img.shape[1] - 24, max_w + pad_x * 2 + 6)
    h = pad_y * 2 + line_h * len(lines)
    x, y = 12, 12
    draw_glass_panel(img, x, y, w, h, radius=16, tint=CREAM, alpha=0.72, border=GOLD_LINE)
    for i, line in enumerate(lines):
        ty = y + pad_y + line_h * i + 17
        color = INK if i == 0 else (SAGE if i == len(lines) - 1 else INK_SOFT)
        font_face = cv2.FONT_HERSHEY_DUPLEX if i == 0 else font
        cv2.putText(img, line, (x + pad_x, ty), font_face, 0.55, color, 1, cv2.LINE_AA)


def select_box_interactive(win_name, frame, banner_lines):
    """Styled, glassmorphic replacement for cv2.selectROI: drag with the mouse
    to draw a box, with a translucent sage selection fill, a live width x
    height readout pill, and the same glass instruction banner used
    elsewhere. Returns (x, y, w, h), or (0, 0, 0, 0) if cancelled."""
    state = {"dragging": False, "start": None, "end": None, "box": None}

    def on_mouse(event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = (mx, my)
            state["end"] = (mx, my)
            state["box"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["end"] = (mx, my)
            x1, y1 = state["start"]
            x2, y2 = state["end"]
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            state["box"] = (x, y, w, h) if w > 2 and h > 2 else None

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        disp = frame.copy()
        if state["start"] and state["end"]:
            x1, y1 = state["start"]
            x2, y2 = state["end"]
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 1 and h > 1:
                overlay = disp.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), SAGE, -1)
                cv2.addWeighted(overlay, 0.22, disp, 0.78, 0, disp)
                cv2.rectangle(disp, (x, y), (x + w, y + h), GOLD_LINE, 2, cv2.LINE_AA)
                for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
                    cv2.circle(disp, (cx, cy), 4, GOLD_LINE, -1, cv2.LINE_AA)
                _draw_pill_label(disp, f"{w} x {h} px", (x, max(20, y - 10)), INK, CREAM)
        _draw_instruction_banner(disp, banner_lines)
        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and state["box"]:
            return state["box"]
        if key in (ord("c"), ord("C")):
            state["start"] = state["end"] = state["box"] = None
        if key == 27:
            return (0, 0, 0, 0)


def _prep_review_photo(crop_img, target_h=420, max_w=620):
    """Upscale a review crop for display -- INTER_CUBIC + a light unsharp-mask pass (see
    dev_notes/LOG.md, 'Round 2' for why INTER_NEAREST was replaced), capped in width so an unusually
    wide context crop doesn't blow up the window."""
    disp = crop_img.copy()
    h0 = max(disp.shape[0], 1)
    scale = target_h / h0
    disp = cv2.resize(disp, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if disp.shape[1] > max_w:
        shrink = max_w / disp.shape[1]
        disp = cv2.resize(disp, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    if scale > 1:
        blurred = cv2.GaussianBlur(disp, (0, 0), sigmaX=1.2)
        disp = cv2.addWeighted(disp, 1.4, blurred, -0.4, 0)
    return disp


def confirm_event_interactive(crop_imgs, event_num, total_events, start_s, end_s,
                               captions=None, flag_note=None):
    """Japandi/glassmorphic Y/N/S review card for one flagged event -- a frosted photo frame
    around each crop (1 or 2, shown side by side; see call site: a wider, highlighted context
    crop from the start *and* end of the fragment, not one tight torso crop, after a live test
    found a single tight crop wasn't enough to make an identity judgement from -- see dev_notes/LOG.md,
    "Round 2"), an optional highlighted flag_note explaining *why* this event was flagged, and
    click-or-key pill buttons below it. Returns True (yes -> promote to staff), False (no ->
    confirmed not staff), or None (skip / decide later -> stays in the review CSV as
    unresolved)."""
    if isinstance(crop_imgs, np.ndarray):
        crop_imgs = [crop_imgs]
    captions = captions or [None] * len(crop_imgs)
    disps = [_prep_review_photo(c) for c in crop_imgs]

    margin, photo_pad, photo_gap, header_h, caption_h, button_h, gap = 36, 18, 24, 130, 30, 72, 20
    flag_h = 30 if flag_note else 0
    row_w = sum(d.shape[1] + photo_pad * 2 for d in disps) + photo_gap * (len(disps) - 1)
    row_h = max(d.shape[0] for d in disps) + photo_pad * 2
    # Header text can be wider than the photo row (e.g. a long flag_note) -- measure every
    # header line with the font it's actually drawn in, or it overflows the card's right edge
    # (same bug pattern as the instruction banner; see dev_notes/LOG.md, "Round 2").
    text_w = max(
        cv2.getTextSize("Is this the staff member?", cv2.FONT_HERSHEY_DUPLEX, 0.95, 2)[0][0],
        cv2.getTextSize(f"Flagged: {flag_note}", cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0]
        if flag_note else 0,
    )
    card_w = max(row_w, text_w, 480) + margin * 2
    card_h = margin + header_h + flag_h + row_h + caption_h + gap + button_h + margin
    canvas = np.full((card_h, card_w, 3), CREAM, dtype=np.uint8)

    cv2.putText(canvas, f"REVIEW  {event_num} OF {total_events}", (margin, margin + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, INK_SOFT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Is this the staff member?", (margin, margin + 64),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{start_s:.1f}s - {end_s:.1f}s in the video", (margin, margin + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK_SOFT, 1, cv2.LINE_AA)
    if flag_note:
        cv2.putText(canvas, f"Flagged: {flag_note}", (margin, margin + header_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, TERRACOTTA, 1, cv2.LINE_AA)

    row_y = margin + header_h + flag_h
    bx = margin + (card_w - margin * 2 - row_w) // 2
    for disp, caption in zip(disps, captions):
        dh, dw = disp.shape[:2]
        photo_x = bx + photo_pad
        photo_y = row_y + photo_pad
        draw_glass_panel(canvas, bx, row_y, dw + photo_pad * 2, dh + photo_pad * 2,
                          radius=14, tint=CREAM, alpha=0.35, border=GOLD_LINE)
        canvas[photo_y:photo_y + dh, photo_x:photo_x + dw] = disp
        if caption:
            cv2.putText(canvas, caption, (photo_x, row_y + dh + photo_pad * 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK_SOFT, 1, cv2.LINE_AA)
        bx += dw + photo_pad * 2 + photo_gap

    btn_y = row_y + row_h + caption_h + gap
    btn_w = (card_w - margin * 2 - gap * 2) // 3
    buttons = [("Y", "Yes, staff", SAGE, CREAM), ("N", "No", TERRACOTTA, CREAM), ("S", "Skip", SAND, INK)]
    regions = []
    bx = margin
    for key, label, color, text_color in buttons:
        draw_glass_panel(canvas, bx, btn_y, btn_w, button_h, radius=button_h // 2,
                          tint=color, alpha=0.85, border=None, shadow=True)
        text = f"{key}   {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = bx + (btn_w - tw) // 2, btn_y + (button_h + th) // 2
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        regions.append((bx, btn_y, bx + btn_w, btn_y + button_h, key))
        bx += btn_w + gap

    win = f"Staff Review  --  {event_num} of {total_events}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Force the window's client area to match the canvas 1:1 -- on some Windows/high-DPI setups
    # WINDOW_NORMAL's initial size doesn't reliably match the shown image, opening noticeably
    # smaller than intended (reported directly from a live test run).
    cv2.resizeWindow(win, canvas.shape[1], canvas.shape[0])
    clicked = {"key": None}

    def on_mouse(event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, k in regions:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    clicked["key"] = k

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, canvas)
    decision = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        chosen = clicked["key"]
        if chosen is None and key != 255:
            if key in (ord("y"), ord("Y")):
                chosen = "Y"
            elif key in (ord("n"), ord("N")):
                chosen = "N"
            elif key in (ord("s"), ord("S"), 27, 13, 32):
                chosen = "S"
        if chosen == "Y":
            decision = True
            break
        if chosen == "N":
            decision = False
            break
        if chosen == "S":
            decision = None
            break
    cv2.destroyWindow(win)
    return decision


def select_reference_interactive(video_path):
    """Scrub to a frame (trackbar), then drag a box around the staff member."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print("STEP 3: Find the staff member")
    print("-" * 64)
    print("A video window will open with a slider (labelled 'frame') at the top.")
    print("  1. Drag that slider left/right to scrub through the video until the")
    print("     staff member is clearly visible.")
    print("  2. Click on the video window first so it's the active window.")
    print("  3. Press ENTER (or Spacebar) to lock in that frame.")
    print("     Press ESC any time to cancel.")
    print()
    print("  Tip: if their clothing changes partway through the video (e.g. a jacket goes")
    print("  on/off), pick whichever frame shows the most distinctive, least generic color")
    print("  (a bright/unusual color beats black/gray/navy).")
    print()

    win = "STEP 3a: Scrub to the staff member, then press ENTER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    state = {"idx": 0}
    cv2.createTrackbar("frame", win, 0, max(n_frames - 1, 1), lambda pos: state.__setitem__("idx", pos))

    frame = None
    chosen_idx = None
    last_idx = -1
    while True:
        if state["idx"] != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, state["idx"])
            ok, frame = cap.read()
            if not ok:
                break
            last_idx = state["idx"]
        disp = frame.copy()
        _draw_instruction_banner(disp, [
            f"Frame {state['idx']} / {n_frames - 1}  (drag the slider above to change)",
            "ENTER = use this frame   |   ESC = cancel",
        ])
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            chosen_idx = state["idx"]
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Reference selection cancelled.")
    cv2.destroyWindow(win)
    print(f"-> Frame {chosen_idx} selected.")

    print()
    print("STEP 4: Draw a box around the staff member")
    print("-" * 64)
    print("  1. Click near one corner of the staff member's body, hold the mouse")
    print("     button down, and drag to the opposite corner -- this draws a box.")
    print("  2. Release the mouse button. Redraw as many times as you like.")
    print("  3. Press ENTER (or Spacebar) to confirm the box.")
    print("     Press 'c' to clear and redraw, or ESC to cancel.")
    print()

    box = select_box_interactive(
        "STEP 3b: Drag a box around the staff member, then press ENTER",
        frame,
        ["Drag a box around the staff member",
         "ENTER = confirm   |   C = clear   |   ESC = cancel"],
    )
    cv2.destroyAllWindows()
    cap.release()
    x, y, w, h = box
    if w == 0 or h == 0:
        sys.exit("No region selected.")
    print(f"-> Box selected: x={x}, y={y}, width={w}, height={h}")
    return chosen_idx, (x, y, w, h), frame


def get_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read frame {frame_idx} from {video_path}")
    return frame


# --------------------------------------------------------------------------- #
# Appearance matching
# --------------------------------------------------------------------------- #
class ClipMatcher:
    def __init__(self, model_name=CLIP_MODEL_NAME, device="cpu"):
        # Imported here, not at module level, so `transformers` is only required when
        # --use-clip is actually passed (this class is only ever instantiated in that case)
        # rather than for every run of the script.
        from transformers import CLIPModel, CLIPImageProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed(self, bgr_images):
        rgb_images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in bgr_images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.get_image_features(**inputs)
        feats = out.pooler_output  # transformers>=5: get_image_features returns a
        # BaseModelOutputWithPooling whose .pooler_output holds the projected CLIP
        # embedding, not a plain tensor -- see dev_notes/LOG.md "Round 2" for how this was found.
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


def mean_lab(bgr_image, mask=None):
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() < 10:
            m = np.ones(mask.shape, dtype=bool)
        return lab[m].mean(axis=0)
    return lab.reshape(-1, 3).mean(axis=0)


def track_motion_stats(entries):
    """avg per-frame centroid speed (px/frame) and total positional range (px) for a
    track's (frame_idx, cx, cy, fx, fy) entries -- used to tell an actually-walking
    person apart from someone sitting/typing at a desk (see call site)."""
    pts = sorted((e[0], e[1], e[2]) for e in entries)
    if len(pts) < 2:
        return 0.0, 0.0
    dists = [np.hypot(pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2]) / max(1, pts[i + 1][0] - pts[i][0])
              for i in range(len(pts) - 1)]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    rng = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    return float(np.mean(dists)), rng


BRIDGE_MAX_GAP_SECONDS = 1.0    # max time gap to bridge across between a confirmed-staff
BRIDGE_MAX_JUMP_PX = 120.0      # fragment and an adjacent one, or across a true detection
BRIDGE_COLOR_FLOOR = 0.3        # gap -- see bridge_track_fragments() / interpolate_staff_gaps().
# Measured directly on sample.mp4 (see dev_notes/LOG.md, "Round 2" evaluation entry): even during the
# plain-shirt period, where color-matching works cleanly, frame-level recall was only ~31%
# because ByteTrack keeps losing and re-acquiring the walking person, splitting one continuous
# walk into many short track fragments -- some too brief to individually clear
# --min-track-seconds or the motion-based track filter, some with zero detection at all for a few
# frames. The person doesn't teleport, so a short, spatially-plausible gap next to an
# already-confirmed staff sighting is almost certainly the same walk continuing.


def bridge_track_fragments(track_scores, track_coords, staff_tracks, max_gap_seconds, max_jump_px,
                            color_floor, fps):
    """Pull short/marginal track fragments into `staff_tracks` (in place) when they're an
    obvious spatial+temporal continuation of an already-confirmed staff fragment -- see the
    BRIDGE_* constants above for why. Runs to a fixed point: a newly bridged fragment can
    itself anchor further bridges, so one genuinely fragmented walk (many short pieces in a
    row) gets pulled in as a chain, not just the one piece directly touching a confirmed track.
    A lenient color floor guards against bridging in an unrelated person who happens to be at
    the right place at the right time (rare, since two people can't occupy the same ~120px
    spot within ~1s of each other except at a literal handoff) -- see dev_notes/LOG.md, "Round 2" for the
    one real handoff found on this footage and why it's handled separately, not by this
    function (that case is a genuine, if brief, tracker ID-switch onto a different person, not
    a fragmented continuation of the same one; see split_event_by_color_outliers()).

    Returns the set of newly-bridged track IDs, for reporting."""
    max_gap_frames = max_gap_seconds * fps
    spans = {}
    for tid, coords in track_coords.items():
        entries = sorted(coords, key=lambda e: e[0])
        spans[tid] = (entries[0][0], entries[-1][0], entries[0][3:5], entries[-1][3:5])

    bridged = set()
    changed = True
    while changed:
        changed = False
        for tid, (start_f, end_f, start_pt, end_pt) in spans.items():
            if tid in staff_tracks:
                continue
            own_scores = [s for _, s in track_scores[tid]]
            if max(own_scores, default=0.0) < color_floor:
                continue
            for other_tid, (o_start, o_end, o_start_pt, o_end_pt) in spans.items():
                if other_tid == tid or other_tid not in staff_tracks:
                    continue
                if 0 <= start_f - o_end <= max_gap_frames:
                    jump = float(np.hypot(start_pt[0] - o_end_pt[0], start_pt[1] - o_end_pt[1]))
                elif 0 <= o_start - end_f <= max_gap_frames:
                    jump = float(np.hypot(end_pt[0] - o_start_pt[0], end_pt[1] - o_start_pt[1]))
                else:
                    continue
                if jump <= max_jump_px:
                    staff_tracks.add(tid)
                    bridged.add(tid)
                    changed = True
                    break
    return bridged


def resolve_simultaneous_staff_conflicts(staff_tracks, track_scores, track_coords):
    """Only one person is ever actually the tagged staff member (per the task brief -- other
    people in frame are explicitly untagged). If two or more auto-qualified tracks are active
    at the *same* time, at most one of them can genuinely be staff -- found necessary on real
    footage: a 201-frame, ~8s track (median score 0.55, just over --staff-threshold) and an
    18-frame track (median score 0.92) were both auto-qualified and shown as STAFF
    simultaneously in the same frames, because each was judged independently against the
    threshold with no awareness of the other. See dev_notes/LOG.md, "Round 2".

    For each group of temporally-overlapping qualified tracks, keeps only the single
    highest-median-score one in `staff_tracks` (in place) and removes the rest. Unlike a
    color/motion-ambiguous case, a demoted track already has strong, objective evidence against
    it (a higher-scoring track was simultaneously active) -- see the call site for why these are
    logged and auto-rejected rather than sent through the interactive Y/N/S review, which is
    reserved for genuine, subjective ambiguity.

    Returns {demoted_track_id: winner_track_id} for reporting/audit."""
    spans = []
    for tid in staff_tracks:
        frames = [f for f, _ in track_scores[tid]]
        median_score = float(np.median([s for _, s in track_scores[tid]]))
        spans.append((min(frames), max(frames), tid, median_score))
    spans.sort()

    demoted = {}
    active = []  # tracks whose span could still overlap a later one
    for start, end, tid, score in spans:
        active = [a for a in active if a[1] >= start]  # drop ones that ended before this starts
        for _, a_end, a_tid, a_score in active:
            if a_tid in demoted or tid in demoted:
                continue
            if score < a_score:
                demoted[tid] = a_tid
            else:
                demoted[a_tid] = tid
        active.append((start, end, tid, score))

    for tid in demoted:
        staff_tracks.discard(tid)
    return demoted


def interpolate_staff_gaps(staff_tracks, track_coords, max_gap_seconds, max_jump_px, fps):
    """Fill true detection gaps (no track at all for a few frames, e.g. brief total occlusion)
    between two staff-track observations that are close in time and space -- same "the person
    doesn't teleport" reasoning as bridge_track_fragments(), but for frames with no fragment to
    bridge at all. Returns {frame_idx: (x, y)} for the synthesized in-between frames only --
    never overwrites or replaces a real detection."""
    obs = []
    for tid in staff_tracks:
        for f, cx, cy, fx, fy in track_coords[tid]:
            obs.append((f, fx, fy))
    obs.sort()
    max_gap_frames = max_gap_seconds * fps
    interpolated = {}
    for (f0, x0, y0), (f1, x1, y1) in zip(obs, obs[1:]):
        gap = f1 - f0
        if 1 < gap <= max_gap_frames:
            jump = float(np.hypot(x1 - x0, y1 - y0))
            if jump <= max_jump_px:
                for f in range(f0 + 1, f1):
                    t = (f - f0) / gap
                    interpolated[f] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return interpolated


EVENT_MERGE_GAP_SECONDS = 5.0   # fragments this close in time are treated as one walking event.
# Tuned empirically: even the confirmed real staff track fragmented into pieces with gaps up
# to ~2.2s between them (tracker losing/re-acquiring during brief occlusion), so 2.0s was too
# tight -- it split one real occurrence into several "events" that then falsely confirmed each
# other as a repeating color via the check below. See dev_notes/LOG.md, "Round 2".
COLOR_REPEAT_LAB_DIST = 20.0    # two events this close in color = same recurring (non-staff) person
FRAGMENT_OUTLIER_LAB_DIST = 35.0  # a fragment's own color this far from its event's majority
# color is treated as a likely different physical person, not just a lighting/pose blip --
# see split_event_by_color_outliers(). Found necessary on real footage: a merged event can
# still silently contain a tracker ID-switch onto a different person for part of its span
# (see dev_notes/LOG.md, "Round 2" -- a second person made brief physical contact with the staff and the
# tracker handed the same "walking" chain off onto them for ~1s), which the event-merging below
# can't catch on its own since it only ever checks *time* proximity between fragments.
MAX_BRIDGE_SPEED_PX_PER_FRAME = 35.0  # an implied speed above this between two time-adjacent
# fragments is a teleport, not a walk -- generous headroom above the real measured max walking
# speed (~12-17px/frame, see "Calibration reference points" in KNOWLEDGE.md), but well below
# what a genuine tracker ID-hop produces. Added after the color check above measurably missed a
# real case: the staff (in a dark jacket) and a second person (in a similarly near-black shirt)
# had almost identical mean Lab color (distance ~4, nowhere near FRAGMENT_OUTLIER_LAB_DIST), so
# only position gave it away -- a ~126px jump between the two fragments' nearest frames, at the
# exact moment the second person made physical contact with the staff. See dev_notes/LOG.md, "Round 2".


def split_event_by_position_jump(ev, track_scores, track_coords, track_lab_colors):
    """Split an event wherever consecutive fragments (by start time) imply a physically
    impossible speed between them -- almost certainly a different physical person picked up
    mid-event by the tracker, not the same one walking. Complements
    split_event_by_color_outliers(), which can miss exactly this when the two people happen to
    be dressed in similarly-colored clothing -- found necessary on real footage, see
    MAX_BRIDGE_SPEED_PX_PER_FRAME above and dev_notes/LOG.md, "Round 2". Run this *before* the color-outlier
    split (find_possible_staff_events does), since a position-implausible boundary should be
    treated as a hard split regardless of what the color check would have concluded on its own.

    Returns a list of event dicts covering the same tracks/frames as `ev`, split at each
    position-implausible boundary (a list of one, unchanged, if none is found)."""
    tids_sorted = sorted(ev["track_ids"], key=lambda t: min(f for f, _ in track_scores[t]))
    if len(tids_sorted) < 2:
        return [ev]

    span = {}
    for t in tids_sorted:
        entries = sorted(track_coords[t], key=lambda e: e[0])
        span[t] = (entries[0][0], entries[-1][0], entries[0][3:5], entries[-1][3:5])

    groups = [[tids_sorted[0]]]
    for prev, cur in zip(tids_sorted, tids_sorted[1:]):
        gap = span[cur][0] - span[prev][1]
        jump = float(np.hypot(span[cur][2][0] - span[prev][3][0], span[cur][2][1] - span[prev][3][1]))
        if gap <= 0:
            # Fragments literally overlap in time (both have a detection in a shared frame) --
            # "implied speed" isn't a coherent idea for zero/negative elapsed time (dividing by
            # a forced denominator of 1 here was a bug: it made *any* overlap between two
            # fragments read as a huge speed and get flagged, including a same person's own
            # fragments during a normal tracker handoff -- found on real footage, where this
            # caused one continuous walk to fragment into 6 separate reviews instead of one).
            # A large spatial gap while genuinely coexisting is still direct proof of two
            # different people (one person can't be in two places at once); a small one is more
            # likely the tracker briefly double-emitting the same person mid-handoff.
            suspicious = jump > BRIDGE_MAX_JUMP_PX
        else:
            suspicious = (jump / gap) > MAX_BRIDGE_SPEED_PX_PER_FRAME
        if suspicious:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    if len(groups) == 1:
        return [ev]

    sub_events = []
    for gi, g in enumerate(groups):
        starts = [span[t][0] for t in g]
        ends = [span[t][1] for t in g]
        colors = [c for t in g for c in track_lab_colors[t]]
        sub_events.append({
            "track_ids": g,
            "start_frame": min(starts),
            "end_frame": max(ends),
            "mean_lab": np.mean(colors, axis=0),
            "representative_track_id": max(g, key=lambda t: len(track_scores[t])),
            "likely_recurring_other_person": ev.get("likely_recurring_other_person", False),
            # Only the group(s) picked up *after* a detected jump are themselves the anomaly --
            # the first group is just "whatever came before the jump," nothing wrong with it.
            "position_jump": gi > 0,
        })
    return sub_events


def split_event_by_color_outliers(ev, track_scores, track_lab_colors):
    """One merged event (grouped by time proximity only, see find_possible_staff_events) can
    still silently contain a different physical person for part of its span -- e.g. a tracker
    ID-switch during brief physical contact between two people, found on real footage (see
    dev_notes/LOG.md, "Round 2"): a second person briefly touched the staff and the tracker handed the
    same walking chain off onto them for about a second before handing it back. A blanket
    Y/N/S confirmation over the whole event would approve (or reject) that switched-in
    fragment right along with everything else, since nothing before this function looks at
    whether a fragment's *own* color is still consistent with the rest of the event.

    Splits `ev` into contiguous sub-segments (preserving time order) wherever a fragment's own
    mean color diverges from the event's majority color (the median across its fragments' own
    means -- robust to one bad fragment skewing a plain pooled average) by more than
    FRAGMENT_OUTLIER_LAB_DIST. Each sub-segment becomes its own independent review item with a
    `color_outlier` flag, instead of one blanket decision covering a possible identity switch
    along with the genuine parts of the event. Returns a list of event dicts (a list of one,
    unchanged, if nothing diverges or the event has too few fragments to judge a "majority")."""
    tids_sorted = sorted(ev["track_ids"], key=lambda t: min(f for f, _ in track_scores[t]))
    if len(tids_sorted) < 2:
        ev["color_outlier"] = False
        ev.setdefault("position_jump", False)
        return [ev]

    frag_means = {t: np.mean(track_lab_colors[t], axis=0) for t in tids_sorted}
    majority = np.median(np.stack(list(frag_means.values())), axis=0)
    is_outlier = {t: float(np.linalg.norm(frag_means[t] - majority)) > FRAGMENT_OUTLIER_LAB_DIST
                  for t in tids_sorted}

    if not any(is_outlier.values()):
        ev["color_outlier"] = False
        ev.setdefault("position_jump", False)
        return [ev]

    # Partition into contiguous runs of same outlier-ness, preserving time order, so a switch
    # in the middle of an event becomes its own separate sub-event rather than splitting every
    # single fragment individually.
    runs = []
    for t in tids_sorted:
        if runs and runs[-1]["outlier"] == is_outlier[t]:
            runs[-1]["tids"].append(t)
        else:
            runs.append({"outlier": is_outlier[t], "tids": [t]})

    sub_events = []
    for run in runs:
        starts = [min(f for f, _ in track_scores[t]) for t in run["tids"]]
        ends = [max(f for f, _ in track_scores[t]) for t in run["tids"]]
        colors = [c for t in run["tids"] for c in track_lab_colors[t]]
        sub_events.append({
            "track_ids": run["tids"],
            "start_frame": min(starts),
            "end_frame": max(ends),
            "mean_lab": np.mean(colors, axis=0),
            "representative_track_id": max(run["tids"], key=lambda t: len(track_scores[t])),
            "likely_recurring_other_person": ev.get("likely_recurring_other_person", False),
            "color_outlier": run["outlier"],
            "position_jump": ev.get("position_jump", False),
        })
    return sub_events


def find_possible_staff_events(candidate_tids, track_scores, track_coords, track_lab_colors, fps):
    """Group unmatched-but-walking tracks into "events" (a real walk-through can fragment
    into several track IDs -- see dev_notes/LOG.md, "Round 2" -- so raw track IDs would flood a review
    list with near-duplicates of the same event). Then drop any event whose color repeats
    across multiple separated events, since a color that keeps recurring is better explained
    by a regular non-staff person than a one-off clothing change. Each surviving event is then
    further split at any internal discontinuity -- first a physically-implausible *position*
    jump between fragments (split_event_by_position_jump), then any remaining *color*
    divergence (split_event_by_color_outliers) -- so a same-event tracker ID-switch onto a
    different person doesn't ride along with a blanket approval of the rest of the event. Position
    is checked first and separately because it catches a case color can't: two people dressed in
    similarly-colored clothing (found on real footage -- see dev_notes/LOG.md, "Round 2"). What's left is
    flagged as `possible_staff`: same idea as a detective using timing/behaviour, not appearance,
    once appearance itself can't be trusted (e.g. staff changes into a jacket -- see conversation
    that led to this, and dev_notes/LOG.md "Round 2").

    Returns a list of (possibly split) events, each a dict: track_ids, start_frame, end_frame,
    mean_lab, representative_track_id (the longest fragment, for a representative crop),
    color_outlier (True if split out for diverging color), position_jump (True if split out for
    an implausible position jump)."""
    if not candidate_tids:
        return []

    spans = []
    for tid in candidate_tids:
        frames = [f for f, _ in track_scores[tid]]
        spans.append((min(frames), max(frames), tid))
    spans.sort()

    gap_frames = EVENT_MERGE_GAP_SECONDS * fps
    events = []
    cur = {"track_ids": [spans[0][2]], "start_frame": spans[0][0], "end_frame": spans[0][1]}
    for start, end, tid in spans[1:]:
        if start - cur["end_frame"] <= gap_frames:
            cur["track_ids"].append(tid)
            cur["end_frame"] = max(cur["end_frame"], end)
        else:
            events.append(cur)
            cur = {"track_ids": [tid], "start_frame": start, "end_frame": end}
    events.append(cur)

    for ev in events:
        all_colors = [c for tid in ev["track_ids"] for c in track_lab_colors[tid]]
        ev["mean_lab"] = np.mean(all_colors, axis=0)
        ev["representative_track_id"] = max(ev["track_ids"], key=lambda t: len(track_scores[t]))

    # Drop events whose color repeats elsewhere -- likely a regular (non-staff) person seen
    # more than once, not a one-off appearance change.
    kept = []
    for i, ev in enumerate(events):
        repeats = sum(
            1 for j, other in enumerate(events)
            if j != i and float(np.linalg.norm(ev["mean_lab"] - other["mean_lab"])) < COLOR_REPEAT_LAB_DIST
        )
        ev["likely_recurring_other_person"] = repeats > 0
        if repeats == 0:
            kept.append(ev)

    split = []
    for ev in kept:
        for piece in split_event_by_position_jump(ev, track_scores, track_coords, track_lab_colors):
            split.extend(split_event_by_color_outliers(piece, track_scores, track_lab_colors))
    return split


def masked_crop(frame, box_xyxy, mask_full, pad_frac=0.08):
    """Crop a box (with small padding) and replace non-mask pixels with a neutral
    background, so appearance matching isn't diluted by whatever furniture/floor a
    loose detector box happens to include (round 1's dominant false-positive cause)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_frac))
    y1 = max(0, int(y1 - bh * pad_frac))
    x2 = min(W, int(x2 + bw * pad_frac))
    y2 = min(H, int(y2 + bh * pad_frac))
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2].copy()
    mask_crop = mask_full[y1:y2, x1:x2]
    bg = np.full_like(crop, NEUTRAL_BG)
    out = np.where(mask_crop[..., None] > 0.5, crop, bg)
    return out, mask_crop


REVIEW_CROP_PAD_FRAC = 1.2   # how much surrounding context to include around the person in a
HIGHLIGHT_COLOR = (0, 215, 255)  # review popup -- a tight torso-only crop (the original
# behavior) was found, directly by a user reviewing a real flagged event, to be too tight to
# make an identity judgement from (no face, no surroundings for scale/context) -- they ended up
# judging by clothing color instead, which is exactly the confounded signal in the one failure
# case found so far (see dev_notes/LOG.md, "Round 2"). A generously padded crop with the actual person
# highlighted (so it's still obvious who's being asked about) trades a bit of screen space for
# a much better-informed human decision.


def padded_highlight_crop(frame, box_xyxy, pad_frac=REVIEW_CROP_PAD_FRAC, highlight_color=HIGHLIGHT_COLOR):
    """Crop a generously padded region around `box_xyxy` (context for a human reviewer, not an
    appearance-matching ROI -- unlike masked_crop, nothing here is fed back into scoring) with a
    bright rectangle drawn around the actual person, so it's unambiguous who's being asked
    about despite the wider view. Returns the annotated crop (a plain frame slice if the box is
    degenerate)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 - bw * pad_frac))
    py1 = max(0, int(y1 - bh * pad_frac))
    px2 = min(W, int(x2 + bw * pad_frac))
    py2 = min(H, int(y2 + bh * pad_frac))
    if px2 <= px1 or py2 <= py1:
        return frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)].copy()
    crop = frame[py1:py2, px1:px2].copy()
    hx1, hy1 = int(x1) - px1, int(y1) - py1
    hx2, hy2 = int(x2) - px1, int(y2) - py1
    cv2.rectangle(crop, (hx1, hy1), (hx2, hy2), highlight_color, 3, cv2.LINE_AA)
    return crop


def hstack_crops(crops, gap=8, gap_color=CREAM):
    """Side-by-side composite of same-height crops (resizing to the shortest one's height
    first if they differ), for saving the review popup's context photos as one audit-trail
    JPG."""
    if len(crops) == 1:
        return crops[0]
    h = min(c.shape[0] for c in crops)
    resized = []
    for c in crops:
        if c.shape[0] != h:
            scale = h / c.shape[0]
            c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        resized.append(c)
    gap_strip = np.full((h, gap, 3), gap_color, dtype=np.uint8)
    parts = []
    for i, c in enumerate(resized):
        if i:
            parts.append(gap_strip)
        parts.append(c)
    return np.hstack(parts)


TRAJECTORY_LINE_BREAK_SECONDS = 1.2  # connecting line breaks across gaps this long -- a
                                     # straight line across a real absence would visually
                                     # imply movement that never happened.
HIGHLIGHT_MERGE_GAP_SECONDS = 0.2   # presence gaps this short are treated as one window
HIGHLIGHT_PAD_SECONDS = 0.5         # context padding added before/after each window


def render_staff_trajectory(points, fps, out_path):
    """points: list of (frame_idx, timestamp_s, x, y) for every staff-present frame, in
    frame order. Saves a scatter+line plot of the tracked (x, y) path, colored by time, to
    out_path. Returns False (nothing written) if there are no staff-present frames at all."""
    if not points:
        return False
    frames = [p[0] for p in points]
    ts = [p[1] for p in points]
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]

    break_frames = max(1, int(TRAJECTORY_LINE_BREAK_SECONDS * fps))
    xs_line, ys_line = [], []
    for i, f in enumerate(frames):
        if i > 0 and f - frames[i - 1] > break_frames:
            xs_line.append(float("nan"))
            ys_line.append(float("nan"))
        xs_line.append(xs[i])
        ys_line.append(ys[i])

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    sca = ax.scatter(xs, ys, c=ts, cmap="viridis", s=14, zorder=3)
    ax.plot(xs_line, ys_line, color="#8a9a7b", linewidth=0.8, alpha=0.6, zorder=2)
    ax.invert_yaxis()  # image coordinates: y grows downward
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Staff member's tracked (x, y) path over the video")
    cbar = fig.colorbar(sca, ax=ax)
    cbar.set_label("time (s)")
    fig.tight_layout()
    fig.savefig(str(out_path), facecolor="white")
    plt.close(fig)
    return True


def compute_highlight_windows(present_frames, fps, n_frames):
    """present_frames: sorted list of frame indices where staff is present. Groups them into
    contiguous windows, pads each with a little context, then merges any windows that
    overlap/touch after padding (so no frame is ever written twice into the highlight clip --
    that would show as a visible stutter/repeat on playback). Returns a list of (start, end)
    frame-index windows, inclusive."""
    if not present_frames:
        return []
    merge_gap = max(1, int(HIGHLIGHT_MERGE_GAP_SECONDS * fps))
    windows = []
    start = prev = present_frames[0]
    for f in present_frames[1:]:
        if f - prev <= merge_gap:
            prev = f
            continue
        windows.append((start, prev))
        start = prev = f
    windows.append((start, prev))

    pad = int(HIGHLIGHT_PAD_SECONDS * fps)
    merged = []
    for s, e in windows:
        s2, e2 = max(0, s - pad), min(n_frames - 1, e + pad)
        if merged and s2 <= merged[-1][1] + 1:
            ps, pe = merged[-1]
            merged[-1] = (ps, max(pe, e2))
        else:
            merged.append((s2, e2))
    return merged


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_reference(video_path, ref_frame, ref_box, seg_model, clip_matcher):
    """Get the one reference crop everything else is matched against (interactively, if
    ref_frame/ref_box aren't given), then segment that same crop with `seg_model` so the
    reference embedding is mask-restricted too -- apples-to-apples with the candidate crops
    scored later, which go through the same masking (see masked_crop()).

    Returns (ref_frame_idx, ref_box, ref_crop, ref_embed, ref_lab): the frame/box actually
    used (echoed back so a scripted run can be reproduced later with --ref-frame/--ref-box),
    the mask-restricted crop image, its CLIP embedding (None unless clip_matcher is given),
    and its mean Lab color."""
    if ref_frame is None or ref_box is None:
        ref_frame, ref_box, frame = select_reference_interactive(video_path)
    else:
        frame = get_frame(video_path, ref_frame)

    x, y, w, h = ref_box
    ref_crop_raw = frame[y:y + h, x:x + w]

    # Segment the reference crop itself so the reference embedding is also mask-restricted,
    # not just the candidate crops -- keeps the comparison apples-to-apples.
    res = seg_model.predict(frame, classes=[PERSON_CLASS], conf=0.05, verbose=False)[0]
    ref_mask = None
    if res.masks is not None and len(res.boxes) > 0:
        cx, cy = x + w / 2, y + h / 2
        boxes = res.boxes.xyxy.cpu().numpy()
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        dists = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
        best = int(dists.argmin())
        if dists[best] < max(w, h):
            m = res.masks.data[best].cpu().numpy()
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
            ref_mask = m[y:y + h, x:x + w]

    if ref_mask is not None and ref_mask.sum() > 10:
        bg = np.full_like(ref_crop_raw, NEUTRAL_BG)
        ref_crop = np.where(ref_mask[..., None] > 0.5, ref_crop_raw, bg)
    else:
        ref_crop = ref_crop_raw
        ref_mask = np.ones(ref_crop_raw.shape[:2], dtype=np.float32)

    ref_embed = clip_matcher.embed([ref_crop])[0] if clip_matcher is not None else None
    ref_lab = mean_lab(ref_crop_raw, ref_mask)
    return ref_frame, ref_box, ref_crop, ref_embed, ref_lab


# --------------------------------------------------------------------------- #
# Pre-flight checks -- fail fast (or warn early) on demo-day risks instead of discovering them
# deep into an interactive step or after several minutes of Pass 1. See dev_notes/UNEXPECTED.md
# for the full list of risks considered and why these specific ones were picked as safe to fix
# automatically (versus ones that change the actual detection/matching logic and need a
# before/after accuracy check before being trusted).
# --------------------------------------------------------------------------- #
REFERENCE_FPS = 25.0  # sample.mp4's frame rate -- every px/frame motion threshold
                       # (--min-walk-speed) was tuned against it. See run()'s use of this below.
PREVIEW_SAMPLE_COUNT = 6      # frames spot-checked before committing to the full Pass 1
PREVIEW_LOW_SCORE_WARNING = 0.35  # below this best-of-sample score, warn about the reference --
                                   # well under --staff-threshold's default (0.5) so this only
                                   # fires on a genuinely poor pick, not normal score variance.
CLUSTER_BAND = 0.1  # a track's median score within this of --staff-threshold counts as a
                     # "close call" for the score-clustering warning below.


def validate_video(video_path):
    """Open `video_path`, confirm it's actually readable, and return (fps, width, height,
    n_frames) -- fails fast with a clear message here rather than discovering a bad/corrupt
    file deep inside an interactive step or partway through Pass 1."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        sys.exit(f"Could not open video: {video_path}\n"
                  f"Check the file exists and its codec is supported by OpenCV -- mp4/h264 is "
                  f"safest; an unusual container/codec can fail here without a clear reason.")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        sys.exit(f"Opened {video_path} but couldn't read its first frame -- the file may be "
                  f"corrupt, or use a codec OpenCV can't decode on this machine.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames <= 0 or W <= 0 or H <= 0:
        sys.exit(f"Video opened but reported invalid metadata (frames={n_frames}, "
                  f"size={W}x{H}) -- OpenCV may not be parsing this file correctly.")
    print(f"Video: {W}x{H}, {fps:.1f}fps, {n_frames} frames ({n_frames / fps:.1f}s)")
    return fps, W, H, n_frames


def check_gui_available():
    """True if cv2 can actually open a GUI window on this machine. Catches the known
    opencv-python vs opencv-python-headless conflict (see CLAUDE.md, "Environment") up front,
    rather than failing deep inside an interactive step with a cryptic cv2.error."""
    try:
        win = "__gui_check__"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(win)
        return True
    except cv2.error:
        return False


def _probe_writable(path):
    """Try to open `path` for writing without truncating it (append mode -- doesn't touch
    existing content, unlike 'w'), to check it's not locked by another program. Retries with
    the same friendly prompt as open_for_write_retrying()."""
    while True:
        try:
            open(path, "ab").close()
            return
        except PermissionError:
            _wait_for_retry(path)


def check_outputs_writable(out_dir):
    """Pre-flight check: confirm every output filename this run could produce isn't locked by
    another program (e.g. left open in Excel/a photo viewer from a previous run), before Pass 1
    starts -- so that's caught in seconds, not after several minutes of processing."""
    for name in ("staff_detections.csv", "annotated.mp4", "reference_crop.jpg",
                 "staff_trajectory.png", "staff_highlight_clip.mp4",
                 "possible_staff_review.csv", "auto_rejected_conflicts.csv"):
        _probe_writable(out_dir / name)


def preview_reference_score(video_path, seg_model, ref_lab, n_frames,
                             sample_count=PREVIEW_SAMPLE_COUNT):
    """Quick spot-check: run detection on a handful of frames spread across the video and score
    every person found against the reference, before committing to the full multi-minute Pass 1.
    Returns the best score found (0.0 if nobody was detected in any sampled frame at all)."""
    print()
    print(f"Quick preview: scoring the reference against {sample_count} sample frames spread "
          f"across the video (a few seconds)...")
    sample_frames = np.linspace(0, max(n_frames - 1, 0), sample_count, dtype=int)
    best_score = 0.0
    for f in sample_frames:
        frame = get_frame(video_path, int(f))
        res = seg_model.predict(frame, classes=[PERSON_CLASS], conf=0.15, verbose=False)[0]
        if res.masks is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        masks = res.masks.data.cpu().numpy()
        for box, mask in zip(boxes, masks):
            mask_full = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            crop, mask_crop = masked_crop(frame, box, mask_full)
            if crop is None:
                continue
            own_lab = mean_lab(crop, mask_crop)
            lab_dist = float(np.linalg.norm(own_lab - ref_lab))
            best_score = max(best_score, max(0.0, 1.0 - lab_dist / LAB_DIST_SCALE))
    print(f"  Best color-match score found in the quick scan: {best_score:.2f} "
          f"(0 = no resemblance, 1 = identical)")
    if best_score < PREVIEW_LOW_SCORE_WARNING:
        print(f"  Warning: that's low -- the reference crop may be a poor pick (bad frame/box), "
              f"or this person simply doesn't appear in any of the {sample_count} sampled "
              f"frames. The full run may find little or no staff presence.")
    return best_score


def run(args):
    """The whole pipeline, start to finish, for one video. Long, but stays in one function
    since each stage depends on state built up by the previous one (per-frame detections,
    track-level scores/coords, the staff/not-staff decision) -- see the "---- Section ----"
    comments below for where each stage starts. In order:

      1. Load the video + models, build the staff reference (build_reference()).
      2. Pass 1 (single YOLO+ByteTrack pass): detect, track, and score every person in every
         frame against the reference; cache per-frame detections for Pass 2's render.
      3. Track-level staff decision: color threshold + motion-based track filter (see
         CLAUDE.md, "Architecture" step 3), then fragment bridging, simultaneous-conflict
         resolution, and gap interpolation (bridge_track_fragments(),
         resolve_simultaneous_staff_conflicts(), interpolate_staff_gaps()).
      4. Possible-staff flagging + interactive Y/N/S review for anything still ambiguous
         (find_possible_staff_events(), confirm_event_interactive()).
      5. Smooth staff coordinates (Savitzky-Golay) and write staff_detections.csv.
      6. staff_trajectory.png (render_staff_trajectory()) and the highlight-clip windows
         (compute_highlight_windows()).
      7. Pass 2: re-read the video (no re-detection) to draw annotated.mp4 and, in the same
         pass, staff_highlight_clip.mp4.

    Writes everything to args.output_dir and prints a running commentary + final summary;
    doesn't return anything."""
    if args.video is None:
        video, out_name = guided_setup()
        args.video = video
        args.output_dir = args.output_dir or out_name
    if args.output_dir is None:
        args.output_dir = "output"

    video_path = Path(args.video)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-flight checks: fail fast before wasting time on model loading / multi-minute
    # processing (see dev_notes/UNEXPECTED.md for the risks these address) ---- #
    fps, W, H, n_frames = validate_video(video_path)

    ref_box_arg = None
    if args.ref_box:
        ref_box_arg = tuple(int(v) for v in args.ref_box.split(","))
    needs_gui = (args.ref_frame is None or ref_box_arg is None) or not args.skip_review
    if needs_gui and not check_gui_available():
        sys.exit(
            "GUI check failed: cv2 windows aren't working on this machine (cv2.namedWindow "
            "raised an error). This is a known issue when both opencv-python and "
            "opencv-python-headless are installed -- the headless one wins the shared cv2 "
            "namespace and GUI calls silently become no-op stubs. Fix: use this project's "
            ".venv (a clean opencv-python install), or `pip uninstall opencv-python-headless`. "
            "See CLAUDE.md, 'Environment'.\n"
            "To bypass GUI entirely for a scripted run: pass --ref-frame/--ref-box together "
            "with --skip-review."
        )
    check_outputs_writable(out_dir)

    print()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading detector ({args.model})...")
    seg_model = YOLO(args.model)
    clip_matcher = None
    if args.use_clip:
        print("Loading CLIP matcher (--use-clip)...")
        clip_matcher = ClipMatcher(device=device)

    print("Building staff reference...")
    ref_frame_idx, ref_box, ref_crop, ref_embed, ref_lab = build_reference(
        video_path, args.ref_frame, ref_box_arg, seg_model, clip_matcher
    )
    imwrite_retrying(out_dir / "reference_crop.jpg", ref_crop)
    print(f"Reference: frame {ref_frame_idx}, box {ref_box}")

    best_preview_score = preview_reference_score(video_path, seg_model, ref_lab, n_frames)
    if best_preview_score < PREVIEW_LOW_SCORE_WARNING and not args.skip_review:
        choice = input("  Continue with this reference anyway? [Y/n]: ").strip().lower()
        if choice.startswith("n"):
            sys.exit("Cancelled -- re-run and pick a different reference frame/box.")

    min_track_frames = max(3, int(fps * args.min_track_seconds))
    # sample.mp4 was calibrated at REFERENCE_FPS (25); a different fps video's
    # --min-walk-speed (a raw px/frame threshold) is scaled here so the same real-world walking
    # speed is what's actually required, not the same raw pixel count per frame -- see
    # dev_notes/UNEXPECTED.md.
    effective_min_walk_speed = args.min_walk_speed * REFERENCE_FPS / fps
    if abs(fps - REFERENCE_FPS) > 0.5:
        print(f"Note: this video is {fps:.1f}fps (sample.mp4 was calibrated at "
              f"{REFERENCE_FPS:.0f}fps) -- --min-walk-speed effectively adjusted to "
              f"{effective_min_walk_speed:.2f}px/frame to require the same real-world walking "
              f"speed.")

    # ---- Pass 1: detect + track + score, single YOLO pass ---- #
    print()
    print("=" * 64)
    print("  PROCESSING")
    print("=" * 64)
    print(f"Pass 1/2: detecting, tracking, scoring ({n_frames} frames, "
          f"{n_frames / fps:.1f}s of footage)...")
    processing_t0 = time.time()
    t0 = processing_t0
    per_frame_dets = {}   # frame_idx -> list of dict(track_id, box, score)
    track_scores = {}     # track_id -> list of (frame_idx, score)
    track_coords = {}     # track_id -> list of (frame_idx, center_x, center_y, foot_x, foot_y)
    track_lab_colors = {}  # track_id -> list of raw mean-Lab vectors (own clothing color,
                            # not distance-to-reference) -- used to compare unmatched tracks
                            # to each other, for the possible-staff flagging below.

    results_stream = seg_model.track(
        str(video_path), classes=[PERSON_CLASS], conf=args.detect_conf,
        tracker="bytetrack.yaml", persist=True, stream=True, verbose=False, device=device,
    )
    for frame_idx, res in enumerate(results_stream):
        dets = []
        if res.boxes is not None and res.boxes.id is not None and res.masks is not None:
            frame = res.orig_img
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            masks = res.masks.data.cpu().numpy()
            for box, tid, mask in zip(boxes, ids, masks):
                mask_full = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                crop, mask_crop = masked_crop(frame, box, mask_full)
                if crop is None:
                    continue
                own_lab = mean_lab(crop, mask_crop)
                lab_dist = float(np.linalg.norm(own_lab - ref_lab))
                lab_score = max(0.0, 1.0 - lab_dist / LAB_DIST_SCALE)
                if clip_matcher is not None:
                    clip_score = float(clip_matcher.embed([crop])[0] @ ref_embed)
                    score = CLIP_WEIGHT * clip_score + LAB_WEIGHT * lab_score
                else:
                    score = lab_score

                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                fx, fy = (x1 + x2) / 2, y2

                dets.append({"track_id": int(tid), "box": (x1, y1, x2, y2), "score": score})
                track_scores.setdefault(int(tid), []).append((frame_idx, score))
                track_coords.setdefault(int(tid), []).append((frame_idx, cx, cy, fx, fy))
                track_lab_colors.setdefault(int(tid), []).append(own_lab)
        per_frame_dets[frame_idx] = dets
        if frame_idx % 200 == 0:
            pct = frame_idx / n_frames * 100
            print(f"  ...frame {frame_idx}/{n_frames} ({pct:.0f}%), "
                  f"{format_duration(time.time() - t0)} elapsed")

    pass1_elapsed = time.time() - t0
    print(f"Pass 1 complete -- took {format_duration(pass1_elapsed)}.")

    # ---- Track-level staff decision (majority vote) ---- #
    # (Decide first, then print one table with the reasoning -- easier to read than a
    # raw dump followed by a separate count.)
    track_motion = {tid: track_motion_stats(coords) for tid, coords in track_coords.items()}
    staff_tracks = set()
    # Color-matched tracks that fail the walking gate used to be silently dropped here -- no
    # trace anywhere except a "not walking (seated?)" row in the diagnostic table. Now they're
    # collected and routed into the same human-review flow as unmatched-color events below
    # (see "stationary_events" near the possible-staff-flagging section), so a genuinely
    # seated/stationary staff member on an unseen video gets a chance to be caught by a human
    # instead of vanishing with zero indication. See dev_notes/UNEXPECTED.md.
    stationary_candidates = []
    for tid, scores in track_scores.items():
        if len(scores) < min_track_frames:
            continue
        vals = [s for _, s in scores]
        if float(np.median(vals)) < args.staff_threshold:
            continue
        speed, rng = track_motion[tid]
        # Restrict to confirmed walking events, not just a high appearance score: this
        # scene has more than one similarly light-clothed person, so a seated/stationary
        # match is ambiguous (color alone can't disambiguate two people at the same desk)
        # while a person actually walking through the open corridor was the one context
        # round 1 visually confirmed as unambiguous. See dev_notes/LOG.md, "Round 2".
        if speed < effective_min_walk_speed or rng < args.min_walk_range:
            stationary_candidates.append(tid)
            continue
        staff_tracks.add(tid)

    # ---- Bridge short/marginal fragments next to an already-confirmed staff sighting ---- #
    # Measured to matter a lot (see dev_notes/LOG.md, "Round 2" evaluation entry): tracking fragmentation,
    # not the appearance-matching ceiling, turned out to be the main cause of missed frames.
    bridged_tracks = bridge_track_fragments(
        track_scores, track_coords, staff_tracks,
        BRIDGE_MAX_GAP_SECONDS, BRIDGE_MAX_JUMP_PX, BRIDGE_COLOR_FLOOR, fps,
    )
    if bridged_tracks:
        print()
        print(f"Bridged {len(bridged_tracks)} short/marginal track fragment(s) into the staff "
              f"result: each sits within {BRIDGE_MAX_GAP_SECONDS:.1f}s and {BRIDGE_MAX_JUMP_PX:.0f}px "
              f"of an already-confirmed staff sighting, so it's almost certainly the same "
              f"walk continuing through a brief tracking gap, not a separate person.")

    # ---- Only one person is ever actually staff -- resolve any auto-qualified tracks that ---- #
    # ---- overlap in time (at most one of them can be right; see resolve_simultaneous_staff_conflicts) ----
    demoted_tracks = resolve_simultaneous_staff_conflicts(staff_tracks, track_scores, track_coords)
    if demoted_tracks:
        print()
        print(f"Auto-rejected {len(demoted_tracks)} track(s) that were auto-qualified as staff "
              f"but overlapped in time with a higher-scoring qualified track -- only one person "
              f"is ever actually staff, so the weaker overlapping track is excluded outright.")
        print("(Not sent to manual review: a simultaneous higher-scoring track is already strong, "
              "objective evidence, unlike the subjective color/motion ambiguity manual review "
              "is for -- see auto_rejected_conflicts.csv for the full list.)")
        for tid in sorted(demoted_tracks):
            print(f"  #{tid} (median {np.median([s for _, s in track_scores[tid]]):.2f}) "
                  f"loses to #{demoted_tracks[tid]} "
                  f"(median {np.median([s for _, s in track_scores[demoted_tracks[tid]]]):.2f})")

        conflicts_csv_path = out_dir / "auto_rejected_conflicts.csv"
        with open_for_write_retrying(conflicts_csv_path, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "start_frame", "end_frame", "start_time_s", "end_time_s",
                              "median_score", "conflicts_with_track_id", "conflicts_with_median_score"])
            for tid in sorted(demoted_tracks):
                win = demoted_tracks[tid]
                frames = [f for f, _ in track_scores[tid]]
                win_score = float(np.median([s for _, s in track_scores[win]]))
                writer.writerow([tid, min(frames), max(frames), round(min(frames) / fps, 2),
                                  round(max(frames) / fps, 2),
                                  round(float(np.median([s for _, s in track_scores[tid]])), 4),
                                  win, round(win_score, 4)])

    all_scores = [s for scores in track_scores.values() for _, s in scores]
    if all_scores:
        pctl = np.percentile(all_scores, [50, 75, 90, 95])
        print()
        print("Color-match score spread across every person seen in the whole video")
        print("(0 = no resemblance to the reference, 1 = identical):")
        print(f"  typical (median): {pctl[0]:.2f}   top 25%: {pctl[1]:.2f}   "
              f"top 10%: {pctl[2]:.2f}   top 5%: {pctl[3]:.2f}")

    # Warn if a lot of tracks are close calls around --staff-threshold -- color likely isn't
    # discriminating cleanly on this video (e.g. several similarly-dressed people), so expect
    # more possible-staff review events than usual. See dev_notes/UNEXPECTED.md.
    track_medians = {tid: float(np.median([s for _, s in scores]))
                      for tid, scores in track_scores.items() if len(scores) >= min_track_frames}
    close_calls = [tid for tid, m in track_medians.items()
                   if abs(m - args.staff_threshold) <= CLUSTER_BAND]
    if len(track_medians) >= 5 and len(close_calls) / len(track_medians) > 0.15:
        print()
        print(f"Note: {len(close_calls)} of {len(track_medians)} eligible tracks have a median "
              f"color score within {CLUSTER_BAND:.2f} of --staff-threshold "
              f"({args.staff_threshold:.2f}) -- color may not discriminate cleanly on this "
              f"video. Expect more possible-staff review events than usual.")

    print()
    print(f"Per-track breakdown ({len(track_scores)} distinct people tracked; "
          f"a track only counts as staff if it's a good color match AND was actually "
          f"walking, not just seated):")
    header = f"  {'track':>6} {'frames':>7} {'typical':>8} {'best':>6} {'moving?':>8}  verdict"
    print(header)
    print("  " + "-" * (len(header) - 2))
    walking_but_unmatched = []  # candidates for the possible-staff flag below
    for tid, scores in sorted(track_scores.items(), key=lambda kv: -len(kv[1])):
        vals = [s for _, s in scores]
        speed, rng = track_motion[tid]
        is_walking = speed >= effective_min_walk_speed and rng >= args.min_walk_range
        if tid in staff_tracks:
            verdict = "STAFF (bridged)" if tid in bridged_tracks else "STAFF"
        elif tid in demoted_tracks:
            verdict = f"auto-rejected: conflicts with #{demoted_tracks[tid]} (higher-scoring, same time)"
        elif len(vals) < min_track_frames:
            verdict = "too short to trust"
        elif float(np.median(vals)) < args.staff_threshold:
            verdict = "color doesn't match"
            if is_walking:
                walking_but_unmatched.append(tid)
        elif not is_walking:
            verdict = "not walking (seated?)"
        else:
            verdict = "-"
        print(f"  {tid:>6} {len(vals):>7} {np.median(vals):>8.2f} {max(vals):>6.2f} "
              f"{'yes' if is_walking else 'no':>8}  {verdict}")

    # ---- Possible-staff flagging + interactive confirmation ---- #
    # A walking person whose clothing color doesn't match could be staff after an outfit
    # change (e.g. a jacket over the tagged shirt) rather than genuinely someone else -- color
    # can't tell those two cases apart. Rather than guessing (or silently dropping them), ask
    # a human -- one keypress per flagged event, right here, instead of a separate manual
    # investigation. See dev_notes/LOG.md, "Round 2" / conversation with the user for why automating
    # this call outright was rejected: color alone has already been shown (on this exact
    # video) to sometimes flag a genuinely different person, and a wrong auto-label would be
    # silent, while a flagged-for-review case costs a human a few seconds.
    possible_events = find_possible_staff_events(
        walking_but_unmatched, track_scores, track_coords, track_lab_colors, fps
    )

    # Color-matched-but-not-walking tracks (collected in the track-level decision loop above)
    # get the same review treatment -- minus bridging, which may have already pulled some of
    # them into staff_tracks anyway, so only genuinely still-unresolved ones are added. Each is
    # its own single-fragment event (no time-based merging needed: unlike the unmatched-color
    # case, there's no "does this color repeat" ambiguity to resolve first). See
    # dev_notes/UNEXPECTED.md.
    for tid in stationary_candidates:
        if tid in staff_tracks:
            continue
        frames = [f for f, _ in track_scores[tid]]
        possible_events.append({
            "track_ids": [tid],
            "start_frame": min(frames),
            "end_frame": max(frames),
            "mean_lab": np.mean(track_lab_colors[tid], axis=0),
            "representative_track_id": tid,
            "likely_recurring_other_person": False,
            "color_outlier": False,
            "position_jump": False,
            "stationary": True,
        })

    review_rows = []  # (event_num, ev, crop_path, resolution)
    if possible_events:
        print()
        print(f"REVIEW NEEDED: {len(possible_events)} event(s) found that could be staff but "
              f"weren't auto-confirmed -- unmatched clothing color (could be a clothing change), "
              f"or a color match that wasn't detected walking (could be seated staff).")
        for i, ev in enumerate(possible_events, 1):
            start_s, end_s = ev["start_frame"] / fps, ev["end_frame"] / fps
            rep_tid = ev["representative_track_id"]
            rep_frames = [f for f in range(ev["start_frame"], ev["end_frame"] + 1)
                          if any(d["track_id"] == rep_tid for d in per_frame_dets.get(f, []))]

            # Two context crops (start and end of the fragment, not just one) -- a single
            # frame can be an unlucky one (motion blur, backlighting; see dev_notes/LOG.md, "Round 2" for
            # a real case that looked like a solid black silhouette), so a second moment gives
            # a reviewer a better chance of a clear view.
            sample_frames = [rep_frames[0]] if len(rep_frames) == 1 else [rep_frames[0], rep_frames[-1]]
            captions, review_crops = [], []
            for si, f in enumerate(sample_frames):
                box = next(d["box"] for d in per_frame_dets[f] if d["track_id"] == rep_tid)
                review_crops.append(padded_highlight_crop(get_frame(video_path, f), box))
                captions.append(f"{f / fps:.1f}s (start)" if si == 0 and len(sample_frames) > 1
                                 else f"{f / fps:.1f}s (end)" if si == 1
                                 else f"{f / fps:.1f}s")

            crop_path = out_dir / f"review_event_{i}.jpg"
            imwrite_retrying(crop_path, hstack_crops(review_crops))

            notes = []
            if ev.get("position_jump"):
                notes.append("position jump from the previous fragment -- likely a different "
                              "person mid-event, e.g. a tracker ID-switch during contact")
            if ev.get("color_outlier"):
                notes.append("color diverges from the rest of a merged event -- likely a "
                              "different person mid-event")
            if ev.get("stationary"):
                notes.append("color matches well but this track wasn't detected walking -- "
                              "possibly seated/stationary staff (missed by the motion-based "
                              "track filter), or a coincidental match with someone who's "
                              "simply not moving")
            flag_note = "; ".join(notes) if notes else None

            resolution = "needs manual review"
            under_cap = args.max_review_events is None or i <= args.max_review_events
            if not args.skip_review and under_cap:
                decision = confirm_event_interactive(review_crops, i, len(possible_events),
                                                      start_s, end_s, captions, flag_note)
                if decision is True:
                    staff_tracks |= set(ev["track_ids"])
                    resolution = "confirmed STAFF"
                elif decision is False:
                    resolution = "confirmed not staff"
            outlier_note = f"  ({flag_note})" if flag_note else ""
            print(f"  [{i}] frames {ev['start_frame']}-{ev['end_frame']} "
                  f"({start_s:.1f}s-{end_s:.1f}s), {len(ev['track_ids'])} track fragment(s) "
                  f"-> {crop_path.name}  [{resolution}]{outlier_note}")
            review_rows.append((i, ev, crop_path, resolution))

        review_csv_path = out_dir / "possible_staff_review.csv"
        with open_for_write_retrying(review_csv_path, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event", "start_frame", "end_frame", "start_time_s", "end_time_s",
                              "n_track_fragments", "representative_crop", "resolution",
                              "color_outlier", "position_jump", "stationary"])
            for i, ev, crop_path, resolution in review_rows:
                writer.writerow([i, ev["start_frame"], ev["end_frame"],
                                  round(ev["start_frame"] / fps, 2), round(ev["end_frame"] / fps, 2),
                                  len(ev["track_ids"]), crop_path.name, resolution,
                                  ev.get("color_outlier", False), ev.get("position_jump", False),
                                  ev.get("stationary", False)])
        n_confirmed = sum(1 for *_, r in review_rows if r == "confirmed STAFF")
        n_rejected = sum(1 for *_, r in review_rows if r == "confirmed not staff")
        n_pending = len(review_rows) - n_confirmed - n_rejected
        print(f"  -> {n_confirmed} confirmed staff, {n_rejected} confirmed not staff, "
              f"{n_pending} still need manual review  ({review_csv_path})")
        if args.max_review_events is not None and len(possible_events) > args.max_review_events:
            print(f"  (Showed only the first {args.max_review_events} of {len(possible_events)} "
                  f"events live -- --max-review-events cap; the rest are 'needs manual review' "
                  f"in the CSV above.)")
    else:
        review_csv_path = None
        print()
        print("REVIEW NEEDED: none -- no unmatched walking events without a "
              "repeating (likely non-staff) color were found.")

    print()
    print(f"RESULT: {len(staff_tracks)} of {len(track_scores)} tracked people identified as staff.")

    # ---- Fill true detection gaps between staff sightings (no track at all for a few frames,
    # e.g. brief total occlusion) -- same reasoning as the fragment-bridging above, but for
    # frames with no fragment to bridge in the first place. See dev_notes/LOG.md, "Round 2". ---- #
    interpolated_coords = interpolate_staff_gaps(
        staff_tracks, track_coords, BRIDGE_MAX_GAP_SECONDS, BRIDGE_MAX_JUMP_PX, fps
    )
    if interpolated_coords:
        print(f"Filled {len(interpolated_coords)} frame(s) of true detection gaps (no track at "
              f"all, briefly) between two staff sightings close in time and space -- marked "
              f"'interpolated' in the results table, not a real detection.")

    # ---- Smooth (x, y) per staff track ---- #
    # scipy is a required dependency (requirements.txt / CLAUDE.md), not an optional one like
    # transformers -- so a missing install is reported once, loudly, rather than silently
    # skipping smoothing for the whole run (a bare `except Exception: pass` here previously
    # swallowed that case with zero indication to the user; fixed after being flagged during a
    # full readability/correctness pass -- see dev_notes/LOG.md).
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        savgol_filter = None
        print("  Warning: scipy is not installed -- staff coordinates will be left unsmoothed "
              "(Savitzky-Golay smoothing skipped for this whole run). Run "
              "`pip install -r requirements.txt` to enable it.")

    smoothed_coords = {}  # track_id -> {frame_idx: (x, y)}
    for tid in staff_tracks:
        entries = sorted(track_coords[tid], key=lambda e: e[0])
        frames_ = np.array([e[0] for e in entries])
        fx = np.array([e[3] for e in entries], dtype=np.float64)
        fy = np.array([e[4] for e in entries], dtype=np.float64)
        if len(entries) >= 5 and savgol_filter is not None:
            # Savitzky-Golay needs an odd window no larger than the number of points. This
            # clamp always resolves to 5 under the `>= 5` guard just above (verified for
            # every len(entries) from 5 up) -- written this way, rather than a bare `k = 5`,
            # so it stays correct on its own if that guard threshold is ever changed.
            k = min(5, len(entries) - (1 - len(entries) % 2))
            k = k if k % 2 == 1 else k - 1
            k = max(k, 3)
            fx = savgol_filter(fx, k, 2)
            fy = savgol_filter(fy, k, 2)
        smoothed_coords[tid] = {int(f): (float(x), float(y)) for f, x, y in zip(frames_, fx, fy)}

    # ---- Write CSV ---- #
    csv_path = out_dir / "staff_detections.csv"
    with open_for_write_retrying(csv_path, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_s", "staff_present", "track_id", "x", "y",
                          "match_score", "interpolated"])
        n_staff_rows = 0
        n_interpolated_rows = 0
        trajectory_points = []
        staff_present_frames = []
        for frame_idx in range(n_frames):
            ts = round(frame_idx / fps, 2)
            rows_written = 0
            for det in per_frame_dets.get(frame_idx, []):
                tid = det["track_id"]
                if tid not in staff_tracks:
                    continue
                x, y = smoothed_coords[tid].get(frame_idx, (None, None))
                if x is None:
                    continue
                writer.writerow([frame_idx, ts, True, tid, round(x, 1), round(y, 1),
                                  round(det["score"], 4), False])
                rows_written += 1
                n_staff_rows += 1
                trajectory_points.append((frame_idx, ts, x, y))
            if rows_written == 0 and frame_idx in interpolated_coords:
                x, y = interpolated_coords[frame_idx]
                writer.writerow([frame_idx, ts, True, "", round(x, 1), round(y, 1), "", True])
                rows_written += 1
                n_staff_rows += 1
                n_interpolated_rows += 1
                trajectory_points.append((frame_idx, ts, x, y))
            if rows_written == 0:
                writer.writerow([frame_idx, ts, False, "", "", "", "", False])
            if rows_written > 0:
                staff_present_frames.append(frame_idx)

    # ---- Trajectory plot: the staff member's (x, y) path over time ---- #
    trajectory_path = out_dir / "staff_trajectory.png"
    wrote_trajectory = render_staff_trajectory(trajectory_points, fps, trajectory_path)

    # ---- Highlight clip windows: only the frames staff is actually present in ---- #
    highlight_windows = compute_highlight_windows(staff_present_frames, fps, n_frames)
    highlight_frame_set = set()
    for s, e in highlight_windows:
        highlight_frame_set.update(range(s, e + 1))
    highlight_path = out_dir / "staff_highlight_clip.mp4"

    # ---- Pass 2: render annotated video (no re-detection, just reads + draws) ---- #
    flagged_tracks = {tid for ev in possible_events for tid in ev["track_ids"]}
    print()
    print(f"Pass 2/2: rendering annotated video ({n_frames} frames)...")
    pass2_t0 = time.time()
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_vid = open_video_writer_retrying(out_dir / "annotated.mp4", fourcc, fps, (W, H))
    writer_highlight = (open_video_writer_retrying(highlight_path, fourcc, fps, (W, H))
                         if highlight_frame_set else None)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        drew_staff = False
        for det in per_frame_dets.get(frame_idx, []):
            tid = det["track_id"]
            x1, y1, x2, y2 = (int(v) for v in det["box"])
            if tid in staff_tracks:
                color, label = STAFF_BOX_COLOR, f"STAFF #{tid} {det['score']:.2f}"
                drew_staff = True
            elif tid in flagged_tracks:
                color, label = REVIEW_BOX_COLOR, f"REVIEW? #{tid} {det['score']:.2f}"
            else:
                color, label = OTHER_BOX_COLOR, f"#{tid} {det['score']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if not drew_staff and frame_idx in interpolated_coords:
            # No detection at all this frame -- draw the gap-filled position as a marker
            # (a point, not a box: there's no detected box to draw here) instead of a rectangle.
            ix, iy = interpolated_coords[frame_idx]
            cv2.circle(frame, (int(ix), int(iy)), 7, STAFF_BOX_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(ix), int(iy)), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "STAFF (interpolated)", (int(ix) + 10, int(iy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, STAFF_BOX_COLOR, 2, cv2.LINE_AA)
        writer_vid.write(frame)
        if writer_highlight is not None and frame_idx in highlight_frame_set:
            writer_highlight.write(frame)
        frame_idx += 1
    cap.release()
    writer_vid.release()
    if writer_highlight is not None:
        writer_highlight.release()
    pass2_elapsed = time.time() - pass2_t0
    print(f"Pass 2 complete -- took {format_duration(pass2_elapsed)}.")

    total_elapsed = time.time() - processing_t0
    present_frames = sum(1 for frame_idx in range(n_frames) if per_frame_dets.get(frame_idx) and
                          any(d["track_id"] in staff_tracks for d in per_frame_dets[frame_idx]))
    print()
    print("=" * 64)
    print("  DONE")
    print("=" * 64)
    print(f"  Video length:        {n_frames} frames ({n_frames / fps:.1f}s)")
    print(f"  Staff found in:      {present_frames} / {n_frames} frames "
          f"({present_frames / n_frames * 100:.1f}%)")
    print(f"  Total time taken:    {format_duration(total_elapsed)}  "
          f"(detect+track+score: {format_duration(pass1_elapsed)}, "
          f"render video: {format_duration(pass2_elapsed)})")
    print(f"  Results table:       {csv_path}  ({n_staff_rows} staff rows, "
          f"{n_interpolated_rows} of them gap-filled/interpolated)")
    print(f"  Annotated video:     {out_dir / 'annotated.mp4'}  "
          f"(green=staff, yellow=still needs review, orange=other)")
    print(f"  Reference image:     {out_dir / 'reference_crop.jpg'}")
    if wrote_trajectory:
        print(f"  Trajectory plot:     {trajectory_path}")
    if writer_highlight is not None:
        highlight_frames = len(highlight_frame_set)
        print(f"  Highlight clip:      {highlight_path}  "
              f"({highlight_frames} frames, {highlight_frames / fps:.1f}s of {n_frames / fps:.1f}s)")
    if bridged_tracks:
        print(f"  Fragments bridged:   {len(bridged_tracks)} short/marginal track(s) pulled "
              f"into the staff result as continuations of a confirmed sighting")
    if demoted_tracks:
        print(f"  Auto-rejected:       {len(demoted_tracks)} track(s) that conflicted with a "
              f"stronger simultaneous staff track  ({out_dir / 'auto_rejected_conflicts.csv'})")
    if review_rows:
        n_confirmed = sum(1 for *_, r in review_rows if r == "confirmed STAFF")
        n_rejected = sum(1 for *_, r in review_rows if r == "confirmed not staff")
        n_pending = len(review_rows) - n_confirmed - n_rejected
        print(f"  Review outcomes:     {n_confirmed} confirmed staff, {n_rejected} confirmed "
              f"not staff, {n_pending} still pending  ({review_csv_path})")
    print("=" * 64)


def parse_args():
    p = argparse.ArgumentParser(description="Round 2 staff identification pipeline.")
    p.add_argument("--video", default=None,
                    help="Path to the input video. Omit this to be guided through picking a "
                         "video and output folder interactively instead.")
    p.add_argument("--output-dir", default=None,
                    help="Folder to write results to. Omit together with --video to be asked "
                         "interactively; omit alone to default to 'output'.")
    p.add_argument("--ref-frame", type=int, default=None,
                    help="Skip interactive picker; frame index for the reference crop.")
    p.add_argument("--ref-box", default=None, help="x,y,w,h — required together with --ref-frame.")
    p.add_argument("--staff-threshold", type=float, default=0.5,
                    help="Median match score (Lab-distance-based by default; see --use-clip) "
                         "for a track to count as staff. Re-check against a new video's "
                         "per-track score printout rather than assuming this transfers.")
    p.add_argument("--use-clip", action="store_true",
                    help="Blend in CLIP cosine similarity (15%% weight). Off by default: "
                         "measured to be anti-correlated with the true match on this footage "
                         "(see dev_notes/LOG.md, 'Round 2') -- kept as an opt-in experiment, not because "
                         "it's expected to help.")
    p.add_argument("--min-track-seconds", type=float, default=0.3,
                    help="Tracks shorter than this are never counted as staff, to reject flicker.")
    p.add_argument("--skip-review", action="store_true",
                    help="Don't interactively confirm flagged possible-staff events (e.g. for "
                         "scripted/headless runs). They're left unresolved in "
                         "possible_staff_review.csv for manual follow-up instead.")
    p.add_argument("--max-review-events", type=int, default=None,
                    help="Cap how many possible-staff events are shown interactively (e.g. to "
                         "keep a time-boxed live demo moving). Any beyond this many are left "
                         "as 'needs manual review' in possible_staff_review.csv without a "
                         "popup. Omit for no cap.")
    p.add_argument("--min-walk-speed", type=float, default=6.0,
                    help="Minimum average centroid speed (px/frame) for a color-matching track "
                         "to count as staff. Restricts detections to confirmed walking events "
                         "rather than any high-scoring seated/stationary match, since this scene "
                         "has more than one similarly light-clothed person -- see dev_notes/LOG.md, 'Round 2'.")
    p.add_argument("--min-walk-range", type=float, default=40.0,
                    help="Minimum total positional range (px) a track must cover, alongside "
                         "--min-walk-speed, to count as a genuine walking event. Kept low "
                         "relative to --min-walk-speed -- speed is the more reliable signal, "
                         "this is just a floor against pure box jitter on a short track.")
    p.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics YOLO segmentation weights.")
    p.add_argument("--detect-conf", type=float, default=0.15,
                    help="Permissive detector confidence -- appearance matching, not detector "
                         "confidence, decides who's staff.")
    p.add_argument("--device", default=None, help="'cpu', 'cuda', etc. Auto-detected if omitted.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

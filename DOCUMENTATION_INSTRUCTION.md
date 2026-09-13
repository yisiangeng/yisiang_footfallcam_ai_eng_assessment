# DOCUMENTATION_INSTRUCTION.md

Outline given for the 1-2 page write-up deliverable, originally drafted as a single
`DOCUMENTATION.md` and later split into `DOCUMENTATION_FINAL.md` (the actual deliverable, this
outline applies to it) and `DOCUMENTATION_FULL.md` (an unabridged personal-reference copy, not
itself held to this outline's 2-page limit). Kept here as the source brief so it can be checked
against as the document evolves.

## Outline

1. Executive summary (5%)
   - My details: name, title of project, github repo link
   - Problem
   - Task
   - Solution

2. Assumptions (25%)
   - The assumptions used to make this model / solution

3. Methodology & Pipeline (30%)
   - Pipeline diagram with brief elaboration
   - Explain and justify the use of models (successful models and failed models)

4. Results (10%)
   - Evaluation metrics
   - and other

5. Steps to use (5%)
   - Explain using layman term, as if speaking to someone not from the technical background
   - How to use the solution
   - How to start by moving the video and navigating to terminal and typing the code

6. Challenges, Solutions, Limitations (25%)
   - Read the LOG.md and KNOWLEDGE.md and SOLUTION_PLAN.md to know the challenges faced during
     the building process and what was done to fix it
   - Write in the format of table
   - Explain briefly the limitations of the solution

## Notes

- Overall length: maximum 2 pages (be concise and straight to the point, use technical but
  simple language, not too bombastic).
- Weightage allocated for each part is in the brackets (feel free to adjust them).
- Feel free to add things you find important / worth flagging in the documentation.
- You may choose to add other content you find suitable for each section.

## Deviations from this outline

- Executive summary: the `Problem` and `Task` bullets were removed (kept only `Name`,
  `Project`, `Repo`, `Solution`) — the Problem/Task framing is redundant with what the Solution
  bullet and the rest of the document already cover, and the section is only worth 5% of the
  total.
- The single draft was later split into two files: `DOCUMENTATION_FULL.md` (the complete
  original, kept unabridged for personal reference) and `DOCUMENTATION_FINAL.md` (the actual
  deliverable). Only `DOCUMENTATION_FINAL.md` was then curated further, acting as an AI-engineer
  reviewer selecting what a hiring panel actually needs:
  - Assumptions: trimmed from 9 to 6 (cut the tracker-ID-trust, "doesn't teleport"
    gap-bridging, and static-camera assumptions as implementation nuance rather than
    decision-shaping; the remaining six were then rewritten as shorter, single sentences).
  - Challenges & Solutions table: trimmed from 10 to 7 rows (cut the YOLO-retest story, the
    reviewer-UX crop redesign, and the dark-reference-photo tip as process/tooling detail
    rather than core model-design decisions), then further to the 4 judged most serious as
    computer-vision problems and most pipeline-shaping: mask-aware ROI extraction, the CLIP
    rejection, the clothing-change/re-ID ceiling, and tracking fragmentation.
  - Known limitations: trimmed from 5 to 4 bullets (merged the two "identity-check gap" bullets
    into one).
  - The Results table's third row ("Identity-aware") was flagged as a candidate cut (no fair
    post-fix counterpart to compare it against) but this was left as an open recommendation,
    not yet acted on — see the file itself for the current row count.

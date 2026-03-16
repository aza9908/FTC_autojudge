# FTC DECODE 2025-2026 — Computer Vision Judge System

Real-time computer vision scoring system for the **FIRST Tech Challenge 2025-2026 DECODE™ presented by RTX** season.

Built for the **Nomadic Dragons** FTC team — used at the FTC World Championship in Houston, TX (April–May 2026).

## What It Does

Automatically tracks and scores purple and green artifacts during FTC DECODE matches using a camera feed:

- **Color detection** — identifies purple and green balls via HSV segmentation with live calibration
- **ArUco marker tracking** — detects robots and field zones via printed markers
- **Entry-based scoring** — scores artifacts only when they enter a goal zone (no phantom double-counting)
- **PATTERN assessment** — evaluates RAMP sequence against the MOTIF (GPP/PGP/PPG) at end of AUTO and MATCH
- **Foul system** — full catalog of MINOR (5pts) and MAJOR (15pts) fouls from the official Competition Manual
- **Match timer** — 30s AUTO + 8s transition + 2:00 TELEOP = 2:38 total with auto period transitions

## Official Scoring (Table 10-2)

| Action | Points |
|--------|--------|
| CLASSIFIED artifact | 3 |
| OVERFLOW artifact | 1 |
| PATTERN match (per artifact) | 2 |
| DEPOT artifact | 1 |
| LEAVE (robot off launch line) | 3 |
| BASE partially returned | 5 |
| BASE fully returned | 10 |
| Both robots fully returned bonus | +10 |
| MINOR FOUL (to opponent) | 5 |
| MAJOR FOUL (to opponent) | 15 |

## Quick Start

```bash
pip install opencv-python opencv-contrib-python numpy
python main.py
```

### Before First Match — Calibrate Colors

1. Press **`b`** → click on 3 different **purple** balls on screen (bright, shadow, mid-range)
2. Press **`n`** → click on 3 different **green** balls
3. Press **`d`** to verify detection masks look correct
4. Set MOTIF: press **`1`** (GPP), **`2`** (PGP), or **`3`** (PPG)
5. Press **`s`** to start the match

### Key Controls

| Key | Action |
|-----|--------|
| `1` / `2` / `3` | Set MOTIF (GPP / PGP / PPG) |
| `s` | Start match |
| `r` | Reset match |
| `p` | Print scores |
| `a` | Assess patterns now |
| `b` / `n` | Calibrate purple / green |
| `d` | Toggle HSV debug overlay |
| `[ ` / `]` | Adjust purple hue range |
| `F1`–`F4` | Manual score: Red P/G, Blue P/G |
| `z` / `x` | Undo last Red / Blue artifact |
| `4` / `5` | MINOR / MAJOR foul on Red |
| `6` / `7` | MINOR / MAJOR foul on Blue |
| `f` | Open specific foul rule menu |
| `F5`–`F12` | Endgame: LEAVE, BASE full/partial/bonus |
| Left-click | Add manual GOAL scoring zone |
| Right-click | Clear all zones |
| `q` | Quit and save match report |

## Foul Catalog

Includes 20+ rule codes from Section 11 of the Competition Manual:

- **G301** — Match delay
- **G403** — Movement during AUTO-TELEOP transition
- **G404** — Movement after match end
- **G408** — Controlling >3 artifacts
- **G416** — Launching outside launch zone
- **G417** — Contacting opponent gate
- **G418** — Meddling with ramp artifacts
- **G419** — Launching onto own ramp / into opponent goal
- **G420** — Deliberately impairing opponent robot
- **G421** — Pinning
- **G425** — Contact in opponent secret tunnel
- **G431** — Drive team member illegal contact

## Optional: Claude AI Integration

Set `ENABLE_CLAUDE_VALIDATION = True` and export `ANTHROPIC_API_KEY` to enable async AI review of scoring events:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The system sends each scoring event to Claude for a second opinion (non-blocking, runs in background thread).

## Requirements

- Python 3.8+
- OpenCV with ArUco module (`opencv-contrib-python`)
- NumPy
- Webcam or video source
- Optional: `anthropic` Python package for AI validation

## Tech Stack

- **OpenCV** — video capture, ArUco detection, HSV color segmentation, morphological operations
- **NumPy** — array math, distance calculations
- **ArUco 4x4_50** — marker dictionary for robot/zone tracking
- **Gaussian blur + morphology** — noise reduction for robust ball detection

## License

**© 2025-2026 Azamat Armanuly. All rights reserved.**

This software and all associated files are the intellectual property of Azamat Armanuly. Unauthorized copying, modification, distribution, or use of this software without explicit written permission from the author is strictly prohibited.

For licensing inquiries, contact via GitHub: [@aza9908](https://github.com/aza9908)

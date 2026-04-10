# Sports Multi-Object Tracker

**Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage**

## 🚀 Live Demo

Try the app directly in your browser (no setup required):

👉 https://huggingface.co/spaces/priyanshuxsinha/sport-ids-tracker

### How to Use
1. Upload a sports video (`.mp4` format recommended)
2. Adjust parameters (optional):
   - Detection Confidence
   - NMS IoU Threshold
   - Max Resolution
3. Click **Run Tracker**
4. View:
   - Tracked players with unique IDs
   - Total IDs detected
   - Frame stats

⚡ Processing may take a few seconds depending on video size.

---

## Pipeline overview

```
Video input
    │
    ▼
Frame extraction (stride=1)
    │
    ▼
YOLOv8n detector  ──→  person bounding boxes (conf=0.35)
    │
    ▼
BoT-SORT tracker  ──→  Kalman filter + CMC + two-stage assignment
    │
    ▼
OSNet Re-ID gallery  ──→  256-dim EMA embeddings, cosine re-association
    │
    ▼
Annotated output video + analytics
(trajectories · heatmap · count chart)
```

| Component | Choice | Why |
|---|---|---|
| Detector | YOLOv8n | Real-time speed, strong person mAP, built-in tracker support |
| Tracker | BoT-SORT | Kalman filter + camera motion compensation + two-stage match |
| Re-ID | OSNet-x0.25 | 256-dim embeddings, lightweight, robust across similar appearances |
| Fallback Re-ID | HSV colour histogram | Runs with zero extra deps if torchreid unavailable |

**Public video used:**  
People Playing Football — https://www.pexels.com/video/people-playing-football-6079618/  
*(Free, no login required)*

---

## Repository structure

```
sports_tracker/
├── src/
│   ├── tracker.py          # Main pipeline class + CLI entry point
│   ├── reid_model.py       # OSNet wrapper (auto-fallback to histogram)
│   └── analytics.py        # Trajectory, heatmap, count chart helpers
├── demo_notebook.ipynb     # Full walkthrough notebook
├── botsort.yaml            # BoT-SORT tracker configuration
├── requirements.txt
├── reports/
│   └── technical_report.md
└── outputs/                # Created automatically on first run
```

---

## Quick start

### 1. Clone / unzip

```bash
cd sports_tracker
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are auto-downloaded on first run.

### 4. Install torchreid (optional but recommended)

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

If torchreid is not installed, the pipeline falls back to a colour-histogram Re-ID
baseline automatically — no code changes needed.

---

## Usage

### Option A — Jupyter Notebook

```bash
jupyter notebook demo_notebook.ipynb
```

Run cells top to bottom. The notebook runs the full pipeline and displays the heatmap,
trajectory image, and count chart inline.

### Option B — Command line

```bash
python src/tracker.py \
    --video   input_video.mp4 \
    --output  outputs/tracked.mp4 \
    --weights yolov8n.pt \
    --stride  1 \
    --conf    0.35 \
    --traj-len 15
```

### Option C — Live demo

No setup required. Upload any short sports clip directly at:  
**https://huggingface.co/spaces/priyanshuxsinha/sport-ids-tracker**

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--output` | `output.mp4` | Output annotated video path |
| `--weights` | `yolov8n.pt` | YOLOv8 weights (auto-downloaded) |
| `--tracker` | `botsort.yaml` | Tracker config file |
| `--conf` | 0.35 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--stride` | 1 | Process every Nth frame (1 = all frames, no flicker) |
| `--max-dim` | 1280 | Resize longest edge before detection |
| `--out-width` | 1280 | Output video width |
| `--traj-len` | 15 | Trajectory tail length in frames |

---

## Outputs

| File | Description |
|---|---|
| `output_final.mp4` | Annotated video — bounding boxes, IDs, trajectory tails |
| `output_final.json` | Per-frame stats + summary (total IDs, active counts) |
| `outputs/heatmap.jpg` | Subject density heatmap overlaid on first frame |
| `outputs/trajectories.jpg` | All track paths overlaid on first frame |
| `outputs/count_over_time.png` | Active subject count chart over time |

---

## Assumptions

1. Input video contains human subjects. For other object classes, change `classes=[0]`
   in `tracker.py` to the appropriate COCO class ID.
2. Camera frame rate is at least 15 fps. Below this the Kalman predictions degrade and
   the Re-ID window may need tuning.
3. Subjects occupy at least ~20×20 pixels in the processed frame. Smaller subjects may
   not be reliably detected by YOLOv8n.
4. On macOS, OpenCV must write with the `avc1` codec. The tracker sets this automatically.

---

## Limitations

- ID switches still occur in very dense occlusion (2+ players fully overlapping for >2s).
- Camera cuts may cause Re-ID to fail if the scene changes dramatically.
- Processing speed on CPU: ~8–15 fps at 1280px. A CUDA GPU gives >60 fps.
- `torchreid` has a complex dependency tree on some platforms — histogram fallback
  activates automatically if it fails to import.

---

## Dependencies

```
ultralytics >= 8.1.0   # YOLOv8 + BoT-SORT
torch >= 2.0.0
torchvision >= 0.15.0
opencv-python >= 4.8.0
numpy >= 1.24.0
lap >= 0.5.12          # required by BoT-SORT
torchreid              # optional — deep Re-ID backbone
matplotlib             # optional — count chart in notebook
yt-dlp                 # optional — video download utility
```

---

## References

- Aharon et al. (2022) — BoT-SORT: Robust Associations for Multi-Pedestrian Tracking
- Zhou et al. (2019) — OSNet: Omni-Scale Feature Learning for Person Re-Identification
- Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics

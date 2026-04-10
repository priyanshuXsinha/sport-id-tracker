
**Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage**

---

## Architecture at a glance

```
Video → Frame extraction (stride=1) → YOLOv8n detector → BoT-SORT tracker
                                                       ↓
                                           OSNet Re-ID gallery
                                                       ↓
                                      Annotated output video + analytics
```

| Component | Choice | Why |
|---|---|---|
| Detector | YOLOv8n | Fast, accurate, ships with tracker integration |
| Tracker | BoT-SORT | Kalman + CMC handles camera motion; two-stage match recovers occluded subjects |
| Re-ID | OSNet-x0.25 | 256-dim embeddings; lightweight; robust to jersey similarity |

**Public video used:**  
People Playing Football — [https://www.pexels.com/video/people-playing-football-6079618/](https://www.pexels.com/video/people-playing-football-6079618/)  

## Repository structure

```
sports_tracker/
├── src/
│   ├── tracker.py       # Main pipeline class + CLI entry point
│   ├── reid_model.py    # OSNet wrapper (falls back to histogram if torchreid missing)
│   └── analytics.py     # Trajectory, heatmap, count chart helpers
├── demo_notebook.ipynb  # Full walkthrough notebook
├── botsort.yaml         # BoT-SORT tracker configuration
├── requirements.txt
├── reports/
│   └── technical_report.md
└── outputs/             # Created automatically on first run
```

---

## Installation

### 1. Clone / unzip the project

```bash
cd sports_tracker
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are auto-downloaded on first run.

#### Installing torchreid (Re-ID — optional but recommended)

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

If torchreid is not installed, the pipeline automatically falls back to a colour-histogram
Re-ID baseline (weaker but fully functional).

---

## Usage

### Option A — Jupyter Notebook (recommended for first run)

```bash
jupyter notebook demo_notebook.ipynb
```

Run cells top to bottom. The notebook downloads the video, runs the full pipeline,
and displays the heatmap, trajectory image, and count chart inline.

### Option B — Command line

```bash
# Download source video
yt-dlp -f "bestvideo[ext=mp4][height<=720]+bestaudio/best[height<=720]" \
       -o input_video.mp4 \
       "https://www.pexels.com/video/people-playing-football-6079618/"

# Run tracker
python src/tracker.py \
    --video   input_video.mp4 \
    --output  outputs_final/tracked.mp4 \
    --weights yolov8n.pt \
    --stride  2 \
    --conf    0.35
```

#### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Input video path |
| `--output` | `output.mp4` | Output annotated video |
| `--weights` | `yolov8n.pt` | YOLOv8 weights (auto-downloaded) |
| `--tracker` | `botsort.yaml` | Tracker config |
| `--conf` | 0.35 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--stride` | 2 | Process every Nth frame |
| `--max-dim` | 1280 | Resize longest edge before detection |
| `--out-width` | 1280 | Output video width |
| `--traj-len` | 60 | Trajectory history in frames |

---

## Outputs

After running, the `outputs/` directory will contain:

| File | Description |
|---|---|
| `tracked.mp4` | Annotated video with bounding boxes, IDs, trajectory tails |
| `tracked.json` | Per-frame stats + summary (total IDs, counts) |
| `heatmap.jpg` | Subject density heatmap overlaid on first frame |
| `trajectories.jpg` | All track paths overlaid on first frame |
| `count_over_time.png` | Active subject count chart (notebook only) |

---

## Assumptions

1. The input video contains human subjects (person class). For vehicles or other objects,
   change `classes=[0]` in `tracker.py` to the appropriate COCO class ID(s).
2. Camera frame rate is at least 15 fps. At lower rates the Kalman filter predictions
   become less reliable and the Re-ID window may need tuning.
3. Subjects are at least ~20×20 pixels in the processed frame. Subjects further away may
   not be detected reliably by YOLOv8n.
4. The video is publicly accessible and can be downloaded with yt-dlp or provided as a
   local file.

---

## Limitations

- **ID switches** still occur in very dense occlusion (2+ players fully overlapping for >2s).
- **Camera cuts** may cause Re-ID to fail if the scene changes dramatically (different area of field).
- Processing speed on CPU: ~8–12 fps at 1280px with stride=2. A GPU (any CUDA device) gives >30 fps.
- `torchreid` has a complex dependency tree on some platforms. The histogram fallback is
  activated automatically if installation fails.

---

## Dependencies

```
ultralytics >= 8.1.0   # YOLOv8 + BoT-SORT
torch >= 2.0.0
torchvision >= 0.15.0
opencv-python >= 4.8.0
numpy >= 1.24.0
torchreid              # optional — Re-ID backbone
yt-dlp                 # optional — video download
matplotlib             # optional — count chart in notebook
```

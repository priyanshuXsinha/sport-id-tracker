"""
Analytics helpers:
  - TrajectoryTracker  : per-ID centroid history
  - HeatmapBuilder     : accumulates spatial density of all centroids
  - CountTimeline      : active subject count per frame
  - save_heatmap_image : export heatmap overlay
  - save_trajectory_image : export trajectory visualisation
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory tracking
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryTracker:
    """
    Maintains a sliding window of centroid positions per track ID.
    """

    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self._tracks: Dict[int, deque] = {}

    def update(self, track_id: int, cx: int, cy: int):
        if track_id not in self._tracks:
            self._tracks[track_id] = deque(maxlen=self.max_len)
        self._tracks[track_id].append((cx, cy))

    def get(self, track_id: int) -> List[Tuple[int, int]]:
        return list(self._tracks.get(track_id, []))

    def all_tracks(self) -> Dict[int, List[Tuple[int, int]]]:
        return {tid: list(hist) for tid, hist in self._tracks.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap
# ──────────────────────────────────────────────────────────────────────────────

class HeatmapBuilder:
    """
    Accumulates a Gaussian-blurred density map of subject centroids.
    """

    def __init__(self, blur_ksize: int = 35):
        self._map  = None
        self._blur = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

    def init(self, h: int, w: int):
        self._map = np.zeros((h, w), dtype=np.float32)
        self._h   = h
        self._w   = w

    def update(self, cx: int, cy: int):
        if self._map is None:
            return
        cx = int(np.clip(cx, 0, self._w - 1))
        cy = int(np.clip(cy, 0, self._h - 1))
        self._map[cy, cx] += 1.0

    def render(self, background: Optional[np.ndarray] = None,
               alpha: float = 0.55) -> np.ndarray:
        """
        Returns a colourmap overlay (JET) blended onto background.
        If background is None, returns the colourmap on black.
        """
        blurred = cv2.GaussianBlur(self._map, (self._blur, self._blur), 0)
        max_val = blurred.max()
        if max_val < 1e-6:
            norm = np.zeros_like(blurred, dtype=np.uint8)
        else:
            norm = (blurred / max_val * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        if background is None:
            return coloured

        bg = cv2.resize(background, (self._w, self._h))
        return cv2.addWeighted(bg, 1 - alpha, coloured, alpha, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Count timeline
# ──────────────────────────────────────────────────────────────────────────────

class CountTimeline:
    """Records active subject count per frame for plotting."""

    def __init__(self):
        self.frames: List[int] = []
        self.counts: List[int] = []

    def record(self, frame_idx: int, count: int):
        self.frames.append(frame_idx)
        self.counts.append(count)


# ──────────────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_heatmap_image(heatmap: HeatmapBuilder,
                       output_path: str,
                       first_frame: Optional[np.ndarray] = None):
    img = heatmap.render(background=first_frame)
    cv2.imwrite(output_path, img)
    print(f"[INFO] Heatmap saved → {output_path}")


def save_trajectory_image(traj_tracker: TrajectoryTracker,
                          frame_hw: Tuple[int, int],
                          id_colors: Dict[int, Tuple[int, int, int]],
                          output_path: str,
                          background: Optional[np.ndarray] = None):
    h, w = frame_hw
    if background is not None:
        canvas = cv2.resize(background.copy(), (w, h))
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for tid, pts in traj_tracker.all_tracks().items():
        color = id_colors.get(tid, (0, 255, 0))
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(canvas, pts[j - 1], pts[j], c, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 4, color, -1)

    cv2.imwrite(output_path, canvas)
    print(f"[INFO] Trajectory image saved → {output_path}")


def save_count_chart(timeline: CountTimeline, output_path: str):
    """
    Simple OpenCV-drawn bar chart of subject count over time.
    (No matplotlib dependency required.)
    """
    if not timeline.frames:
        return

    W, H   = 800, 300
    pad    = 40
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 245

    max_c  = max(timeline.counts) or 1
    n      = len(timeline.frames)
    bar_w  = max(1, (W - 2 * pad) // n)

    for i, (_, c) in enumerate(zip(timeline.frames, timeline.counts)):
        bar_h  = int((c / max_c) * (H - 2 * pad))
        x0     = pad + i * bar_w
        y0     = H - pad - bar_h
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w - 1, H - pad),
                      (70, 130, 200), -1)

    # axes
    cv2.line(canvas, (pad, H - pad), (W - pad, H - pad), (50, 50, 50), 1)
    cv2.line(canvas, (pad, pad),     (pad, H - pad),     (50, 50, 50), 1)
    cv2.putText(canvas, "Active subjects over time",
                (pad, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    cv2.putText(canvas, f"max={max_c}",
                (W - pad - 80, pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imwrite(output_path, canvas)
    print(f"[INFO] Count chart saved → {output_path}")

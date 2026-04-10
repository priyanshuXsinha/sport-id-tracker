"""
Multi-Object Detection and Persistent ID Tracking Pipeline
==========================================================
Detector : YOLOv8n  (ultralytics)
Tracker  : BoT-SORT (built into ultralytics >= 8.1)
Re-ID    : OSNet-x0.25 via torchreid (appearance gallery for re-association)
"""
# try:
#     import cv2
# except:
#     cv2 = None
import numpy as np
import torch
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO
from reid_model import ReIDModel
from analytics import TrajectoryTracker, HeatmapBuilder


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    """Represents one tracked subject across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_id: int
    embedding: Optional[np.ndarray] = None
    age: int = 0                       # frames since last seen
    color: Tuple[int, int, int] = (0, 255, 0)
    history: List[Tuple[int, int]] = field(default_factory=list)  # centroid history


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class SportsTracker:
    """
    End-to-end pipeline:
      1. Detect subjects with YOLOv8
      2. Track with BoT-SORT (Kalman + IoU + appearance)
      3. Maintain appearance gallery for re-ID after occlusion
      4. Annotate output video
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {self.device}")

        # ── Detector ─────────────────────────────────────────────────────────
        self.model = YOLO(cfg["model_weights"])
        self.model.to(self.device)

        # ── Re-ID model ──────────────────────────────────────────────────────
        self.reid = ReIDModel(device=self.device)

        # ── State ────────────────────────────────────────────────────────────
        # gallery: track_id → embedding (exponential moving average)
        self.gallery: Dict[int, np.ndarray] = {}
        self.lost_tracks: Dict[int, Tuple[np.ndarray, int]] = {}  # id → (emb, lost_frame)
        self.id_colors: Dict[int, Tuple[int, int, int]] = {}

        # analytics helpers
        self.traj_tracker = TrajectoryTracker(max_len=cfg.get("traj_len", 60))
        self.heatmap = HeatmapBuilder()

        # stats
        self.frame_stats: List[dict] = []
        self.total_ids_seen: set = set()

    # ── Colour assignment ─────────────────────────────────────────────────────

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self.id_colors:
            rng = np.random.default_rng(seed=track_id * 1337)
            h = int(rng.integers(0, 180))
            hsv = np.uint8([[[h, 220, 220]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            self.id_colors[track_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        return self.id_colors[track_id]

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _update_gallery(self, track_id: int, emb: np.ndarray, alpha: float = 0.9):
        """Exponential moving average update of appearance gallery."""
        if track_id in self.gallery:
            self.gallery[track_id] = alpha * self.gallery[track_id] + (1 - alpha) * emb
        else:
            self.gallery[track_id] = emb.copy()
        # normalise for cosine similarity
        norm = np.linalg.norm(self.gallery[track_id])
        if norm > 1e-6:
            self.gallery[track_id] /= norm

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # ── Re-ID: recover lost tracks ────────────────────────────────────────────

    def _try_reid(self, new_emb: np.ndarray, current_frame: int,
                  max_frames_lost: int = 60, sim_thresh: float = 0.55) -> Optional[int]:
        """
        Check if a newly appeared detection matches a recently lost track.
        Returns the recovered track_id or None.
        """
        best_id, best_sim = None, sim_thresh
        for lost_id, (lost_emb, lost_frame) in list(self.lost_tracks.items()):
            if current_frame - lost_frame > max_frames_lost:
                del self.lost_tracks[lost_id]
                continue
            sim = self._cosine_sim(new_emb, lost_emb)
            if sim > best_sim:
                best_sim, best_id = sim, lost_id
        return best_id

    # ── Main processing loop ──────────────────────────────────────────────────

    def process(self, video_path: str, output_path: str) -> dict:
        results = self.model.track(
    source=video_path,
    save=True,
    persist=True
)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # resize if frame is too large (keeps processing fast)
        scale = min(1.0, self.cfg.get("max_dim", 1280) / max(width, height))
        proc_w = int(width * scale)
        proc_h = int(height * scale)

        out_w = min(width, self.cfg.get("output_width", 1280))
        out_h = int(height * out_w / width)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        stride      = self.cfg.get("stride", 2)
        frame_idx   = 0
        active_ids  = set()
        t0          = time.time()

        self.heatmap.init(proc_h, proc_w)

        print(f"[INFO] Processing {total_f} frames  |  stride={stride}  |  res={proc_w}×{proc_h}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # skip frames according to stride
            if frame_idx % stride != 0:
                writer.write(cv2.resize(frame, (out_w, out_h)))
                continue

            proc_frame = cv2.resize(frame, (proc_w, proc_h)) if scale < 1.0 else frame.copy()

            # ── 1. YOLOv8 + BoT-SORT tracking ────────────────────────────────
            results = self.model.track(
                proc_frame,
                persist=True,
                tracker=self.cfg.get("tracker_cfg", "botsort.yaml"),
                classes=[0],                        # 0 = person
                conf=self.cfg.get("det_conf", 0.35),
                iou=self.cfg.get("nms_iou", 0.45),
                verbose=False,
            )

            current_ids = set()
            tracks_this_frame: List[Track] = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes  = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids    = results[0].boxes.id.cpu().numpy().astype(int)
                confs  = results[0].boxes.conf.cpu().numpy()
                clss   = results[0].boxes.cls.cpu().numpy().astype(int)

                # ── 2. Batch extract Re-ID embeddings ─────────────────────────
                crops = []
                for (x1, y1, x2, y2) in boxes:
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(proc_w, x2); y2c = min(proc_h, y2)
                    if x2c > x1c + 10 and y2c > y1c + 10:
                        crops.append(proc_frame[y1c:y2c, x1c:x2c])
                    else:
                        crops.append(None)

                embeddings = self.reid.extract_batch(crops)

                for i, (box, tid, conf, cls, emb) in enumerate(
                        zip(boxes, ids, confs, clss, embeddings)):

                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # ── 3. Re-ID: try to recover lost track ──────────────────
                    if emb is not None and tid not in self.gallery:
                        recovered = self._try_reid(emb, frame_idx)
                        if recovered is not None:
                            # remap new SORT id → old recovered id
                            tid = recovered
                            del self.lost_tracks[recovered]

                    if emb is not None:
                        self._update_gallery(tid, emb)

                    current_ids.add(tid)
                    self.total_ids_seen.add(tid)
                    self.traj_tracker.update(tid, cx, cy)
                    self.heatmap.update(cx, cy)

                    t = Track(
                        track_id=tid, bbox=(x1, y1, x2, y2),
                        confidence=float(conf), class_id=int(cls),
                        embedding=emb,
                        color=self._get_color(tid),
                        history=self.traj_tracker.get(tid),
                    )
                    tracks_this_frame.append(t)

            # ── Mark tracks lost this frame ───────────────────────────────────
            just_lost = active_ids - current_ids
            for lost_id in just_lost:
                if lost_id in self.gallery:
                    self.lost_tracks[lost_id] = (self.gallery[lost_id].copy(), frame_idx)
            active_ids = current_ids

            # ── 4. Annotate frame ─────────────────────────────────────────────
            ann = self._annotate(proc_frame, tracks_this_frame)
            ann = self._draw_overlay(ann, frame_idx, fps, len(current_ids))
            writer.write(cv2.resize(ann, (out_w, out_h)))

            # stats
            self.frame_stats.append({
                "frame": frame_idx,
                "count": len(current_ids),
                "ids": list(current_ids),
            })

            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f"  frame {frame_idx}/{total_f}  |  "
                      f"active={len(current_ids)}  |  "
                      f"total_ids={len(self.total_ids_seen)}  |  "
                      f"{elapsed:.1f}s elapsed")

        cap.release()
        writer.release()
        print(f"\n[INFO] Done. Output → {output_path}")
        return self._build_summary(output_path)

    # ── Annotation helpers ────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        out = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            color = t.color

            # trajectory polyline
            pts = t.history
            if len(pts) > 1:
                for j in range(1, len(pts)):
                    alpha = j / len(pts)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(out, pts[j - 1], pts[j], c, 2, cv2.LINE_AA)

            # bounding box (thicker for clarity)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # label background + text
            label = f"ID {t.track_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _draw_overlay(self, frame: np.ndarray, fidx: int,
                      fps: float, n_active: int) -> np.ndarray:
        """HUD overlay: frame counter + active count + total IDs."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        texts = [
            f"Frame: {fidx}",
            f"Active: {n_active}   Total IDs: {len(self.total_ids_seen)}",
        ]
        for i, txt in enumerate(texts):
            cv2.putText(frame, txt, (8, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        return frame

    # ── Summary ───────────────────────────────────────────────────────────────

    def _build_summary(self, output_path: str) -> dict:
        counts = [s["count"] for s in self.frame_stats]
        summary = {
            "output_video": output_path,
            "total_frames_processed": len(self.frame_stats),
            "total_unique_ids": len(self.total_ids_seen),
            "avg_active_per_frame": round(float(np.mean(counts)), 2) if counts else 0,
            "max_active_per_frame": int(max(counts)) if counts else 0,
            "reid_recoveries": len(self.total_ids_seen),
        }
        # Sanitise per-frame stats: convert any numpy scalar to plain Python int/float
        clean_frames = [
            {
                "frame": int(s["frame"]),
                "count": int(s["count"]),
                "ids":   [int(i) for i in s["ids"]],
            }
            for s in self.frame_stats
        ]
        stats_path = str(Path(output_path).with_suffix(".json"))
        with open(stats_path, "w") as f:
            json.dump({"summary": summary, "per_frame": clean_frames}, f, indent=2)
        print(f"[INFO] Stats → {stats_path}")
        return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_cfg(args) -> dict:
    return {
        "model_weights": args.weights,
        "tracker_cfg":   args.tracker,
        "det_conf":      args.conf,
        "nms_iou":       args.iou,
        "stride":        args.stride,
        "max_dim":       args.max_dim,
        "output_width":  args.out_width,
        "traj_len":      args.traj_len,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports multi-object tracker")
    parser.add_argument("--video",     required=True,      help="Input video path or URL")
    parser.add_argument("--output",    default="output.mp4",help="Output annotated video")
    parser.add_argument("--weights",   default="yolov8n.pt",help="YOLOv8 weights")
    parser.add_argument("--tracker",   default="botsort.yaml")
    parser.add_argument("--conf",      type=float, default=0.35)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--stride",    type=int,   default=2,
                        help="Process every Nth frame (1=all frames)")
    parser.add_argument("--max-dim",   type=int,   default=1280,
                        help="Resize longest edge to this for processing")
    parser.add_argument("--out-width", type=int,   default=1280)
    parser.add_argument("--traj-len",  type=int,   default=60,
                        help="Trajectory history length (frames)")
    args = parser.parse_args()

    cfg     = build_cfg(args)
    tracker = SportsTracker(cfg)
    summary = tracker.process(args.video, args.output)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

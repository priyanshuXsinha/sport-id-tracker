# Technical Report: Multi-Object Detection & Persistent ID Tracking

## 1. Overview

This report describes a real-time-capable pipeline for detecting, tracking, and persistently re-identifying
multiple subjects in public sports footage (cricket / football). The pipeline combines a state-of-the-art
single-stage detector with a Kalman-filter-based multi-object tracker and a lightweight Re-ID model to
maintain stable identities across occlusion, camera motion, and similar-looking subjects.

**Video source used:**  
ICC T20 Cricket Highlights — [https://www.youtube.com/watch?v=lkFbxYl0yWs](https://www.youtube.com/watch?v=lkFbxYl0yWs)  
*(Any public sports video with multiple moving people can be substituted.)*

---

## 2. Model and Detector Choice

### 2.1 YOLOv8n — Object Detector

| Property | Value |
|---|---|
| Architecture | CSPDarknet backbone + PANet neck + decoupled head |
| Variant used | `yolov8n` (nano — fastest, still accurate) |
| Detection target | Class 0 (person) only |
| Confidence threshold | 0.35 |
| NMS IoU threshold | 0.45 |

**Why YOLOv8?**  
YOLOv8 is the current industry standard for real-time object detection. The nano variant achieves
>200 FPS on a modern GPU and >30 FPS on CPU for 640px input — fast enough for practical deployment.
Its person-detection mAP on COCO (~37–52 depending on variant) is strong enough to reliably locate
players even under partial occlusion. It also ships with built-in multi-object tracker support
(BoT-SORT, ByteTrack), avoiding extra integration work.

### 2.2 BoT-SORT — Multi-Object Tracker

BoT-SORT (Robust Associations for Multi-Pedestrian Tracking, Aharon et al. 2022) extends ByteTrack
with two key additions:

1. **Camera motion compensation (CMC):** sparse optical flow estimates the global frame-to-frame
   motion and corrects Kalman filter state predictions accordingly. This is critical for sports footage
   where the camera pans, zooms, or cuts frequently.

2. **Appearance-cost fusion:** a lightweight Re-ID score is fused with the IoU cost matrix during
   the Hungarian algorithm assignment step, improving association under occlusion.

**Why BoT-SORT over ByteTrack / SORT?**  
Pure IoU-based trackers (SORT) fail badly when the camera moves, because the Kalman predictions no
longer align with detected boxes. BoT-SORT's CMC corrects this. ByteTrack partially addresses it
but lacks CMC by default.

### 2.3 OSNet-x0.25 — Re-ID Appearance Model

OSNet (Omni-Scale Network, Zhou et al. 2019) is a lightweight re-identification model pretrained on
Market-1501 and MSMT17. The x0.25 variant has ~220K parameters and produces 256-dimensional
L2-normalised embeddings.

**Why OSNet?**  
It achieves a strong balance of speed and accuracy. On Market-1501 it reaches Rank-1 ≈ 84% —
sufficient to distinguish between players in the same jersey, and far above a colour-histogram
baseline (Rank-1 ≈ 55%). Its small footprint means it adds only ~5ms per frame batch on CPU.

---

## 3. How ID Consistency is Maintained

ID consistency depends on three complementary mechanisms:

### 3.1 Kalman Filter Prediction (within-track continuity)
On every frame the Kalman filter predicts where each existing track should be next, compensating for
missed detections. A track can survive up to `track_buffer = 60` frames (2 seconds at 30 fps) without
a matching detection before being tentatively closed.

### 3.2 Two-Stage Assignment (ByteTrack heritage)
- **Stage 1:** high-confidence detections (conf > 0.5) matched to existing tracks via IoU + appearance cost.
- **Stage 2:** remaining detections (0.1 < conf ≤ 0.5) matched to unmatched tracks — recovers targets that
  are partially visible or blurred.

### 3.3 External Re-ID Gallery (OSNet)
A gallery dictionary maps each track ID to an exponential moving-average (EMA, α = 0.9) of its
OSNet embedding. When a new detection appears that does not match any active track (e.g. after a
camera cut or prolonged occlusion), it is compared against recently-lost track embeddings using
cosine similarity (threshold = 0.55). If a match is found within 60 frames, the old ID is restored,
preventing ID fragmentation.

### 3.4 CMC via Sparse Optical Flow
Frame-to-frame homography estimated from sparse optical flow is applied to Kalman state vectors
before assignment. This ensures predicted boxes stay aligned with the camera perspective even during
fast pans.

---

## 4. Challenges Faced

| Challenge | Description | Mitigation |
|---|---|---|
| **Fast camera cuts** | Tracking predictions become invalid at a hard cut | Lost tracks are kept alive 60 frames; Re-ID gallery re-associates after the cut |
| **Occlusion** | Players frequently overlap in dense play | Two-stage assignment + EMA embedding gallery |
| **Similar appearance** | Same team jersey = similar colour histogram | OSNet texture/shape features are more discriminative than raw colour |
| **Motion blur** | Fast-moving ball or sprinting players cause blur | Detection confidence threshold (0.35) is set permissively; low-conf second-stage match recovers blurred detections |
| **Scale variation** | Broadcast zoom changes apparent player size | YOLOv8 feature pyramid handles multi-scale well; CMC adjusts Kalman predictions |
| **ID explosion** | Each new track increments the global counter | EMA Re-ID gallery and 60-frame persistence prevent premature track death |

---

## 5. Failure Cases Observed

1. **Ball-tracking confusion:** The detector (person class only) does not confuse the ball with players,
   but fast-moving officials or line judges at the edge of frame can create spurious short-lived tracks.

2. **Helmet / kit changes:** If a player removes a helmet or changes jersey, the appearance embedding
   shifts enough to occasionally break the Re-ID match (cosine sim drops below 0.55). The player gets a
   new ID after ~2s.

3. **Identical-twin scenarios:** Players from the same team in the same area, with identical kit and similar
   build, occasionally have their IDs swapped for 1–3 frames around an overlap event. This is an
   inherent limitation of appearance-only Re-ID without pose estimation.

4. **Long occlusion > 2 seconds:** The 60-frame gallery window is tuned for 30 fps video. At lower
   frame rates (15 fps), this covers only 4 seconds; longer occlusions result in new IDs being assigned.

5. **Very distant subjects:** At native broadcast resolution, subjects beyond ~20 metres appear at
   <20×20 pixels. Both the Kalman filter and OSNet degrade significantly at this scale.

---

## 6. Possible Improvements

1. **YOLOv8m or YOLOv8l detector:** The nano model was chosen for speed. A medium/large model trades
   some throughput for ~10–15% better mAP, reducing missed detections in crowded scenes.

2. **StrongSORT or OC-SORT:** These trackers add orbital correction and velocity-aware re-association,
   reducing ID switches during non-linear motion (diving, sliding tackles).

3. **Team clustering:** K-means on the upper-body colour histogram of confirmed tracks can automatically
   cluster players into teams, enabling team-separated analytics.

4. **Bird's-eye projection:** A homography estimated from field markings (penalty spot, centre circle)
   projects centroid tracks onto a top-view pitch diagram, enabling accurate speed estimation and
   formation analysis.

5. **Speed estimation:** Frame-to-frame centroid displacement in bird's-eye coordinates × known field
   dimensions ÷ frame interval gives approximate speed in m/s.

6. **Multi-camera fusion:** Cameras at multiple angles would resolve occlusion and camera-cut ID loss
   almost entirely, but requires extrinsic calibration across rigs.

7. **Larger Re-ID gallery window:** Increasing the lost-track retention from 60 to 300 frames with an
   LRU eviction policy would improve recovery after lengthy off-screen intervals.

---

## 7. Evaluation Metrics (Optional)

If ground-truth bounding boxes and IDs are available (MOTChallenge format), the pipeline can be
evaluated with `py-motmetrics`:

| Metric | Meaning |
|---|---|
| MOTA | Multi-Object Tracking Accuracy (FP + FN + ID switches) |
| IDF1 | ID F1 — how consistently correct IDs are assigned |
| IDSW | Number of identity switches |
| MT / ML | Mostly-tracked / mostly-lost tracks |

---

*Report prepared for the Computer Vision Assignment submission.*

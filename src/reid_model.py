"""
Re-ID feature extractor using OSNet-x0.25 via torchreid.
Provides lightweight 256-dim appearance embeddings for each detection crop.

If torchreid is unavailable, falls back to a colour-histogram baseline
so the rest of the pipeline still runs (with weaker Re-ID).
"""

import cv2
import numpy as np
import torch
from typing import List, Optional

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("[WARN] torchreid not found – using colour-histogram fallback for Re-ID.")


# ──────────────────────────────────────────────────────────────────────────────
# OSNet wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ReIDModel:
    """
    Thin wrapper around OSNet-x0.25.
    Exposes a single `extract_batch(crops)` method.
    """

    # Target input size required by OSNet
    INPUT_H = 256
    INPUT_W = 128

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model  = None
        self._build()

    def _build(self):
        if not TORCHREID_AVAILABLE:
            return
        try:
            self.model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=1000,      # pretrained on Market-1501 / MSMT17
                pretrained=True,
            )
            self.model.eval()
            self.model.to(self.device)
            print("[INFO] OSNet-x0.25 loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load OSNet: {e}  – using fallback.")
            self.model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_batch(self, crops: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """
        Given a list of BGR image crops (or None for invalid crops),
        return a list of L2-normalised 256-dim numpy embeddings (or None).
        """
        if self.model is not None and TORCHREID_AVAILABLE:
            return self._extract_osnet(crops)
        return self._extract_histogram(crops)

    # ── OSNet extraction ───────────────────────────────────────────────────────

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        img = cv2.resize(crop, (self.INPUT_W, self.INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    def _extract_osnet(self, crops: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        results = []
        valid_tensors, valid_indices = [], []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                valid_tensors.append(self._preprocess(crop))
                valid_indices.append(i)
            else:
                results.append(None)

        if not valid_tensors:
            return results

        batch = torch.cat(valid_tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)          # (N, 512) for osnet_x0_25 → usually 256

        feats = feats.cpu().numpy()
        # L2 normalise
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-6)
        feats = feats / norms

        emb_iter = iter(feats)
        for i in range(len(crops)):
            if i in valid_indices:
                results.append(next(emb_iter))
            # None already appended for invalid crops above

        return results

    # ── Colour-histogram fallback ──────────────────────────────────────────────

    def _extract_histogram(self, crops: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """
        48-bin HSV histogram per channel → 144-dim vector.
        Weaker than deep features but zero extra deps.
        """
        results = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(None)
                continue
            hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = []
            for ch in range(3):
                h = cv2.calcHist([hsv], [ch], None, [48], [0, 256])
                hist.append(h.flatten())
            vec  = np.concatenate(hist).astype(np.float32)
            norm = np.linalg.norm(vec)
            results.append(vec / norm if norm > 1e-6 else vec)
        return results

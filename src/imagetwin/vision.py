import hashlib
import io
import json
import threading
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings

ENCODER_LAYER = "onnx_node!mobilenetv20_features_pool0_fwd"


def normalize(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise ValueError("Only PNG and JPEG are supported")
                if (
                    source.width * source.height > settings.max_pixels
                    or source.width < 32
                    or source.height < 32
                ):
                    raise ValueError("Image dimensions are outside the allowed range")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        # EXIF и прочие метаданные не нужны для сравнения; сохраняем только пиксели.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        pixels = np.array(image)
        pixel_sha = hashlib.sha256(str(image.size).encode() + pixels.tobytes()).hexdigest()
        return content, pixel_sha, image.width, image.height
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("Invalid or oversized image") from error


class Encoder:
    def __init__(self):
        root = Path(settings.model_dir)
        manifest = json.loads((root / "source.json").read_text())
        path = root / manifest["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Encoder checksum mismatch")
        self.version = "mobilenet-v2-" + manifest["sha256"][:12]
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.lock = threading.Lock()
        cv2.setNumThreads(1)

    def extract(self, data):
        rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)[:8, :8].flatten()
        bits = dct > np.median(dct[1:])
        bits[0] = False
        phash = "".join("1" if bit else "0" for bit in bits)
        resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255
        normalized = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        blob = normalized.transpose(2, 0, 1)[None, ...]
        # OpenCV Net хранит вход внутри себя, поэтому один экземпляр защищаем блокировкой.
        with self.lock:
            self.net.setInput(blob)
            vector = self.net.forward(ENCODER_LAYER).reshape(-1).copy()
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scale = min(1, 640 / max(gray.shape))
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        keypoints, descriptors = cv2.ORB_create(nfeatures=600).detectAndCompute(gray, None)
        points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
        points /= np.array([gray.shape[1], gray.shape[0]], np.float32)
        return {
            "phash": phash,
            "vector": vector.tolist(),
            "points": points.tolist(),
            "descriptors": descriptors.tobytes() if descriptors is not None else b"",
            "encoder_version": self.version,
            "contrast": float(gray.std()),
        }


@lru_cache(maxsize=1)
def encoder():
    return Encoder()


def compare(first, second):
    distance = sum(a != b for a, b in zip(first["phash"], second["phash"], strict=True))
    cosine = float(np.clip(np.dot(first["vector"], second["vector"]), -1, 1))
    descriptors = [
        np.frombuffer(item["descriptors"], dtype=np.uint8).reshape(-1, 32)
        for item in [first, second]
    ]
    inliers = 0
    ratio = 0.0
    coverage = 0.0
    if min(map(len, descriptors)) >= 12:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors[0], descriptors[1], k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        if len(matches) >= 12:
            a = np.array([first["points"][m.queryIdx] for m in matches], np.float32)
            b = np.array([second["points"][m.trainIdx] for m in matches], np.float32)
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, 0.012, maxIters=2000, confidence=0.995)
            if mask is not None:
                selected = mask.ravel().astype(bool)
                inliers = int(selected.sum())
                ratio = inliers / len(matches)
                if inliers >= 4:
                    coverage = min(
                        float(cv2.contourArea(cv2.convexHull(a[selected]))),
                        float(cv2.contourArea(cv2.convexHull(b[selected]))),
                    )
    near_hash = distance <= 8 and cosine >= 0.92 and min(first["contrast"], second["contrast"]) >= 5
    geometric = inliers >= 12 and ratio >= 0.5 and coverage >= 0.06
    return {
        "duplicate": bool(near_hash or geometric),
        "reason": "hash_and_embedding"
        if near_hash
        else "geometric_match"
        if geometric
        else "insufficient_evidence",
        "phash_distance": distance,
        "cosine_similarity": cosine,
        "geometric_inliers": inliers,
        "inlier_ratio": ratio,
        "matched_area": coverage,
    }

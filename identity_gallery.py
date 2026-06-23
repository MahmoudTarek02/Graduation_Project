from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from config import (
    ENTRANCE_DETECTOR_MODEL_PATH,
    ENTRANCE_MIN_DETECTION_CONFIDENCE,
    YOLO_PERSON_CLASS_ID,
)
from reid import REID


class YOLODetector:
    def __init__(
        self,
        model_path: str = ENTRANCE_DETECTOR_MODEL_PATH,
        conf_threshold: float = ENTRANCE_MIN_DETECTION_CONFIDENCE,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[list[float]]:
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls != YOLO_PERSON_CLASS_ID or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2, conf, cls])

        return detections


class IdentityGallery:
    def __init__(
        self,
        detector: YOLODetector | None = None,
        reid: REID | None = None,
    ):
        self.detector = detector or YOLODetector()
        self.reid = reid or REID()
        self.embeddings: dict[str, list[torch.Tensor]] = {}

    def enroll(self, person_id: str, image) -> torch.Tensor:
        if image is None:
            raise ValueError("Cannot enroll from an empty image")

        detections = self.detector.detect(image)
        if not detections:
            raise ValueError(f"No person detected for {person_id}")

        x1, y1, x2, y2, _, _ = self._largest_detection(detections)
        crop = self._crop(image, x1, y1, x2, y2)
        embedding = self.reid._feature(crop)

        self.embeddings.setdefault(person_id, []).append(embedding)
        return embedding

    def enroll_from_images(self, person_id: str, list_of_image_paths) -> list[torch.Tensor]:
        enrolled = []
        for image_path in list_of_image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            enrolled.append(self.enroll(person_id, image))
        return enrolled

    def save(self, path: str | Path) -> None:
        payload = {
            person_id: [embedding.numpy() for embedding in embeddings]
            for person_id, embeddings in self.embeddings.items()
        }

        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str | Path) -> "IdentityGallery":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)

        self.embeddings = {
            person_id: [torch.from_numpy(embedding) for embedding in embeddings]
            for person_id, embeddings in payload.items()
        }
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "IdentityGallery":
        return cls().load(path)

    @staticmethod
    def _largest_detection(detections: list[list[float]]) -> list[float]:
        return max(
            detections,
            key=lambda detection: (detection[2] - detection[0]) * (detection[3] - detection[1]),
        )

    @staticmethod
    def _crop(image, x1: float, y1: float, x2: float, y2: float):
        height, width = image.shape[:2]
        left = max(int(x1), 0)
        top = max(int(y1), 0)
        right = min(int(x2), width)
        bottom = min(int(y2), height)

        if right <= left or bottom <= top:
            raise ValueError("Detected person crop is empty")

        return image[top:bottom, left:right]

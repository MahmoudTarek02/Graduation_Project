from __future__ import annotations

import numpy as np
import torch

from config import FORWARD_DETECTOR_MODEL_PATH, REID_MATCH_THRESHOLD
from identity_gallery import IdentityGallery, YOLODetector


class IdentityMatcher:
    def __init__(
        self,
        gallery: IdentityGallery,
        detector: YOLODetector | None = None,
    ):
        self.gallery = gallery
        self.detector = detector or YOLODetector(model_path=FORWARD_DETECTOR_MODEL_PATH)

    def match(self, query_embedding):
        best_person_id = None
        best_score = None

        for person_id, stored_embeddings in self.gallery.embeddings.items():
            if not stored_embeddings:
                continue

            gallery_embeddings = torch.cat(stored_embeddings, dim=0)
            distance_matrix = self.gallery.reid.compute_distance(
                query_embedding,
                gallery_embeddings,
            )
            median_distance = float(np.median(distance_matrix))

            if best_score is None or median_distance < best_score:
                best_person_id = person_id
                best_score = median_distance

        if best_score is None:
            return None, None

        if best_score > REID_MATCH_THRESHOLD:
            return None, best_score

        return best_person_id, best_score

    def match_image(self, image):
        if image is None:
            raise ValueError("Cannot match from an empty image")

        detections = self.detector.detect(image)
        if not detections:
            raise ValueError("No person detected for identity matching")

        x1, y1, x2, y2, _, _ = IdentityGallery._largest_detection(detections)
        crop = IdentityGallery._crop(image, x1, y1, x2, y2)
        embedding = self.gallery.reid._feature(crop)
        return self.match(embedding)

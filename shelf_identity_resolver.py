from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    FORWARD_CAPTURE_BURST_FRAMES,
    FORWARD_DETECTOR_MODEL_PATH,
    FORWARD_DETECTION_CONFIDENCE_THRESHOLD,
    FORWARD_POSE_CONFIDENCE_THRESHOLD,
    FORWARD_POSE_MODEL_PATH,
    FORWARD_REACH_SELECTION_THRESHOLD,
    IDENTITY_GALLERY_PATH,
    POSE_LEFT_ELBOW_INDEX,
    POSE_LEFT_SHOULDER_INDEX,
    POSE_LEFT_WRIST_INDEX,
    POSE_RIGHT_ELBOW_INDEX,
    POSE_RIGHT_SHOULDER_INDEX,
    POSE_RIGHT_WRIST_INDEX,
)
from identity_gallery import IdentityGallery, YOLODetector
from identity_matcher import IdentityMatcher


@dataclass
class PoseCandidate:
    box: tuple[int, int, int, int]
    pose_confidence: float
    arm_side: str | None
    reach_score: float


@dataclass
class ShelfIdentityResolutionResult:
    event: dict
    source_label: str
    event_type: str
    frame_index: int
    person_count: int
    selection_method: str
    arm_side: str | None
    selected_box: tuple[int, int, int, int] | None
    person_id: str | None
    score: float | None


class ShelfIdentityResolver:
    PREVIEW_WINDOW_NAME = "Forward Camera Trigger Preview"

    def __init__(
        self,
        gallery: IdentityGallery | None = None,
        matcher: IdentityMatcher | None = None,
        person_detector: YOLODetector | None = None,
        pose_model: YOLO | None = None,
        capture_burst_frames: int = FORWARD_CAPTURE_BURST_FRAMES,
        live_frame_provider: Callable[[int | str | Path, int], list[np.ndarray]] | None = None,
        show_trigger_preview: bool = True,
    ):
        self.gallery = gallery or (matcher.gallery if matcher is not None else IdentityGallery.from_file(IDENTITY_GALLERY_PATH))
        self.person_detector = person_detector or YOLODetector(
            model_path=FORWARD_DETECTOR_MODEL_PATH,
            conf_threshold=FORWARD_DETECTION_CONFIDENCE_THRESHOLD,
        )
        self.matcher = matcher or IdentityMatcher(self.gallery, detector=self.person_detector)
        self.pose_model = pose_model or YOLO(FORWARD_POSE_MODEL_PATH)
        self.capture_burst_frames = capture_burst_frames
        self.live_frame_provider = live_frame_provider
        self.show_trigger_preview = show_trigger_preview

    def resolve_event(
        self,
        trigger_event: dict,
        source: int | str | Path,
    ) -> ShelfIdentityResolutionResult:
        source_path = str(source)
        frame_results: list[ShelfIdentityResolutionResult] = []

        if self.live_frame_provider is not None and isinstance(source, int):
            frames = self.live_frame_provider(source, self.capture_burst_frames)
            for frame_idx, frame in enumerate(frames):
                frame_results.append(
                    self.resolve_frame(
                        frame,
                        trigger_event=trigger_event,
                        frame_index=frame_idx,
                        source_label=source_path,
                    )
                )
        else:
            capture_source = source if isinstance(source, int) else source_path
            cap = cv2.VideoCapture(capture_source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open forward camera source: {source_path}")

            try:
                for frame_idx in range(self.capture_burst_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if self.show_trigger_preview:
                        preview = frame.copy()
                        self._draw_trigger_preview_overlay(
                            preview,
                            source_label=source_path,
                            frame_idx=frame_idx,
                        )
                        cv2.imshow(self.PREVIEW_WINDOW_NAME, preview)
                        cv2.waitKey(1)

                    frame_results.append(
                        self.resolve_frame(
                            frame,
                            trigger_event=trigger_event,
                            frame_index=frame_idx,
                            source_label=source_path,
                        )
                    )
            finally:
                cap.release()
                if self.show_trigger_preview:
                    try:
                        cv2.destroyWindow(self.PREVIEW_WINDOW_NAME)
                    except Exception:
                        pass

        if not frame_results:
            return self._empty_result(trigger_event, source_path, reason_frame_index=0)

        confident_results = [result for result in frame_results if result.score is not None]
        if confident_results:
            return min(confident_results, key=lambda result: result.score if result.score is not None else float("inf"))

        return frame_results[-1]

    def resolve_frame(
        self,
        frame,
        trigger_event: dict | None = None,
        frame_index: int = 0,
        source_label: str = "",
    ) -> ShelfIdentityResolutionResult:
        event = dict(trigger_event) if trigger_event else {}
        event_type = self._event_type(event)
        detections = self.person_detector.detect(frame)
        person_count = len(detections)

        if person_count == 0:
            return ShelfIdentityResolutionResult(
                event=event,
                source_label=source_label,
                event_type=event_type,
                frame_index=frame_index,
                person_count=0,
                selection_method="no_person",
                arm_side=None,
                selected_box=None,
                person_id=None,
                score=None,
            )

        if person_count == 1:
            selected_detection = detections[0]
            selection_method = "direct_reid"
            arm_side = None
        else:
            pose_candidate = self._select_pose_candidate(frame)
            if pose_candidate is None:
                return ShelfIdentityResolutionResult(
                    event=event,
                    source_label=source_label,
                    event_type=event_type,
                    frame_index=frame_index,
                    person_count=person_count,
                    selection_method="pose_unresolved",
                    arm_side=None,
                    selected_box=None,
                    person_id=None,
                    score=None,
                )

            selected_detection = [
                float(pose_candidate.box[0]),
                float(pose_candidate.box[1]),
                float(pose_candidate.box[2]),
                float(pose_candidate.box[3]),
                pose_candidate.pose_confidence,
                0,
            ]
            selection_method = "pose_reid"
            arm_side = pose_candidate.arm_side

        x1, y1, x2, y2, _, _ = selected_detection
        crop = IdentityGallery._crop(frame, x1, y1, x2, y2)
        embedding = self.gallery.reid._feature(crop)
        person_id, score = self.matcher.match(embedding)

        return ShelfIdentityResolutionResult(
            event=event,
            source_label=source_label,
            event_type=event_type,
            frame_index=frame_index,
            person_count=person_count,
            selection_method=selection_method,
            arm_side=arm_side,
            selected_box=(int(x1), int(y1), int(x2), int(y2)),
            person_id=person_id,
            score=score,
        )

    def _select_pose_candidate(self, frame) -> PoseCandidate | None:
        candidates = self._pose_candidates(frame)
        if not candidates:
            return None

        best_candidate = max(
            candidates,
            key=lambda candidate: candidate.pose_confidence * candidate.reach_score,
        )
        best_score = best_candidate.pose_confidence * best_candidate.reach_score
        if best_score < FORWARD_REACH_SELECTION_THRESHOLD:
            return None
        return best_candidate

    def _pose_candidates(self, frame) -> list[PoseCandidate]:
        results = self.pose_model.predict(
            frame,
            conf=FORWARD_POSE_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        candidates: list[PoseCandidate] = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
            keypoints_conf = None
            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.detach().cpu().numpy()

            for idx, box in enumerate(boxes):
                pose_confidence = float(confidences[idx]) if idx < len(confidences) else 0.0
                arm_side, reach_score = self._reach_score(
                    keypoints_xy[idx],
                    keypoints_conf[idx] if keypoints_conf is not None else None,
                    box,
                )
                candidates.append(
                    PoseCandidate(
                        box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        pose_confidence=pose_confidence,
                        arm_side=arm_side,
                        reach_score=reach_score,
                    )
                )

        return candidates

    def _reach_score(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray | None,
        box: np.ndarray,
    ) -> tuple[str | None, float]:
        best_side = None
        best_score = 0.0
        box_width = max(float(box[2] - box[0]), 1.0)
        box_height = max(float(box[3] - box[1]), 1.0)
        scale = max(box_width, box_height)

        for side, shoulder_idx, elbow_idx, wrist_idx in (
            ("left", POSE_LEFT_SHOULDER_INDEX, POSE_LEFT_ELBOW_INDEX, POSE_LEFT_WRIST_INDEX),
            ("right", POSE_RIGHT_SHOULDER_INDEX, POSE_RIGHT_ELBOW_INDEX, POSE_RIGHT_WRIST_INDEX),
        ):
            if not self._keypoints_visible(keypoints_conf, shoulder_idx, elbow_idx, wrist_idx):
                continue

            shoulder = keypoints_xy[shoulder_idx]
            elbow = keypoints_xy[elbow_idx]
            wrist = keypoints_xy[wrist_idx]
            angle_score = self._normalized_elbow_angle(shoulder, elbow, wrist)
            extension_score = min(1.0, float(np.linalg.norm(wrist - shoulder) / scale))
            vertical_bonus = max(0.0, float(shoulder[1] - wrist[1]) / box_height)
            score = 0.5 * angle_score + 0.3 * extension_score + 0.2 * min(1.0, vertical_bonus)

            if score > best_score:
                best_score = score
                best_side = side

        return best_side, best_score

    def _keypoints_visible(
        self,
        keypoints_conf: np.ndarray | None,
        shoulder_idx: int,
        elbow_idx: int,
        wrist_idx: int,
    ) -> bool:
        if keypoints_conf is None:
            return True
        return (
            float(keypoints_conf[shoulder_idx]) >= FORWARD_POSE_CONFIDENCE_THRESHOLD
            and float(keypoints_conf[elbow_idx]) >= FORWARD_POSE_CONFIDENCE_THRESHOLD
            and float(keypoints_conf[wrist_idx]) >= FORWARD_POSE_CONFIDENCE_THRESHOLD
        )

    def _normalized_elbow_angle(
        self,
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
    ) -> float:
        upper_arm = shoulder - elbow
        lower_arm = wrist - elbow
        upper_norm = float(np.linalg.norm(upper_arm))
        lower_norm = float(np.linalg.norm(lower_arm))
        if upper_norm == 0.0 or lower_norm == 0.0:
            return 0.0

        cosine = float(np.dot(upper_arm, lower_arm) / (upper_norm * lower_norm))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        angle_degrees = float(np.degrees(np.arccos(cosine)))
        return angle_degrees / 180.0

    def _event_type(self, event: dict) -> str:
        quantity = int(event.get("quantity", 0) or 0)
        if quantity > 0:
            return "taken"
        if quantity < 0:
            return "returned"
        return str(event.get("event_type", "unknown"))

    def _draw_trigger_preview_overlay(
        self,
        frame: np.ndarray,
        source_label: str,
        frame_idx: int,
    ) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (20, 20, 20), -1)
        lines = [
            f"Forward camera triggered | source {source_label}",
            f"Capture frame: {frame_idx + 1}/{self.capture_burst_frames}",
            "Resolving person for shelf event",
        ]
        for i, line in enumerate(lines):
            y = 28 + i * 24
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
            )

    def _empty_result(
        self,
        trigger_event: dict,
        source_label: str,
        reason_frame_index: int,
    ) -> ShelfIdentityResolutionResult:
        event = dict(trigger_event) if trigger_event else {}
        return ShelfIdentityResolutionResult(
            event=event,
            source_label=source_label,
            event_type=self._event_type(event),
            frame_index=reason_frame_index,
            person_count=0,
            selection_method="no_frame",
            arm_side=None,
            selected_box=None,
            person_id=None,
            score=None,
        )


if __name__ == "__main__":
    resolver = ShelfIdentityResolver()

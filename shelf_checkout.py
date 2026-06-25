from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    DEFAULT_LIVE_CAMERA_FPS,
    LIVE_SHELF_ANALYSIS_FRAMES,
    SHELF_ITEM_CONFIDENCE_THRESHOLD,
    SHELF_LIVE_BUFFER_FLUSH_FRAMES,
    SHELF_SHOW_PREVIEW,
)

@dataclass
class ShelfInteractionResult:
    source_label: str
    taken_events: list[dict]
    counts: dict[str, int]
    actor_counts: dict[str, int]


class ShelfCameraResolver:
    def __init__(self, shelf_camera_map: dict | None = None):
        self.shelf_camera_map = shelf_camera_map or {}

    def resolve(self, event: dict) -> int | str | None:
        return self.shelf_camera_map.get(event["zone_id"])


class ShelfCameraSession:
    def __init__(self, source: int | str | Path):
        self.source = source
        self.source_label = str(source)
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open shelf source: {self.source_label}")

    def ensure_open(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self.open()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ShelfItemMonitor:
    def __init__(
        self,
        model_path: str,
        pose_model_path: str = "",
        conf_threshold: float = SHELF_ITEM_CONFIDENCE_THRESHOLD,
        pose_conf_threshold: float = 0.35,
        live_analysis_frames: int = LIVE_SHELF_ANALYSIS_FRAMES,
        show_preview: bool = SHELF_SHOW_PREVIEW,
        hand_touch_margin: int = 60,
    ):
        self.model_path = model_path
        self.pose_model_path = pose_model_path
        self.conf_threshold = conf_threshold
        self.pose_conf_threshold = pose_conf_threshold
        self.live_analysis_frames = live_analysis_frames
        self.show_preview = show_preview
        self.hand_touch_margin = hand_touch_margin
        self._model = None
        self._class_names = {}
        self._sessions: dict[str, ShelfCameraSession] = {}
        self._hand_tracker = None

    def prepare_sources(self, sources: list[int | str | Path]) -> None:
        self._ensure_models()
        for source in sources:
            key = str(source)
            if key in self._sessions:
                continue

            session = ShelfCameraSession(source)
            session.open()
            self._sessions[key] = session
            print(f"[ShelfItemMonitor] prepared shelf source: {session.source_label}")

    def analyze_source(
        self,
        source: int | str | Path,
        trigger_event: dict | None = None,
        actor_side_hint: str = "unknown-side",
    ) -> ShelfInteractionResult:
        self._ensure_models()
        self._ensure_hand_tracker()
        session = self._get_or_create_session(source)
        cap = session.cap
        source_label = session.source_label

        if isinstance(source, int):
            for _ in range(SHELF_LIVE_BUFFER_FLUSH_FRAMES):
                ok, _ = cap.read()
                if not ok:
                    break

        all_frame_counts: list[Counter] = []
        settle_window = 10
        stable_count_history = deque(maxlen=settle_window)

        # Min and max frames to process dynamically
        min_frames = max(30, self.live_analysis_frames // 2)
        max_frames = max(120, self.live_analysis_frames * 2)

        frame_idx = 0
        while cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            detections = self._detect_items(frame)
            frame_counts = Counter()
            for det in detections:
                cls = int(det[5])
                class_name = self._class_names.get(cls, str(cls))
                frame_counts[class_name] += 1

            all_frame_counts.append(frame_counts)
            stable_count_history.append(frame_counts)

            # Check hand presence
            hands = self._hand_tracker.detect(frame)
            hand_present = len(hands) > 0

            # Check stability of counts
            counts_stable = len(stable_count_history) == settle_window and all(
                h == stable_count_history[0] for h in list(stable_count_history)[1:]
            )

            # Exit criteria
            if frame_idx >= min_frames:
                if not hand_present and counts_stable:
                    print(f"[ShelfItemMonitor] Settled and hand-free at frame {frame_idx + 1}")
                    break
                if frame_idx >= max_frames:
                    print(f"[ShelfItemMonitor] Reached max frame limit ({max_frames}) at frame {frame_idx + 1}")
                    break

            if self.show_preview:
                preview = frame.copy()
                status_text = "Interacting (hand present)" if hand_present else ("Settling..." if not counts_stable else "Stable")
                self._draw_preview_overlay(
                    preview=preview,
                    source_label=source_label,
                    frame_idx=frame_idx,
                    frame_limit=max_frames,
                    status_text=status_text,
                )
                self._draw_detections(preview, detections)
                cv2.imshow(f"Shelf Live Preview - source {source_label}", preview)
                cv2.waitKey(1)

            frame_idx += 1

        if self.show_preview:
            cv2.destroyWindow(f"Shelf Live Preview - source {source_label}")

        total_frames = len(all_frame_counts)
        # Establish stable "before" count from the start
        N_before = min(10, max(1, total_frames // 2))
        before_frames = all_frame_counts[:N_before]
        before_counts = self._get_stable_counts(before_frames)

        # Establish stable "after" count from the end (settled state)
        after_frames = all_frame_counts[-settle_window:] if total_frames >= settle_window else all_frame_counts
        after_counts = self._get_stable_counts(after_frames)

        taken_events: list[dict] = []
        all_classes = set(before_counts.keys()).union(after_counts.keys())

        for class_name in all_classes:
            b_count = before_counts.get(class_name, 0)
            a_count = after_counts.get(class_name, 0)
            delta = b_count - a_count

            if delta > 0:
                for _ in range(delta):
                    taken_events.append({
                        "class_name": class_name,
                        "track_id": -1,
                        "frame_taken": total_frames - 1,
                        "actor_side": actor_side_hint,
                        "person_label": f"person_{trigger_event.get('person_id')}" if trigger_event else "unknown-person",
                        "handedness": trigger_event.get("handedness", "unknown-hand") if trigger_event else "unknown-hand",
                        "quantity": 1,
                    })
            elif delta < 0:
                for _ in range(abs(delta)):
                    taken_events.append({
                        "class_name": class_name,
                        "track_id": -1,
                        "frame_taken": total_frames - 1,
                        "actor_side": actor_side_hint,
                        "person_label": f"person_{trigger_event.get('person_id')}" if trigger_event else "unknown-person",
                        "handedness": trigger_event.get("handedness", "unknown-hand") if trigger_event else "unknown-hand",
                        "quantity": -1,
                    })

        # Generate net counts report
        counts = {}
        for class_name in all_classes:
            b_count = before_counts.get(class_name, 0)
            a_count = after_counts.get(class_name, 0)
            net_change = b_count - a_count
            if net_change != 0:
                counts[class_name] = net_change

        actor_counts = {}
        if taken_events:
            actor_counts[actor_side_hint] = len(taken_events)

        return ShelfInteractionResult(
            source_label=source_label,
            taken_events=taken_events,
            counts=counts,
            actor_counts=actor_counts,
        )

    def _ensure_models(self) -> None:
        if self._model is None:
            model_path = Path(self.model_path).expanduser()
            if not model_path.exists():
                raise RuntimeError(f"Shelf-item model not found: {model_path}")
            self.model_path = str(model_path)
            self._model = YOLO(self.model_path)
            self._class_names = self._model.names
            print(f"[ShelfItemMonitor] loaded item model: {self.model_path}")

    def _ensure_hand_tracker(self) -> None:
        if self._hand_tracker is None:
            from hand_tracker import MediaPipeHandTracker
            self._hand_tracker = MediaPipeHandTracker()

    def _get_or_create_session(self, source: int | str | Path) -> ShelfCameraSession:
        key = str(source)
        session = self._sessions.get(key)
        if session is None:
            session = ShelfCameraSession(source)
            session.open()
            self._sessions[key] = session
        else:
            session.ensure_open()
        return session

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        if self._hand_tracker is not None:
            try:
                self._hand_tracker.close()
            except Exception:
                pass
            self._hand_tracker = None

    def _detect_items(self, frame: np.ndarray) -> np.ndarray:
        results = self._model(frame, conf=self.conf_threshold, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])

        if not boxes:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(boxes, dtype=np.float32)

    def _draw_detections(self, frame: np.ndarray, detections: np.ndarray) -> None:
        for row in detections:
            x1, y1, x2, y2 = map(int, row[:4])
            conf = float(row[4])
            cls = int(row[5])
            class_name = self._class_names.get(cls, str(cls))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 80), 2)
            cv2.putText(
                frame,
                f"{class_name} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    def _draw_preview_overlay(
        self,
        preview: np.ndarray,
        source_label: str,
        frame_idx: int,
        frame_limit: int | None,
        status_text: str = "Active",
    ) -> None:
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 115), (20, 20, 20), -1)
        total_text = f"{frame_idx + 1}/{frame_limit}" if frame_limit is not None else f"{frame_idx + 1}"
        lines = [
            f"Shelf source: {source_label} | Frame: {total_text}",
            f"Status: {status_text}",
            "Before/After count comparison active (no tracking)",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 24
            color = (50, 255, 50) if "Stable" in line else ((50, 150, 255) if "Settling" in line else (235, 235, 235))
            cv2.putText(
                preview,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

    @staticmethod
    def _get_stable_counts(frames_list: list[Counter]) -> dict[str, int]:
        if not frames_list:
            return {}
        totals = Counter()
        for f in frames_list:
            for cls, count in f.items():
                totals[cls] += count
        n = len(frames_list)
        return {cls: round(totals[cls] / n) for cls in totals}

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
from ultralytics import YOLO

from config import (
    DEFAULT_LIVE_CAMERA_FPS,
    SHELF_DISAPPEAR_THRESHOLD,
    SHELF_ITEM_CONFIDENCE_THRESHOLD,
    SHELF_ITEM_MODEL_PATH,
    SHELF_STABLE_FRAMES,
)


@dataclass
class _TrackState:
    class_name: str
    seen_frames: int = 0
    missing_frames: int = 0
    stable: bool = False
    taken: bool = False


class ItemDetector:
    def __init__(self):
        self.model = YOLO(SHELF_ITEM_MODEL_PATH)
        self.tracker = ByteTrack(
            track_buffer=SHELF_DISAPPEAR_THRESHOLD,
            frame_rate=DEFAULT_LIVE_CAMERA_FPS,
        )
        self.frame_idx = 0
        self.track_states: dict[int, _TrackState] = {}
        self.callbacks: list[Callable[[dict], None]] = []
        self.last_tracks: list[dict] = []

    def register_callback(self, fn: Callable[[dict], None]) -> None:
        self.callbacks.append(fn)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        detections = self._detect_items(frame)
        tracks = self.tracker.update(detections, frame)

        active_track_ids: set[int] = set()
        events: list[dict] = []
        self.last_tracks = []

        for row in tracks if tracks is not None else []:
            track = self._parse_track_row(row)
            active_track_ids.add(track["track_id"])
            self._mark_track_seen(track["track_id"], track["class_name"])
            self.last_tracks.append(track)

        for track_id, state in list(self.track_states.items()):
            if track_id in active_track_ids:
                continue

            state.missing_frames += 1
            if state.stable and not state.taken and state.missing_frames >= SHELF_DISAPPEAR_THRESHOLD:
                state.taken = True
                event = {
                    "class_name": state.class_name,
                    "track_id": track_id,
                    "frame_taken": self.frame_idx,
                }
                events.append(event)
                self._fire_callbacks(event)
                self.track_states.pop(track_id, None)
            elif not state.stable and state.missing_frames >= SHELF_DISAPPEAR_THRESHOLD:
                self.track_states.pop(track_id, None)

        self.frame_idx += 1
        return events

    def _detect_items(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            frame,
            conf=SHELF_ITEM_CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.detach().cpu().numpy().reshape(-1, 1)
        return np.hstack((xyxy, conf, cls)).astype(np.float32)

    def _parse_track_row(self, row: np.ndarray) -> dict:
        x1, y1, x2, y2 = map(int, row[:4])
        track_id = int(row[4])
        confidence = float(row[5])
        class_id = int(row[6])
        class_name = self._class_name(class_id)
        return {
            "box": (x1, y1, x2, y2),
            "class_name": class_name,
            "class_id": class_id,
            "confidence": confidence,
            "track_id": track_id,
        }

    def _mark_track_seen(self, track_id: int, class_name: str) -> None:
        state = self.track_states.get(track_id)
        if state is None:
            state = _TrackState(class_name=class_name)
            self.track_states[track_id] = state

        state.class_name = class_name
        state.seen_frames += 1
        state.missing_frames = 0
        if state.seen_frames >= SHELF_STABLE_FRAMES:
            state.stable = True

    def _class_name(self, class_id: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _fire_callbacks(self, event: dict) -> None:
        for callback in self.callbacks:
            callback(event)

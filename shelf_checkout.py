from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from boxmot import ByteTrack
from ultralytics import YOLO

from config import (
    DEFAULT_LIVE_CAMERA_FPS,
    LIVE_SHELF_ANALYSIS_FRAMES,
    POSE_LEFT_WRIST_INDEX,
    POSE_RIGHT_WRIST_INDEX,
    SHELF_CAMERA_WARMUP_FRAMES,
    SHELF_DISAPPEAR_THRESHOLD,
    SHELF_HAND_TOUCH_MARGIN,
    SHELF_HAND_TOUCH_TTL_FRAMES,
    SHELF_ITEM_CONFIDENCE_THRESHOLD,
    SHELF_LIVE_BUFFER_FLUSH_FRAMES,
    SHELF_POSE_CONFIDENCE_THRESHOLD,
    SHELF_REID_IOU_THRESHOLD,
    SHELF_REID_MAX_DIST,
    SHELF_REID_WINDOW_FRAMES,
    SHELF_SHOW_PREVIEW,
    SHELF_STABLE_FRAMES,
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
        self.tracker: ByteTrack | None = None

    def open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open shelf source: {self.source_label}")

        self.tracker = ByteTrack()

        for _ in range(SHELF_CAMERA_WARMUP_FRAMES):
            ok, _ = self.cap.read()
            if not ok:
                break

    def ensure_open(self) -> None:
        if self.cap is None or not self.cap.isOpened() or self.tracker is None:
            self.open()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.tracker = None


class ShelfItemMonitor:
    def __init__(
        self,
        model_path: str,
        pose_model_path: str,
        conf_threshold: float = SHELF_ITEM_CONFIDENCE_THRESHOLD,
        pose_conf_threshold: float = SHELF_POSE_CONFIDENCE_THRESHOLD,
        live_analysis_frames: int = LIVE_SHELF_ANALYSIS_FRAMES,
        show_preview: bool = SHELF_SHOW_PREVIEW,
        hand_touch_margin: int = SHELF_HAND_TOUCH_MARGIN,
    ):
        self.model_path = model_path
        self.pose_model_path = pose_model_path
        self.conf_threshold = conf_threshold
        self.pose_conf_threshold = pose_conf_threshold
        self.live_analysis_frames = live_analysis_frames
        self.show_preview = show_preview
        self.hand_touch_margin = hand_touch_margin
        self.hand_touch_ttl_frames = SHELF_HAND_TOUCH_TTL_FRAMES
        self._model = None
        self._pose_model = None
        self._class_names = {}
        self._sessions: dict[str, ShelfCameraSession] = {}

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

    def analyze_source(self, source: int | str | Path) -> ShelfInteractionResult:
        self._ensure_models()
        session = self._get_or_create_session(source)
        cap = session.cap
        tracker = session.tracker
        source_label = session.source_label

        frame_limit = None
        stable_frames = SHELF_STABLE_FRAMES
        disappear_threshold = SHELF_DISAPPEAR_THRESHOLD
        reid_window_frames = SHELF_REID_WINDOW_FRAMES
        reid_iou_threshold = SHELF_REID_IOU_THRESHOLD
        reid_max_dist = SHELF_REID_MAX_DIST

        if isinstance(source, int):
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = DEFAULT_LIVE_CAMERA_FPS
            frame_limit = max(1, int(self.live_analysis_frames))
            stable_frames = 8
            disappear_threshold = max(8, int(fps * 0.75))
            reid_window_frames = max(disappear_threshold * 2, int(fps * 2.0))

            for _ in range(SHELF_LIVE_BUFFER_FLUSH_FRAMES):
                ok, _ = cap.read()
                if not ok:
                    break

        shelf_items: dict[int, dict] = {}
        disappeared: defaultdict[int, int] = defaultdict(int)
        reid_candidates: dict[int, dict] = {}
        taken_events: list[dict] = []

        frame_idx = 0
        while cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame_limit is not None and frame_idx >= frame_limit:
                break

            people = self._detect_people_and_hands(frame)
            detections = self._detect_items(frame)
            tracks = tracker.update(detections, frame)
            active_ids = set()

            if self.show_preview:
                preview = frame.copy()
                self._draw_preview_overlay(
                    preview=preview,
                    source_label=source_label,
                    frame_idx=frame_idx,
                    frame_limit=frame_limit,
                )
                self._draw_people(preview, people)
                self._draw_tracks(preview, tracks, shelf_items, frame_idx)
                cv2.imshow(f"Shelf Live Preview - source {source_label}", preview)
                cv2.waitKey(1)

            for row in tracks if tracks is not None else []:
                x1, y1, x2, y2 = map(int, row[:4])
                track_id = int(row[4])
                conf = float(row[5]) if len(row) > 5 else 0.0
                cls = int(row[6]) if len(row) > 6 else 0
                class_name = self._class_names.get(cls, str(cls))
                box = (x1, y1, x2, y2)
                cx, cy = self._box_center(box)

                active_ids.add(track_id)

                if track_id not in shelf_items:
                    matched_id = self._find_reid_match(
                        class_name=class_name,
                        box=box,
                        frame_idx=frame_idx,
                        reid_candidates=reid_candidates,
                        reid_window_frames=reid_window_frames,
                        reid_max_dist=reid_max_dist,
                        reid_iou_threshold=reid_iou_threshold,
                    )
                    if matched_id is not None:
                        old_entry = shelf_items.pop(matched_id, None)
                        if old_entry:
                            shelf_items[track_id] = old_entry
                            shelf_items[track_id]["reappeared"] = True
                        else:
                            shelf_items[track_id] = {
                                "class_name": class_name,
                                "stable_count": stable_frames,
                                "status": "stable",
                                "last_touch": None,
                            }
                        reid_candidates.pop(matched_id, None)
                        disappeared[track_id] = 0
                    else:
                        shelf_items[track_id] = {
                            "class_name": class_name,
                            "stable_count": 1,
                            "status": "new",
                            "last_touch": None,
                        }
                else:
                    item = shelf_items[track_id]
                    item["stable_count"] += 1
                    if item["status"] == "new" and item["stable_count"] >= stable_frames:
                        item["status"] = "stable"

                shelf_items[track_id]["last_box"] = box
                shelf_items[track_id]["cx"] = cx
                shelf_items[track_id]["cy"] = cy
                shelf_items[track_id]["last_conf"] = conf
                disappeared[track_id] = 0

                best_touch = None
                for person in people:
                    for hand in person["hands"]:
                        distance = self._point_to_box_distance(hand["wrist"], box)
                        if distance > self.hand_touch_margin:
                            continue

                        candidate = {
                            "frame_idx": frame_idx,
                            "distance": distance,
                            "actor_side": person["actor_side"],
                            "person_label": person["person_label"],
                            "handedness": hand["handedness"],
                            "wrist": hand["wrist"],
                        }
                        if best_touch is None or candidate["distance"] < best_touch["distance"]:
                            best_touch = candidate

                if best_touch is not None:
                    shelf_items[track_id]["last_touch"] = best_touch

            for track_id, item in list(shelf_items.items()):
                if track_id in active_ids:
                    continue

                disappeared[track_id] += 1

                if item["status"] == "stable" and track_id not in reid_candidates:
                    reid_candidates[track_id] = {
                        "class_name": item["class_name"],
                        "last_box": item.get("last_box", (0, 0, 0, 0)),
                        "cx": item.get("cx", 0),
                        "cy": item.get("cy", 0),
                        "lost_frame": frame_idx,
                    }

                if item["status"] == "stable" and disappeared[track_id] >= disappear_threshold:
                    item["status"] = "taken"
                    reid_candidates.pop(track_id, None)
                    touch = item.get("last_touch")

                    taken_events.append(
                        {
                            "class_name": item["class_name"],
                            "track_id": track_id,
                            "frame_taken": frame_idx,
                            "actor_side": touch["actor_side"] if touch else "unknown-side",
                            "person_label": touch["person_label"] if touch else "unknown-person",
                            "handedness": touch["handedness"] if touch else "unknown-hand",
                        }
                    )
                    shelf_items.pop(track_id, None)

            frame_idx += 1

        if self.show_preview:
            cv2.destroyWindow(f"Shelf Live Preview - source {source_label}")

        counts = dict(Counter(event["class_name"] for event in taken_events))
        actor_counts = dict(Counter(event["actor_side"] for event in taken_events))
        return ShelfInteractionResult(
            source_label=source_label,
            taken_events=taken_events,
            counts=counts,
            actor_counts=actor_counts,
        )

    def _ensure_models(self):
        if self._model is None:
            model_path = Path(self.model_path).expanduser()
            if not model_path.exists():
                raise RuntimeError(f"Shelf-item model not found: {model_path}")
            self.model_path = str(model_path)
            self._model = YOLO(self.model_path)
            self._class_names = self._model.names
            print(f"[ShelfItemMonitor] loaded item model: {self.model_path}")

        if self._pose_model is None:
            pose_model_path = Path(self.pose_model_path).expanduser()
            if not pose_model_path.exists():
                raise RuntimeError(f"Shelf pose model not found: {pose_model_path}")
            self.pose_model_path = str(pose_model_path)
            self._pose_model = YOLO(self.pose_model_path)
            print(f"[ShelfItemMonitor] loaded pose model: {self.pose_model_path}")

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

    def _detect_people_and_hands(self, frame: np.ndarray) -> list[dict]:
        result = self._pose_model(frame, conf=self.pose_conf_threshold, verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else result.boxes.conf
        keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, "cpu") else result.keypoints.xy

        people = []
        for idx in range(len(boxes)):
            x1, y1, x2, y2 = boxes[idx].astype(int)
            person = {
                "person_box": (x1, y1, x2, y2),
                "person_center_x": (x1 + x2) / 2,
                "person_label": None,
                "actor_side": None,
                "hands": [],
                "conf": float(confs[idx]),
            }

            kp = keypoints[idx]
            left_wrist = tuple(map(int, kp[POSE_LEFT_WRIST_INDEX]))
            right_wrist = tuple(map(int, kp[POSE_RIGHT_WRIST_INDEX]))

            for hand_name, wrist in (("Left", left_wrist), ("Right", right_wrist)):
                wx, wy = wrist
                if wx <= 0 and wy <= 0:
                    continue
                person["hands"].append({"handedness": hand_name, "wrist": wrist})

            people.append(person)

        people.sort(key=lambda item: item["person_center_x"])
        for idx, person in enumerate(people):
            person["person_label"] = f"person_{idx}"
            if idx == 0:
                person["actor_side"] = "left-shopper"
            elif idx == len(people) - 1:
                person["actor_side"] = "right-shopper"
            else:
                person["actor_side"] = f"middle-shopper-{idx}"

        return people

    def _draw_tracks(self, frame: np.ndarray, tracks, shelf_items: dict, frame_idx: int) -> None:
        if tracks is None:
            return

        for row in tracks:
            x1, y1, x2, y2 = map(int, row[:4])
            track_id = int(row[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 80), 2)
            text = f"ID {track_id}"
            touch = shelf_items.get(track_id, {}).get("last_touch")
            if touch is not None and frame_idx - touch["frame_idx"] <= self.hand_touch_ttl_frames:
                text += f" {touch['person_label']} {touch['handedness']}"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    def _draw_people(self, frame: np.ndarray, people: list[dict]) -> None:
        for person in people:
            x1, y1, x2, y2 = person["person_box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(
                frame,
                f"{person['person_label']} {person['actor_side']}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                frame,
                f"{person['person_label']} {person['actor_side']}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 120),
                1,
            )
            for hand in person["hands"]:
                wx, wy = hand["wrist"]
                cv2.circle(frame, (wx, wy), 7, (40, 40, 255), -1)
                cv2.putText(
                    frame,
                    hand["handedness"],
                    (wx, max(20, wy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    def _draw_preview_overlay(
        self,
        preview: np.ndarray,
        source_label: str,
        frame_idx: int,
        frame_limit: int | None,
    ) -> None:
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 90), (20, 20, 20), -1)
        total_text = f"{frame_idx + 1}/{frame_limit}" if frame_limit is not None else f"{frame_idx + 1}"
        lines = [
            f"Shelf source: {source_label}",
            f"Analyzing live shelf interaction frame {total_text}",
            "Multi-person wrist attribution active",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 24
            cv2.putText(
                preview,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (235, 235, 235),
                2,
            )

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _point_to_box_distance(point, box):
        px, py = point
        x1, y1, x2, y2 = box
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        return float(np.hypot(dx, dy))

    def _find_reid_match(
        self,
        class_name: str,
        box: tuple,
        frame_idx: int,
        reid_candidates: dict,
        reid_window_frames: int,
        reid_max_dist: int,
        reid_iou_threshold: float,
    ) -> int | None:
        cx, cy = self._box_center(box)
        best_id, best_score = None, -1.0

        for tid, cand in reid_candidates.items():
            if cand["class_name"] != class_name:
                continue
            if frame_idx - cand["lost_frame"] > reid_window_frames:
                continue

            dist = np.hypot(cx - cand["cx"], cy - cand["cy"])
            iou = self._box_iou(box, cand["last_box"])
            if dist > reid_max_dist and iou < reid_iou_threshold:
                continue

            score = iou - (dist / (reid_max_dist * 4))
            if score > best_score:
                best_score = score
                best_id = tid

        return best_id

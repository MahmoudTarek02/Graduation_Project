from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from boxmot import ByteTrack
from ultralytics import YOLO


@dataclass
class ShelfInteractionResult:
    video_path: str
    taken_events: list[dict]
    counts: dict[str, int]


class ShelfVideoResolver:
    """
    Temporary interaction resolver.

    For now it prompts the operator to choose the shelf-camera video file that
    corresponds to a back-camera shelf event. Later this can be replaced with a
    direct shelf_id -> camera stream mapping.
    """

    def __init__(self, videos_dir: str | Path = "Videos", shelf_camera_map: dict | None = None):
        self.videos_dir = Path(videos_dir)
        self.shelf_camera_map = shelf_camera_map or {}

    def resolve(self, event: dict) -> str | None:
        mapped_source = self.shelf_camera_map.get(event["zone_id"])
        if mapped_source:
            return str(mapped_source)

        candidates = []
        if self.videos_dir.exists():
            for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                candidates.extend(sorted(self.videos_dir.glob(pattern)))

        print("\n" + "=" * 60)
        print(
            f"Shelf interaction detected: person {event['person_id']} -> shelf {event['zone_id']}"
        )
        print("Choose the shelf-camera video for this interaction.")
        if candidates:
            for idx, path in enumerate(candidates, start=1):
                print(f"  [{idx}] {path}")
        else:
            print("No videos found in the default videos directory.")
        print("Type a number, type a full path, or press Enter to skip.")
        print("=" * 60)

        while True:
            choice = input("Shelf video selection: ").strip()
            if not choice:
                return None

            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(candidates):
                    return str(candidates[index])
                print("Invalid number. Try again.")
                continue

            custom_path = Path(choice).expanduser()
            if custom_path.exists():
                return str(custom_path)

            print("File not found. Try again.")


class ShelfItemMonitor:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.35):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self._tracker = None
        self._class_names = {}

    def analyze_video(self, video_path: str | Path) -> ShelfInteractionResult:
        video_path = str(video_path)
        self._ensure_model()
        self._tracker = ByteTrack()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open shelf video: {video_path}")

        stable_frames = 15
        disappear_threshold = 90
        reid_window_frames = 120
        reid_iou_threshold = 0.25
        reid_max_dist = 120

        shelf_items: dict[int, dict] = {}
        disappeared: defaultdict[int, int] = defaultdict(int)
        reid_candidates: dict[int, dict] = {}
        taken_events: list[dict] = []

        def box_center(box):
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        def box_iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / union if union > 0 else 0.0

        def find_reid_match(class_name, box, frame_idx):
            cx, cy = box_center(box)
            best_id, best_score = None, -1.0

            for tid, cand in reid_candidates.items():
                if cand["class_name"] != class_name:
                    continue
                if frame_idx - cand["lost_frame"] > reid_window_frames:
                    continue

                dist = np.hypot(cx - cand["cx"], cy - cand["cy"])
                iou = box_iou(box, cand["last_box"])

                if dist > reid_max_dist and iou < reid_iou_threshold:
                    continue

                score = iou - (dist / (reid_max_dist * 4))
                if score > best_score:
                    best_score = score
                    best_id = tid

            return best_id

        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            detections = self._detect_items(frame)
            tracks = self._tracker.update(detections, frame)
            active_ids = set()

            for row in tracks if tracks is not None else []:
                x1, y1, x2, y2 = map(int, row[:4])
                track_id = int(row[4])
                conf = float(row[5]) if len(row) > 5 else 0.0
                cls = int(row[6]) if len(row) > 6 else 0
                class_name = self._class_names.get(cls, str(cls))
                box = (x1, y1, x2, y2)
                cx, cy = box_center(box)

                active_ids.add(track_id)

                if track_id not in shelf_items:
                    matched_id = find_reid_match(class_name, box, frame_idx)
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
                            }
                        reid_candidates.pop(matched_id, None)
                        disappeared[track_id] = 0
                    else:
                        shelf_items[track_id] = {
                            "class_name": class_name,
                            "stable_count": 1,
                            "status": "new",
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
                    taken_events.append(
                        {
                            "class_name": item["class_name"],
                            "track_id": track_id,
                            "frame_taken": frame_idx,
                        }
                    )
                    shelf_items.pop(track_id, None)

            frame_idx += 1

        cap.release()

        counts = dict(Counter(event["class_name"] for event in taken_events))
        return ShelfInteractionResult(
            video_path=video_path,
            taken_events=taken_events,
            counts=counts,
        )

    def _ensure_model(self):
        if self._model is not None:
            return

        resolved_path = self.model_path or input(
            "Enter shelf-item YOLO model path (.pt) for shelf-camera analysis: "
        ).strip()
        if not resolved_path:
            raise RuntimeError("Shelf-item model path is required for shelf-camera analysis.")

        model_path = Path(resolved_path).expanduser()
        if not model_path.exists():
            raise RuntimeError(f"Shelf-item model not found: {model_path}")

        self.model_path = str(model_path)
        self._model = YOLO(self.model_path)
        self._class_names = self._model.names
        print(f"[ShelfItemMonitor] loaded model: {self.model_path}")

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

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from boxmot import ByteTrack
from reid import REID
import operator
from datetime import datetime
import time
from collections import defaultdict

# ── New imports ────────────────────────────────────────────────────────────────
from hand_tracker import MediaPipeHandTracker   # swap this class to change model
from shelf_zone_selector import ShelfZoneSelector
from shelf_event_detector import ShelfEventDetector
from shelf_checkout import ShelfItemMonitor, ShelfVideoResolver


# ══════════════════════════════════════════════════════════════════════════════
# Existing classes (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", device="cuda"):
        print(f"Initializing YOLO detector with model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        print("YOLO detector loaded successfully")

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for b in results.boxes:
            if b.cls[0] != 0:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])
        return np.array(boxes)


class TrackerWrapper:
    def __init__(self, device="cuda"):
        print(f"Initializing ByteTrack tracker on device: {device}")
        self.tracker = ByteTrack(device=device, half=False)
        print("ByteTrack tracker initialized")

    def update(self, detections, frame):
        return self.tracker.update(detections, frame)

    def draw(self, frame):
        self.tracker.plot_results(frame, show_trajectories=True)


class ReIDManager:
    def __init__(self, threshold=600):
        print(f"Initializing ReID Manager with threshold: {threshold}")
        self.reid = REID()
        self.threshold = threshold
        self.images_by_id = {}
        self.feats = {}
        self.exist_ids = set()
        self.final_fuse_id = {}
        print("ReID Manager initialized")

    def process(self, res, frame):
        for row in res:
            tid = int(row[4])
            x1, y1, x2, y2 = map(int, row[:4])
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            if tid not in self.images_by_id:
                self.images_by_id[tid] = []
                self.feats[tid] = []
            self.images_by_id[tid].append(crop)
            self.feats[tid].append(self.reid._feature(crop))

        if len(res) == 0:
            return self.final_fuse_id

        current_ids = set(res[:, 4].astype(int))

        if len(self.exist_ids) == 0:
            for i in current_ids:
                self.final_fuse_id[i] = [i]
            self.exist_ids = current_ids
            print(f"Initial IDs detected: {self.final_fuse_id}")
            return self.final_fuse_id

        new_ids = current_ids - self.exist_ids

        for nid in new_ids:
            if nid not in self.images_by_id or len(self.images_by_id[nid]) < 10:
                continue
            self.exist_ids.add(nid)
            dis = []
            unpickable = []
            for i in current_ids:
                for key, group in self.final_fuse_id.items():
                    if i in group:
                        unpickable += group
            candidates = (self.exist_ids - set(unpickable)) & set(self.final_fuse_id.keys())
            print(f'exist_ids: {self.exist_ids}, unpickable: {unpickable}')
            for oid in candidates:
                d = np.median(
                    self.reid.compute_distance(
                        torch.cat(self.feats[nid], 0),
                        torch.cat(self.feats[oid], 0),
                    )
                )
                print(f'nid {nid}, oid {oid}, distance {d:.2f}')
                dis.append([oid, d])

            if not dis:
                self.final_fuse_id[nid] = [nid]
                print(f"New ID {nid} created (no candidates)")
                continue

            dis.sort(key=operator.itemgetter(1))
            if dis[0][1] < self.threshold:
                combined_id = dis[0][0]
                self.images_by_id[combined_id] += self.images_by_id[nid]
                self.final_fuse_id[combined_id].append(nid)
                print(f"ID {nid} merged with ID {combined_id} (distance: {dis[0][1]:.2f})")
            else:
                self.final_fuse_id[nid] = [nid]
                print(f"New ID {nid} created (distance too large: {dis[0][1]:.2f})")

        print(f"Current ID fusion: {self.final_fuse_id}")
        return self.final_fuse_id


# ══════════════════════════════════════════════════════════════════════════════
# Updated pipeline
# ══════════════════════════════════════════════════════════════════════════════

class PersonTrackingPipeline:
    def __init__(self, src=0, shelf_model_path: str | None = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
        print("=" * 60)

        self.cap = cv2.VideoCapture(
            src
        )
        self.device = device

        self.detector    = YOLODetector(device=self.device)
        self.tracker     = TrackerWrapper(device=self.device)
        self.reid_manager = ReIDManager()
        self.person_carts = defaultdict(lambda: defaultdict(int))
        self.shelf_video_resolver = ShelfVideoResolver(videos_dir="Videos")
        self.shelf_item_monitor = ShelfItemMonitor(model_path=shelf_model_path)
        self.processed_shelf_events = set()

        # ── Shelf zone setup ───────────────────────────────────────────────────
        # Read first frame for the zone-drawing GUI
        ok, first_frame = self.cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame from video source.")

        # Let user draw zones; returns {} if user skips / presses Escape
        zones = ShelfZoneSelector().select(first_frame)

        # ── Hand tracker (swap MediaPipeHandTracker → your own class here) ─────
        hand_tracker = MediaPipeHandTracker(
            max_hands=6,
            detection_confidence=0.5,
            tracking_confidence=0.5,
        )

        # ── Event detector ─────────────────────────────────────────────────────
        self.shelf_detector = ShelfEventDetector(
            zones=zones,
            hand_tracker=hand_tracker,
            cooldown_frames=60,           # ~2 s @ 30 fps between repeat events
            person_bbox_expand=0.5,
        )

        # Rewind so the first frame is processed again
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)

    def _handle_shelf_event(self, event: dict):
        person_id = event.get("person_id")
        if person_id is None:
            print(
                f"[PIPELINE] Skipping shelf {event['zone_id']} because the hand "
                "could not be matched to a tracked person."
            )
            return

        event_key = (
            person_id,
            event.get("zone_id"),
            event.get("handedness"),
            event.get("position"),
        )
        if event_key in self.processed_shelf_events:
            return

        self.processed_shelf_events.add(event_key)

        print(
            f"\n[PIPELINE] Pausing to inspect shelf {event['zone_id']} "
            f"for person {event['person_id']}."
        )
        shelf_video = self.shelf_video_resolver.resolve(event)
        if not shelf_video:
            print("[PIPELINE] Shelf-video selection skipped.")
            return

        try:
            result = self.shelf_item_monitor.analyze_video(shelf_video)
        except Exception as exc:
            self.processed_shelf_events.discard(event_key)
            print(f"[PIPELINE] Shelf analysis failed: {exc}")
            return

        if not result.counts:
            print(
                f"[PIPELINE] No taken items detected in shelf video: {result.video_path}"
            )
            return

        for class_name, qty in result.counts.items():
            self.person_carts[person_id][class_name] += qty

        print(f"[PIPELINE] Cart updated for person {person_id}:")
        for class_name, qty in sorted(self.person_carts[person_id].items()):
            print(f"  {qty}x {class_name}")

    # ── Frame processing ───────────────────────────────────────────────────────
    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        tracks     = self.tracker.update(detections, frame)
        fuse       = self.reid_manager.process(tracks, frame)

        # Draw ByteTrack trajectories first, then shelf overlay on top
        self.tracker.draw(frame)

        # Detect hand–zone interactions and draw overlay
        events = self.shelf_detector.process(frame, tracks, fuse)

        return frame, fuse, events

    # ── Main loop ──────────────────────────────────────────────────────────────
    def start(self):
        print("\nStarting video processing...")
        print("Press 'q' to quit\n")

        all_events = []

        with torch.inference_mode():
            frame_count = 0
            while True:
                t1 = time.time()
                ok, frame = self.cap.read()
                if not ok:
                    print("\nEnd of video stream")
                    break

                disp, fuse, events = self.process_frame(frame)
                all_events.extend(events)
                for event in events:
                    self._handle_shelf_event(event)

                frame_count += 1
                fps = 1 / (time.time() - t1)
                print(f"Frame: {frame_count}, FPS: {fps:.2f}", end="\r")

                cv2.imshow("YOLO + ByteTrack + ReID + Shelf Detection", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"\nUser quit after {frame_count} frames")
                    break

        self.cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("Final Fused IDs:")
        print(self.reid_manager.final_fuse_id)
        print(f"Total unique persons detected: {len(self.reid_manager.final_fuse_id)}")
        print(f"\nTotal shelf-interaction events: {len(all_events)}")
        for ev in all_events:
            print(
                f"  Person {ev['person_id']} ({ev['handedness']}) → "
                f"Shelf {ev['zone_id']} at {ev['position']}"
            )
        if self.person_carts:
            print("\nDetected carts:")
            for person_id in sorted(self.person_carts, key=str):
                cart = self.person_carts[person_id]
                items = ", ".join(
                    f"{qty}x {name}" for name, qty in sorted(cart.items())
                ) or "empty"
                print(f"  Person {person_id}: {items}")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = PersonTrackingPipeline(
        src=r"C:\Users\omara\OneDrive - Alexandria University\Desktop\College\grad_project\Final\Videos\video_back.mp4"
    )
    pipeline.start()

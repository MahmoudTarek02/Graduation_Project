import cv2
import torch
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
from boxmot import ByteTrack
from reid import REID
import operator
from datetime import datetime
import time
from collections import defaultdict
from queue import Empty

from config import (
    BACK_DETECTOR_MODEL_PATH,
    BACK_HAND_DETECTION_CONFIDENCE,
    BACK_HAND_FINGERTIP,
    BACK_HAND_MAX_HANDS,
    BACK_HAND_TRACKING_CONFIDENCE,
    LIVE_SHELF_ANALYSIS_FRAMES,
    MAX_CAMERA_INDEX_TO_PROBE,
    REID_MATCH_THRESHOLD,
    SHELF_EVENT_COOLDOWN_FRAMES,
    SHELF_EVENT_PERSON_BBOX_EXPAND,
    SHELF_ITEM_MODEL_PATH,
    SHELF_POSE_MODEL_PATH,
)

# ── New imports ────────────────────────────────────────────────────────────────
from hand_tracker import MediaPipeHandTracker   # swap this class to change model
from shelf_zone_selector import ShelfZoneSelector
from shelf_event_detector import ShelfEventDetector
from shelf_checkout import ShelfItemMonitor, ShelfCameraResolver


def _serialize_shelf_result(result):
    return {
        "source_label": result.source_label,
        "taken_events": result.taken_events,
        "counts": result.counts,
        "actor_counts": result.actor_counts,
    }


def _shelf_analysis_worker_main(
    source,
    model_path: str,
    pose_model_path: str,
    live_analysis_frames: int,
    task_queue,
    result_queue,
):
    monitor = None
    try:
        monitor = ShelfItemMonitor(
            model_path=model_path,
            pose_model_path=pose_model_path,
            live_analysis_frames=live_analysis_frames,
        )
        monitor.prepare_sources([source])
        result_queue.put(
            {
                "kind": "worker_ready",
                "source": source,
            }
        )

        while True:
            task = task_queue.get()
            if task is None or task.get("kind") == "shutdown":
                break

            if task.get("kind") != "analyze":
                continue

            request_id = task["request_id"]
            event = task["event"]
            try:
                result = monitor.analyze_source(source)
                result_queue.put(
                    {
                        "kind": "analysis_result",
                        "source": source,
                        "request_id": request_id,
                        "event": event,
                        "result": _serialize_shelf_result(result),
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "kind": "analysis_error",
                        "source": source,
                        "request_id": request_id,
                        "event": event,
                        "error": str(exc),
                    }
                )
    except Exception as exc:
        result_queue.put(
            {
                "kind": "worker_error",
                "source": source,
                "error": str(exc),
            }
        )
    finally:
        if monitor is not None:
            monitor.close()


class ShelfAnalysisWorker:
    def __init__(
        self,
        mp_ctx,
        source,
        model_path: str,
        pose_model_path: str,
        live_analysis_frames: int,
    ):
        self.source = source
        self.task_queue = mp_ctx.Queue()
        self.result_queue = mp_ctx.Queue()
        self.process = mp_ctx.Process(
            target=_shelf_analysis_worker_main,
            args=(
                source,
                model_path,
                pose_model_path,
                live_analysis_frames,
                self.task_queue,
                self.result_queue,
            ),
            daemon=True,
        )

    def start(self):
        self.process.start()

    def submit(self, payload: dict):
        self.task_queue.put(payload)

    def try_get(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        if self.process.is_alive():
            self.task_queue.put({"kind": "shutdown"})
            self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# Existing classes (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class YOLODetector:
    def __init__(self, model_path=BACK_DETECTOR_MODEL_PATH, device="cuda"):
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
    def __init__(self, threshold=REID_MATCH_THRESHOLD):
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

        # print(f"Current ID fusion: {self.final_fuse_id}")
        return self.final_fuse_id


class CameraSelector:
    WINDOW_NAME = "Camera Preview - Y: select, N: next, Q: quit"

    def __init__(self, max_index: int = MAX_CAMERA_INDEX_TO_PROBE):
        self.max_index = max_index

    def choose_camera(self, label: str, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        print(f"\nSearching camera indices for {label}...")
        for index in range(self.max_index + 1):
            if index in excluded:
                continue
            print(f"[CameraSelector] Checking camera {index} for {label}...")
            result = self._preview_and_confirm(index, label)
            if result is True:
                print(f"[CameraSelector] {label} -> camera {index}")
                return index
            if result is None:
                raise RuntimeError("Camera selection aborted by user.")

        raise RuntimeError(f"No available cameras found for {label}.")

    def _preview_and_confirm(self, index: int, label: str) -> bool | None:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return False

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        result = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            preview = frame.copy()
            lines = [
                f"{label}: camera {index}",
                "Press Y to select this camera",
                "Press N to try the next camera",
                "Press Q to abort",
            ]
            self._draw_hud(preview, lines)
            cv2.imshow(self.WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("y"), ord("Y")):
                result = True
                break
            if key in (ord("n"), ord("N")):
                break
            if key in (ord("q"), ord("Q"), 27):
                result = None
                break

        cap.release()
        cv2.destroyWindow(self.WINDOW_NAME)
        return result

    @staticmethod
    def _draw_hud(frame, lines):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (20, 20, 20), -1)
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


# ══════════════════════════════════════════════════════════════════════════════
# Updated pipeline
# ══════════════════════════════════════════════════════════════════════════════

class PersonTrackingPipeline:
    def __init__(
        self,
        src=0,
        shelf_model_path: str = SHELF_ITEM_MODEL_PATH,
        source_index: int | None = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
        print("=" * 60)

        self.cap = cv2.VideoCapture(src)
        self.src = src
        self.device = device
        self.source_index = source_index if source_index is not None else src if isinstance(src, int) else None
        self.mp_ctx = mp.get_context("spawn")

        self.detector    = YOLODetector(device=self.device)
        self.tracker     = TrackerWrapper(device=self.device)
        self.reid_manager = ReIDManager()
        self.person_carts = defaultdict(lambda: defaultdict(int))
        self.shelf_camera_resolver = ShelfCameraResolver()
        self.shelf_model_path = shelf_model_path
        self.shelf_workers = {}
        self.pending_shelf_requests = set()
        self.pending_request_meta = {}
        self.active_shelf_batches = {}
        self.next_shelf_request_id = 1
        self.back_frame_width = None

        # ── Shelf zone setup ───────────────────────────────────────────────────
        zone_selector = ShelfZoneSelector()
        if isinstance(self.source_index, int):
            zones = zone_selector.select_live(self.cap)
        else:
            ok, first_frame = self.cap.read()
            if not ok:
                raise RuntimeError("Cannot read first frame from video source.")
            zones = zone_selector.select(first_frame)
        self._assign_shelf_cameras(zones)

        # ── Hand tracker (swap MediaPipeHandTracker → your own class here) ─────
        hand_tracker = MediaPipeHandTracker(
            max_hands=BACK_HAND_MAX_HANDS,
            detection_confidence=BACK_HAND_DETECTION_CONFIDENCE,
            tracking_confidence=BACK_HAND_TRACKING_CONFIDENCE,
            use_fingertip=BACK_HAND_FINGERTIP,
        )

        # ── Event detector ─────────────────────────────────────────────────────
        self.shelf_detector = ShelfEventDetector(
            zones=zones,
            hand_tracker=hand_tracker,
            cooldown_frames=SHELF_EVENT_COOLDOWN_FRAMES,
            person_bbox_expand=SHELF_EVENT_PERSON_BBOX_EXPAND,
        )

        # Rewind only for file-based sources so the first frame is processed again.
        # For live cameras, reopen the capture to start tracking from a fresh stream.
        if self.source_index is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot reopen live camera source: {self.src}")

        print("=" * 60)
        print("Pipeline initialization complete!")
        print("=" * 60)

    def _assign_shelf_cameras(self, zones: dict):
        if not zones:
            print("[Pipeline] No shelf zones were defined, so no shelf cameras were assigned.")
            return

        selector = CameraSelector()
        used_indices = {self.source_index} if isinstance(self.source_index, int) else set()

        shelf_camera_map = {}
        print("\nAssign a shelf camera to each shelf zone.")
        for zone_id in sorted(zones):
            camera_index = selector.choose_camera(
                label=f"Shelf {zone_id} camera",
                excluded=used_indices,
            )
            shelf_camera_map[zone_id] = camera_index
            used_indices.add(camera_index)

        self.shelf_camera_resolver = ShelfCameraResolver(shelf_camera_map=shelf_camera_map)
        for source in shelf_camera_map.values():
            worker = ShelfAnalysisWorker(
                mp_ctx=self.mp_ctx,
                source=source,
                model_path=self.shelf_model_path,
                pose_model_path=SHELF_POSE_MODEL_PATH,
                live_analysis_frames=LIVE_SHELF_ANALYSIS_FRAMES,
            )
            worker.start()
            self.shelf_workers[str(source)] = worker

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
        )
        if event_key in self.pending_shelf_requests:
            return

        shelf_source = self.shelf_camera_resolver.resolve(event)
        if shelf_source is None:
            print("[PIPELINE] No shelf camera mapped to this zone.")
            return

        worker = self.shelf_workers.get(str(shelf_source))
        if worker is None:
            print(f"[PIPELINE] No analysis worker running for shelf source {shelf_source}.")
            return

        request_id = self.next_shelf_request_id
        self.next_shelf_request_id += 1
        actor_side_hint = self._infer_actor_side_hint(event)
        source_key = str(shelf_source)

        self.pending_shelf_requests.add(event_key)
        self.pending_request_meta[request_id] = {
            "event_key": event_key,
            "person_id": person_id,
            "zone_id": event.get("zone_id"),
            "actor_side_hint": actor_side_hint,
            "shelf_source": shelf_source,
        }
        active_batch = self.active_shelf_batches.get(source_key)
        if active_batch is not None:
            active_batch["request_ids"].append(request_id)
            print(
                f"\n[PIPELINE] Attached person {person_id} on shelf {event['zone_id']} "
                f"to active worker batch on source {shelf_source}."
            )
            return

        self.active_shelf_batches[source_key] = {"request_ids": [request_id]}
        worker.submit(
            {
                "kind": "analyze",
                "request_id": request_id,
                "event": event,
            }
        )
        print(
            f"\n[PIPELINE] Submitted shelf {event['zone_id']} analysis for "
            f"person {person_id} to worker source {shelf_source}."
        )

    def _drain_shelf_results(self):
        for worker in self.shelf_workers.values():
            while True:
                message = worker.try_get()
                if message is None:
                    break
                self._handle_shelf_worker_message(message)

    def _handle_shelf_worker_message(self, message: dict):
        kind = message.get("kind")
        if kind == "worker_ready":
            print(f"[PIPELINE] Shelf worker ready for source {message['source']}.")
            return

        if kind == "worker_error":
            print(
                f"[PIPELINE] Shelf worker for source {message['source']} failed: "
                f"{message['error']}"
            )
            return

        source_key = str(message.get("source"))
        request_id = message.get("request_id")
        batch = self.active_shelf_batches.pop(source_key, None)
        request_ids = batch["request_ids"] if batch is not None else [request_id]

        if kind == "analysis_error":
            for batch_request_id in request_ids:
                meta = self.pending_request_meta.pop(batch_request_id, None)
                if meta is not None:
                    self.pending_shelf_requests.discard(meta["event_key"])
            print(
                f"[PIPELINE] Shelf analysis failed for source {message.get('source')}: "
                f"{message.get('error')}"
            )
            return

        result = message.get("result", {})
        for batch_request_id in request_ids:
            meta = self.pending_request_meta.pop(batch_request_id, None)
            if meta is None:
                continue

            self.pending_shelf_requests.discard(meta["event_key"])

            taken_events = result.get("taken_events", [])
            if not taken_events:
                print(
                    f"[PIPELINE] No taken items detected from shelf source: "
                    f"{result.get('source_label', meta['shelf_source'])}"
                )
                continue

            actor_side_hint = meta["actor_side_hint"]
            matched_events = [
                item_event
                for item_event in taken_events
                if item_event.get("actor_side") == actor_side_hint
            ]
            actor_counts = result.get("actor_counts", {})
            if not matched_events and len(actor_counts) == 1:
                matched_events = taken_events
            if not matched_events:
                print(
                    f"[PIPELINE] Shelf source {result.get('source_label', meta['shelf_source'])} "
                    f"saw items, but none matched the back-camera side hint "
                    f"({actor_side_hint})."
                )
                print(f"[PIPELINE] Shelf-side counts: {actor_counts}")
                continue

            person_id = meta["person_id"]
            for item_event in matched_events:
                class_name = item_event["class_name"]
                self.person_carts[person_id][class_name] += 1

            print(f"[PIPELINE] Cart updated for person {person_id}:")
            for class_name, qty in sorted(self.person_carts[person_id].items()):
                print(f"  {qty}x {class_name}")
            print(f"[PIPELINE] Matched shelf actor side: {actor_side_hint}")
            for item_event in matched_events:
                print(
                    f"    {item_event['class_name']} from "
                    f"{item_event['person_label']} {item_event['handedness']}"
                )

    def _infer_actor_side_hint(self, event: dict) -> str:
        if self.back_frame_width is None:
            return "unknown-side"
        x = event.get("position", (self.back_frame_width // 2, 0))[0]
        return "left-shopper" if x < self.back_frame_width / 2 else "right-shopper"

    # ── Frame processing ───────────────────────────────────────────────────────
    def process_frame(self, frame):
        self.back_frame_width = frame.shape[1]
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
                self._drain_shelf_results()

                frame_count += 1
                fps = 1 / (time.time() - t1)
                print(f"Frame: {frame_count}, FPS: {fps:.2f}", end="\r")

                cv2.imshow("YOLO + ByteTrack + ReID + Shelf Detection", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"\nUser quit after {frame_count} frames")
                    break

        self.cap.release()
        self._drain_shelf_results()
        for worker in self.shelf_workers.values():
            worker.stop()
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
    mp.freeze_support()
    camera_selector = CameraSelector()
    back_camera_index = camera_selector.choose_camera(label="Back camera")

    pipeline = PersonTrackingPipeline(src=back_camera_index, source_index=back_camera_index)
    pipeline.start()

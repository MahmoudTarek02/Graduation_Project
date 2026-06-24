from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from config import (
    DEFAULT_LIVE_CAMERA_FPS,
    ITEM_DETECTOR_BOX_THICKNESS,
    ITEM_DETECTOR_LABEL_Y_OFFSET,
    ITEM_DETECTOR_MIN_LABEL_Y,
    ITEM_DETECTOR_OUTPUT_VIDEO_CODEC,
    ITEM_DETECTOR_OUTPUT_VIDEO_PATH,
    ITEM_DETECTOR_TEXT_COLOR,
    ITEM_DETECTOR_TEXT_SCALE,
    ITEM_DETECTOR_TEXT_SHADOW_COLOR,
    ITEM_DETECTOR_TEXT_THICKNESS,
    ITEM_DETECTOR_TRACK_BOX_COLOR,
)
from item_detector import ItemDetector


def draw_tracked_items(frame, tracks: list[dict]):
    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            ITEM_DETECTOR_TRACK_BOX_COLOR,
            ITEM_DETECTOR_BOX_THICKNESS,
        )
        label = f"{track['class_name']} ID {track['track_id']}"
        label_origin = (x1, max(ITEM_DETECTOR_MIN_LABEL_Y, y1 - ITEM_DETECTOR_LABEL_Y_OFFSET))
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            ITEM_DETECTOR_TEXT_SCALE,
            ITEM_DETECTOR_TEXT_SHADOW_COLOR,
            ITEM_DETECTOR_BOX_THICKNESS,
        )
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            ITEM_DETECTOR_TEXT_SCALE,
            ITEM_DETECTOR_TEXT_COLOR,
            ITEM_DETECTOR_TEXT_THICKNESS,
        )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 item detection on a video file")
    parser.add_argument("video", help="Path to the shelf-camera video file")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_LIVE_CAMERA_FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(ITEM_DETECTOR_OUTPUT_VIDEO_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*ITEM_DETECTOR_OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    detector = ItemDetector()
    take_events: list[dict] = []

    def print_take_event(event: dict) -> None:
        print(
            f"[TAKE] frame={event['frame_taken']} "
            f"class={event['class_name']} "
            f"track_id={event['track_id']}",
            flush=True,
        )

    detector.register_callback(print_take_event)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            events = detector.process_frame(frame)
            take_events.extend(events)
            annotated_frame = draw_tracked_items(frame.copy(), detector.last_tracks)
            writer.write(annotated_frame)
    finally:
        cap.release()
        writer.release()

    print(f"Output video saved to {output_path}")
    print(f"Total take events: {len(take_events)}")
    for class_name, count in Counter(event["class_name"] for event in take_events).items():
        print(f"{class_name} x{count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

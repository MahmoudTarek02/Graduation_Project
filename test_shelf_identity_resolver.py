from __future__ import annotations

import argparse
from pathlib import Path

from config import IDENTITY_GALLERY_PATH
from shelf_identity_resolver import ShelfIdentityResolver


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 forward-camera identity resolution on a video file")
    parser.add_argument("video", help="Path to the forward-camera video file")
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Positive for taken events, negative for returned events",
    )
    parser.add_argument(
        "--class-name",
        default="shelf_item",
        help="Shelf item class name to attach to the trigger event",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    resolver = ShelfIdentityResolver()
    trigger_event = {
        "class_name": args.class_name,
        "quantity": args.quantity,
        "frame_taken": 0,
    }
    result = resolver.resolve_event(trigger_event=trigger_event, source=video_path)

    print(f"Gallery: {IDENTITY_GALLERY_PATH}")
    print(f"Source: {video_path}")
    print(f"Event type: {result.event_type}")
    print(f"Frame index: {result.frame_index}")
    print(f"Person count: {result.person_count}")
    print(f"Selection method: {result.selection_method}")
    print(f"Arm side: {result.arm_side}")
    print(f"Selected box: {result.selected_box}")
    print(f"Matched person_id: {result.person_id}")
    print(f"Score: {result.score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from config import SHELF_ITEM_CONFIDENCE_THRESHOLD, SHELF_ITEM_MODEL_PATH
from pipeline_orchestrator import RetailPipelineOrchestrator
from shelf_checkout import ShelfItemMonitor


def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return str(Path(value).expanduser())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end shelf counting plus forward identity pipeline")
    parser.add_argument("shelf_source", help="Shelf-counting camera index or video path")
    parser.add_argument("forward_source", help="Forward-facing camera index or video path")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the shelf monitor OpenCV preview window",
    )
    parser.add_argument(
        "--actor-side",
        default="unknown-side",
        help="Optional shelf-side hint attached to generated shelf events",
    )
    parser.add_argument(
        "--synthetic-event",
        action="store_true",
        help="Skip shelf counting and send one synthetic shelf event to the forward resolver",
    )
    parser.add_argument(
        "--class-name",
        default="shelf_item",
        help="Class name used with --synthetic-event",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Positive means taken, negative means returned when using --synthetic-event",
    )
    args = parser.parse_args()

    shelf_source = _parse_source(args.shelf_source)
    forward_source = _parse_source(args.forward_source)

    shelf_monitor = ShelfItemMonitor(
        model_path=SHELF_ITEM_MODEL_PATH,
        conf_threshold=SHELF_ITEM_CONFIDENCE_THRESHOLD,
        show_preview=args.preview,
    )
    orchestrator = RetailPipelineOrchestrator(shelf_monitor=shelf_monitor)

    try:
        if args.synthetic_event:
            shelf_result = None
            pipeline_results = [
                orchestrator.resolve_shelf_event(
                    {
                        "class_name": args.class_name,
                        "quantity": args.quantity,
                        "frame_taken": 0,
                        "actor_side": args.actor_side,
                    },
                    forward_source=forward_source,
                )
            ]
        else:
            shelf_result, pipeline_results = orchestrator.run_once(
                shelf_source=shelf_source,
                forward_source=forward_source,
                actor_side_hint=args.actor_side,
            )
    finally:
        orchestrator.close()

    if shelf_result is None:
        print("Shelf source: synthetic event")
        print("Shelf counts delta: not run")
        print("Shelf events: 1")
    else:
        print(f"Shelf source: {shelf_result.source_label}")
        print(f"Shelf counts delta: {shelf_result.counts}")
        print(f"Shelf events: {len(shelf_result.taken_events)}")
    print(f"Resolved events: {len(pipeline_results)}")

    for idx, result in enumerate(pipeline_results, start=1):
        row = result.to_dict()
        print(
            f"[{idx}] event={row['event_type']} "
            f"class={row['class_name']} "
            f"quantity={row['quantity']} "
            f"person_id={row['person_id']} "
            f"score={row['score']} "
            f"method={row['selection_method']} "
            f"person_count={row['person_count']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

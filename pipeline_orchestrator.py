from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from config import SHELF_ITEM_CONFIDENCE_THRESHOLD, SHELF_ITEM_MODEL_PATH, SHELF_SHOW_PREVIEW
from shelf_checkout import ShelfInteractionResult, ShelfItemMonitor
from shelf_identity_resolver import ShelfIdentityResolutionResult, ShelfIdentityResolver


@dataclass
class PipelineResult:
    shelf_event: dict
    identity_result: ShelfIdentityResolutionResult

    @property
    def event_type(self) -> str:
        return self.identity_result.event_type

    @property
    def person_id(self) -> str | None:
        return self.identity_result.person_id

    @property
    def score(self) -> float | None:
        return self.identity_result.score

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "class_name": self.shelf_event.get("class_name"),
            "quantity": self.shelf_event.get("quantity"),
            "person_id": self.person_id,
            "score": self.score,
            "selection_method": self.identity_result.selection_method,
            "person_count": self.identity_result.person_count,
            "arm_side": self.identity_result.arm_side,
            "shelf_event": dict(self.shelf_event),
            "identity_result": asdict(self.identity_result),
        }


class RetailPipelineOrchestrator:
    def __init__(
        self,
        shelf_monitor: ShelfItemMonitor | None = None,
        identity_resolver: ShelfIdentityResolver | None = None,
    ):
        self.shelf_monitor = shelf_monitor or ShelfItemMonitor(
            model_path=SHELF_ITEM_MODEL_PATH,
            conf_threshold=SHELF_ITEM_CONFIDENCE_THRESHOLD,
            show_preview=SHELF_SHOW_PREVIEW,
        )
        self.identity_resolver = identity_resolver or ShelfIdentityResolver()

    def run_once(
        self,
        shelf_source: int | str | Path,
        forward_source: int | str | Path,
        trigger_event: dict | None = None,
        actor_side_hint: str = "unknown-side",
        zone_presence: dict | None = None,
        on_hand_detected: Callable[[dict], None] | None = None,
    ) -> tuple[ShelfInteractionResult, list[PipelineResult]]:
        shelf_result = self.shelf_monitor.analyze_source(
            source=shelf_source,
            trigger_event=trigger_event,
            actor_side_hint=actor_side_hint,
            zone_presence=zone_presence,
            on_hand_detected=on_hand_detected,
        )

        pipeline_results = [
            self.resolve_shelf_event(event, forward_source)
            for event in shelf_result.taken_events
            if self._is_inventory_change_event(event)
        ]
        return shelf_result, pipeline_results

    def resolve_shelf_event(
        self,
        shelf_event: dict,
        forward_source: int | str | Path,
    ) -> PipelineResult:
        identity_result = self.identity_resolver.resolve_event(
            trigger_event=shelf_event,
            source=forward_source,
        )
        return PipelineResult(
            shelf_event=dict(shelf_event),
            identity_result=identity_result,
        )

    def close(self) -> None:
        self.shelf_monitor.close()
        close_method = getattr(self.identity_resolver, "close", None)
        if callable(close_method):
            close_method()

    def _is_inventory_change_event(self, event: dict) -> bool:
        return int(event.get("quantity", 0) or 0) != 0

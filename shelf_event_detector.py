"""
shelf_event_detector.py
-----------------------
Detects when a tracked person's bounding box overlaps with a shelf zone.

Design
------
ShelfEventDetector.process(frame, tracks, fuse_id)
    - Checks if any tracked person's center or bottom-center overlaps with any zone
    - Monitors a dwell-time counter to ensure the person stands there for >= 15 frames
    - Fires event callbacks on zone entry (with per-person-per-zone cooldown)
    - Draws overlay (zones, active indicators, event banners) on frame in-place

Callbacks
---------
Register any callable with register_callback(fn).
It receives an event dict:
    {
        'person_id' : int,           # fused/canonical person ID
        'zone_id'   : int,           # shelf number
        'handedness': 'unknown-hand',
        'position'  : (x, y),        # shopper center position
        'type'      : 'enter'
    }
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

import cv2
import numpy as np

from config import (
    SHELF_EVENT_BANNER_DURATION,
    SHELF_EVENT_COOLDOWN_FRAMES,
    SHELF_EVENT_PERSON_BBOX_EXPAND,
)

# ── colours (BGR) ──────────────────────────────────────────────────────────────
_ZONE_COLORS = [
    (50, 220, 50),
    (50, 100, 255),
    (255, 50, 50),
    (50, 220, 220),
    (220, 50, 220),
    (0, 180, 255),
]

_BANNER_DURATION = SHELF_EVENT_BANNER_DURATION


def _zone_color(zid: int) -> tuple:
    return _ZONE_COLORS[(zid - 1) % len(_ZONE_COLORS)]


# ── main class ─────────────────────────────────────────────────────────────────

class ShelfEventDetector:
    """
    Parameters
    ----------
    zones       : dict returned by ShelfZoneSelector.select()
                  {zone_id: [(x, y), ...]}
    cooldown_frames: frames to wait before re-firing the same
                     (person_id, zone_id) enter event
    person_bbox_expand: fractional expansion of person bounding box
    """

    def __init__(
        self,
        zones: dict,
        cooldown_frames: int = SHELF_EVENT_COOLDOWN_FRAMES,
        person_bbox_expand: float = SHELF_EVENT_PERSON_BBOX_EXPAND,
    ):
        self._zones = {
            zid: np.array(pts, np.int32) for zid, pts in zones.items()
        }
        self._cooldown_max = cooldown_frames
        self._expand = person_bbox_expand

        # State
        self._cooldown: dict[tuple, int] = {}          # (pid, zid) → frames left
        self._in_zone: dict[tuple, bool] = defaultdict(bool)  # (pid, zid) → bool
        self._dwell_time: dict[tuple, int] = defaultdict(int) # (pid, zid) → frames present

        # Callbacks
        self._callbacks: list[Callable] = []

        # Visual banners  {msg: frames_remaining}
        self._banners: dict[str, int] = {}

        print(
            f"[ShelfEventDetector] init — zones={list(zones.keys())}, "
            f"cooldown={cooldown_frames} frames"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_callback(self, fn: Callable):
        """Register a function to call on every zone-entry event."""
        self._callbacks.append(fn)

    def process(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        fuse_id: dict,
    ) -> list[dict]:
        """
        Run shopper zone checking and drawing.

        Parameters
        ----------
        frame   : current BGR frame (modified in-place)
        tracks  : ByteTrack output  M×(x1,y1,x2,y2,tid,conf,cls,idx)
        fuse_id : ReID fusion map   {canonical_id: [tracker_ids, ...]}

        Returns
        -------
        List of event dicts fired this frame.
        """
        if not self._zones:
            return []

        events = []

        # Tick cooldowns
        for key in list(self._cooldown):
            self._cooldown[key] -= 1
            if self._cooldown[key] <= 0:
                del self._cooldown[key]

        # Tick banners
        for msg in list(self._banners):
            self._banners[msg] -= 1
            if self._banners[msg] <= 0:
                del self._banners[msg]

        active_keys = set()

        for row in tracks if tracks is not None else []:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tid = int(row[4])
            person_id = _resolve_fused_id(tid, fuse_id)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            bottom_center = ((x1 + x2) // 2, y2)
            points_to_check = [center, bottom_center]

            for zid, poly in self._zones.items():
                inside = any(_point_in_poly(pt, poly) for pt in points_to_check)
                key = (person_id, zid)
                active_keys.add(key)
                was_inside = self._in_zone[key]

                if inside:
                    self._dwell_time[key] += 1
                    if self._dwell_time[key] >= 15 and not was_inside and key not in self._cooldown:
                        # ── ENTRY EVENT ───────────────────────────────────────
                        self._in_zone[key] = True
                        self._cooldown[key] = self._cooldown_max

                        ev = {
                            "person_id": person_id,
                            "zone_id": zid,
                            "handedness": "unknown-hand",
                            "position": center,
                            "type": "enter",
                        }
                        events.append(ev)

                        # Console
                        msg = f"[SHELF EVENT] Person {person_id} entered Shelf {zid} zone"
                        print(f"\n{'─'*55}")
                        print(msg)
                        print(f"{'─'*55}")

                        # Banner
                        self._banners[msg] = _BANNER_DURATION

                        # User callbacks
                        for cb in self._callbacks:
                            try:
                                cb(ev)
                            except Exception as exc:
                                print(f"[ShelfEventDetector] callback error: {exc}")
                else:
                    self._dwell_time[key] = 0
                    self._in_zone[key] = False

        # Reset dwell times and in-zone states for inactive track IDs
        for key in list(self._dwell_time):
            if key not in active_keys:
                self._dwell_time[key] = 0
                self._in_zone[key] = False

        # ── Draw everything ───────────────────────────────────────────────────
        self._draw(frame, tracks, fuse_id)

        return events

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self, frame, tracks, fuse_id):
        # Zone fills (semi-transparent)
        overlay = frame.copy()
        for zid, poly in self._zones.items():
            cv2.fillPoly(overlay, [poly], _zone_color(zid))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Zone borders & labels
        for zid, poly in self._zones.items():
            color = _zone_color(zid)
            # Highlight if any person is inside right now
            active = any(
                self._in_zone[(pid, zid)]
                for pid in _all_person_ids(fuse_id)
            )
            thickness = 3 if active else 2
            cv2.polylines(frame, [poly], True, color, thickness)
            cx, cy = np.mean(poly, axis=0).astype(int)
            _label(frame, f"Shelf {zid}", (cx, cy), color)

        # Event banners (stacked at bottom)
        if self._banners:
            bh = 36
            base_y = frame.shape[0] - bh * len(self._banners)
            for i, (msg, frames_left) in enumerate(self._banners.items()):
                alpha = min(1.0, frames_left / 15)   # fade-out in last 15 frames
                _banner(frame, msg, base_y + i * bh, alpha)


# ── Utility functions ──────────────────────────────────────────────────────────

def _point_in_poly(point: tuple, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def _resolve_fused_id(tid: int, fuse_id: dict) -> int:
    """Map a tracker ID to its canonical (master) fused person ID."""
    for master, group in fuse_id.items():
        if tid in group:
            return master
    return tid   # not yet in fuse_id → return raw tid


def _all_person_ids(fuse_id: dict) -> list:
    return list(fuse_id.keys())


def _label(img, text, center, color, scale=0.75):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x, y = center[0] - tw // 2, center[1] + th // 2
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y), font, scale, color, 2)


def _banner(img, text, y: int, alpha: float = 1.0):
    """Draw a semi-transparent notification banner at y."""
    h, w = img.shape[:2]
    bar_h = 34
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y), (w, y + bar_h), (20, 20, 160), -1)
    cv2.addWeighted(overlay, alpha * 0.80, img, 1 - alpha * 0.80, 0, img)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, (12, y + bar_h - 9), font, 0.65, (0, 0, 0), 3)
    cv2.putText(img, text, (11, y + bar_h - 10), font, 0.65, (255, 255, 255), 1)

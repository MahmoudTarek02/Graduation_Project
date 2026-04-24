"""
shelf_event_detector.py
-----------------------
Detects when a tracked person's hand enters a shelf zone.

Design
------
ShelfEventDetector.process(frame, tracks, fuse_id)
    - Runs the hand tracker on the frame
    - Associates each detected hand to a person (via bounding-box proximity)
    - Checks hand fingertip against every defined zone polygon
    - Fires event callbacks on zone entry (with per-person-per-zone cooldown)
    - Draws overlay (zones, hand landmarks, event banners) on frame in-place

Callbacks
---------
Register any callable with register_callback(fn).
It receives an event dict:
    {
        'person_id' : int,           # fused/canonical person ID
        'zone_id'   : int,           # shelf number
        'handedness': 'Left'|'Right',
        'position'  : (x, y),        # fingertip pixel position
        'type'      : 'enter'        # 'enter' | 'exit'  (future-proof)
    }
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

import cv2
import numpy as np

from hand_tracker import HandTrackerBase


# ── colours (BGR) ──────────────────────────────────────────────────────────────
_ZONE_COLORS = [
    (50, 220, 50),
    (50, 100, 255),
    (255, 50, 50),
    (50, 220, 220),
    (220, 50, 220),
    (0, 180, 255),
]

_BANNER_DURATION = 60   # frames an on-screen event banner stays visible


def _zone_color(zid: int) -> tuple:
    return _ZONE_COLORS[(zid - 1) % len(_ZONE_COLORS)]


# ── main class ─────────────────────────────────────────────────────────────────

class ShelfEventDetector:
    """
    Parameters
    ----------
    zones       : dict returned by ShelfZoneSelector.select()
                  {zone_id: [(x, y), ...]}
    hand_tracker: Any HandTrackerBase subclass
    cooldown_frames: frames to wait before re-firing the same
                     (person_id, zone_id) enter event
    person_bbox_expand: fractional expansion of person bounding box
                        when associating a hand to a person (helps
                        with outstretched arms)
    """

    def __init__(
        self,
        zones: dict,
        hand_tracker: HandTrackerBase,
        cooldown_frames: int = 60,
        person_bbox_expand: float = 0.5,
    ):
        self._zones = {
            zid: np.array(pts, np.int32) for zid, pts in zones.items()
        }
        self._tracker = hand_tracker
        self._cooldown_max = cooldown_frames
        self._expand = person_bbox_expand

        # State
        self._cooldown: dict[tuple, int] = {}          # (pid, zid) → frames left
        self._in_zone: dict[tuple, bool] = defaultdict(bool)  # (pid, zid) → bool

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
        Run hand detection, zone checking, and drawing.

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

        hands = self._tracker.detect(frame)
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

        for hand in hands:
            tip = hand["fingertip"]
            person_id = self._associate(hand, tracks, fuse_id)

            for zid, poly in self._zones.items():
                inside = _point_in_poly(tip, poly)
                key = (person_id, zid)
                was_inside = self._in_zone[key]

                if inside and not was_inside and key not in self._cooldown:
                    # ── ENTRY EVENT ───────────────────────────────────────────
                    self._in_zone[key] = True
                    self._cooldown[key] = self._cooldown_max

                    ev = {
                        "person_id": person_id,
                        "zone_id": zid,
                        "handedness": hand["handedness"],
                        "position": tip,
                        "type": "enter",
                    }
                    events.append(ev)

                    # Console
                    msg = (
                        f"[SHELF EVENT] Person {person_id} "
                        f"({hand['handedness']} hand) → Shelf {zid}"
                    )
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

                elif not inside:
                    if was_inside:
                        # optional future: fire 'exit' event here
                        pass
                    self._in_zone[key] = False

        # ── Draw everything ───────────────────────────────────────────────────
        self._draw(frame, hands, tracks, fuse_id)

        return events

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self, frame, hands, tracks, fuse_id):
        # Zone fills (semi-transparent)
        overlay = frame.copy()
        for zid, poly in self._zones.items():
            cv2.fillPoly(overlay, [poly], _zone_color(zid))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Zone borders & labels
        for zid, poly in self._zones.items():
            color = _zone_color(zid)
            # Highlight if any hand is inside right now
            active = any(
                self._in_zone[(pid, zid)]
                for pid in _all_person_ids(fuse_id)
            )
            thickness = 3 if active else 2
            cv2.polylines(frame, [poly], True, color, thickness)
            cx, cy = np.mean(poly, axis=0).astype(int)
            _label(frame, f"Shelf {zid}", (cx, cy), color)

        # Hand landmarks
        for hand in hands:
            for lm in hand["landmarks"]:
                cv2.circle(frame, lm, 3, (0, 230, 230), -1)
            cv2.circle(frame, hand["wrist"], 8, (0, 180, 255), -1)
            cv2.circle(frame, hand["fingertip"], 8, (60, 60, 255), -1)

        # Event banners (stacked at bottom)
        if self._banners:
            bh = 36
            base_y = frame.shape[0] - bh * len(self._banners)
            for i, (msg, frames_left) in enumerate(self._banners.items()):
                alpha = min(1.0, frames_left / 15)   # fade-out in last 15 frames
                _banner(frame, msg, base_y + i * bh, alpha)

    # ── Person association ──────────────────────────────────────────────────────

    def _associate(
        self,
        hand: dict,
        tracks: np.ndarray,
        fuse_id: dict,
    ) -> int | None:
        """
        Return the canonical (fused) person ID for a detected hand.

        Strategy
        --------
        1. Check each tracked bounding box expanded by `_expand` fraction.
           If the wrist falls inside, pick the box whose *bottom-centre*
           is closest to the wrist (handles overlapping boxes near shelves).
        2. If no box contains the wrist, fall back to the overall closest
           bottom-centre (arm-reach scenario).

        Finally, resolve the tracker ID to the canonical fused ID.
        """
        if tracks is None or len(tracks) == 0:
            return None

        wrist = hand["wrist"]
        wx, wy = wrist

        inside_candidates = []
        all_candidates = []

        for row in tracks:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tid = int(row[4])

            bw = x2 - x1
            bh = y2 - y1
            mx = bw * self._expand
            my = bh * self._expand

            # Expanded box — extend sideways and downward (arm reaches forward)
            ex1 = x1 - mx
            ey1 = y1
            ex2 = x2 + mx
            ey2 = y2 + my

            # Bottom-centre of the person box
            bc = ((x1 + x2) / 2, y2)
            dist = math.hypot(wx - bc[0], wy - bc[1])

            all_candidates.append((dist, tid))

            if ex1 <= wx <= ex2 and ey1 <= wy <= ey2:
                inside_candidates.append((dist, tid))

        pool = inside_candidates if inside_candidates else all_candidates
        pool.sort(key=lambda t: t[0])
        best_tid = pool[0][1]

        return _resolve_fused_id(best_tid, fuse_id)


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

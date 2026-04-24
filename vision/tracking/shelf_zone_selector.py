"""
shelf_zone_selector.py
----------------------
Displays the first video frame and lets the user draw polygon ROIs
for each shelf. Returns a dict of {shelf_id: [(x, y), ...]} polygons.

Controls
--------
Left-click  : add vertex to current zone polygon
Right-click : close & save current zone, start next
n           : same as right-click (next zone)
z           : undo last vertex
r           : restart (clear all zones)
Enter       : finish — confirm all zones and exit
Escape      : quit without saving
"""

import cv2
import numpy as np


# One distinct color per zone (BGR)
_ZONE_COLORS = [
    (50, 220, 50),    # green
    (50, 100, 255),   # orange-red
    (255, 50, 50),    # blue
    (50, 220, 220),   # yellow
    (220, 50, 220),   # magenta
    (0, 180, 255),    # gold
]


def _zone_color(zone_id: int) -> tuple:
    return _ZONE_COLORS[(zone_id - 1) % len(_ZONE_COLORS)]


class ShelfZoneSelector:
    """
    Usage
    -----
        selector = ShelfZoneSelector()
        zones = selector.select(first_frame)
        # zones: {1: [(x,y), ...], 2: [...], ...}
    """

    WINDOW = "Draw Shelf Zones - press Enter when done"

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._zones: dict[int, list] = {}       # completed zones
        self._cur_id: int = 1
        self._cur_pts: list = []
        self._mouse_pos: tuple = (0, 0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(self, frame: np.ndarray) -> dict[int, list]:
        self._frame = frame.copy()
        self._zones = {}
        self._cur_id = 1
        self._cur_pts = []

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._mouse_cb)

        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 13:          # Enter — done
                self._save_current()
                break
            elif key == ord("n"):  # next zone (same as right-click)
                self._save_current()
            elif key == ord("z"):  # undo last vertex
                if self._cur_pts:
                    self._cur_pts.pop()
                    self._redraw()
            elif key == ord("r"):  # reset everything
                self._zones = {}
                self._cur_id = 1
                self._cur_pts = []
                self._redraw()
            elif key == 27:        # Escape — abort
                self._zones = {}
                break

        cv2.destroyWindow(self.WINDOW)

        if self._zones:
            print(f"[ShelfZoneSelector] Zones saved: {list(self._zones.keys())}")
        else:
            print("[ShelfZoneSelector] No zones defined — running without zone detection.")

        return self._zones

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_current(self):
        if len(self._cur_pts) < 3:
            if self._cur_pts:
                print(f"[ShelfZoneSelector] Zone {self._cur_id} needs ≥3 points, skipped.")
            self._cur_pts = []
            return

        self._zones[self._cur_id] = self._cur_pts.copy()
        print(
            f"[ShelfZoneSelector] Shelf {self._cur_id} saved "
            f"({len(self._cur_pts)} vertices)"
        )
        self._cur_id += 1
        self._cur_pts = []
        self._redraw()

    def _mouse_cb(self, event, x, y, flags, param):
        self._mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._cur_pts.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._save_current()
        elif event == cv2.EVENT_MOUSEMOVE:
            self._redraw()   # live preview line

    def _redraw(self):
        canvas = self._frame.copy()

        # --- Completed zones ---
        overlay = canvas.copy()
        for zid, pts in self._zones.items():
            poly = np.array(pts, np.int32)
            color = _zone_color(zid)
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

        for zid, pts in self._zones.items():
            poly = np.array(pts, np.int32)
            color = _zone_color(zid)
            cv2.polylines(canvas, [poly], True, color, 2)
            cx, cy = np.mean(poly, axis=0).astype(int)
            _label(canvas, f"Shelf {zid}", (cx, cy), color, scale=0.85)

        # --- Current in-progress zone ---
        if self._cur_pts:
            color = _zone_color(self._cur_id)
            for pt in self._cur_pts:
                cv2.circle(canvas, pt, 5, color, -1)
            # Edges drawn so far
            if len(self._cur_pts) > 1:
                cv2.polylines(
                    canvas, [np.array(self._cur_pts)], False, color, 2
                )
            # Preview line to mouse
            cv2.line(canvas, self._cur_pts[-1], self._mouse_pos, color, 1, cv2.LINE_AA)

        # --- HUD ---
        lines = [
            f"Drawing: Shelf {self._cur_id}  |  "
            f"Zones done: {list(self._zones.keys()) or 'none'}",
            "LClick: add point    RClick / N: next zone    Z: undo    R: reset    Enter: finish",
        ]
        _hud(canvas, lines)

        cv2.imshow(self.WINDOW, canvas)


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------

def _label(img, text, center, color, scale=0.75):
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - tw // 2
    y = center[1] + th // 2
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


def _hud(img, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 10
    line_h = 24
    total_h = pad * 2 + line_h * len(lines)
    cv2.rectangle(img, (0, 0), (img.shape[1], total_h), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        y = pad + line_h * i + 16
        cv2.putText(img, line, (pad + 1, y + 1), font, 0.55, (0, 0, 0), 2)
        cv2.putText(img, line, (pad, y), font, 0.55, (230, 230, 230), 1)

"""
FTC DECODE 2025-2026 - CV Judge v5 (FINAL PRODUCTION)
====================================================
KEY INSIGHT: Green balls have HSV too similar to background.
Solution: ONLY detect in a tight floor region + strict size/shape.

Performance: Downscale processing, skip frames, minimal drawing.

ArUco: DICT_4X4_1000
Basket IDs auto-assigned by left/right position.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import math
import sys


class BallTracker:
    def __init__(self, max_gone=12, max_dist=70):
        self.nid = 0
        self.objs = {}   # id -> (cx, cy, color, area)
        self.gone = {}
        self.mg = max_gone
        self.md = max_dist

    def update(self, dets):
        if not dets:
            for o in list(self.gone):
                self.gone[o] += 1
                if self.gone[o] > self.mg:
                    self.objs.pop(o, None); self.gone.pop(o, None)
            return self.objs

        if not self.objs:
            for d in dets:
                self.objs[self.nid] = d; self.gone[self.nid] = 0; self.nid += 1
            return self.objs

        oids = list(self.objs.keys())
        oc = np.array([(self.objs[o][0], self.objs[o][1]) for o in oids])
        dc = np.array([(d[0], d[1]) for d in dets])
        D = np.linalg.norm(oc[:, None] - dc[None, :], axis=2)

        ur, uc = set(), set()
        for idx in np.argsort(D, axis=None):
            r, c = divmod(int(idx), len(dets))
            if r in ur or c in uc: continue
            if D[r, c] > self.md: break
            ur.add(r); uc.add(c)
            self.objs[oids[r]] = dets[c]; self.gone[oids[r]] = 0

        for r in range(len(oids)):
            if r not in ur:
                self.gone[oids[r]] = self.gone.get(oids[r], 0) + 1
                if self.gone[oids[r]] > self.mg:
                    self.objs.pop(oids[r], None); self.gone.pop(oids[r], None)

        for c in range(len(dets)):
            if c not in uc:
                self.objs[self.nid] = dets[c]; self.gone[self.nid] = 0; self.nid += 1
        return self.objs


class FTCDecodeJudge:
    def __init__(self):
        self.red_auto = self.red_teleop = self.blue_auto = self.blue_teleop = 0
        self.state = 'STOPPED'
        self.start_time = None
        self.match_time = 0.0
        self.AUTO_DUR, self.TRANS_DUR, self.TELEOP_DUR = 30, 8, 120
        self.PTS = 3

        # ArUco DICT_4X4_1000
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        ap = aruco.DetectorParameters()
        ap.adaptiveThreshWinSizeMin = 3
        ap.adaptiveThreshWinSizeMax = 30
        ap.adaptiveThreshWinSizeStep = 5
        ap.minMarkerPerimeterRate = 0.005
        ap.maxMarkerPerimeterRate = 4.0
        ap.polygonalApproxAccuracyRate = 0.08
        ap.minCornerDistanceRate = 0.01
        self.aruco_det = aruco.ArucoDetector(self.aruco_dict, ap)

        self.baskets = {}
        self.basket_radius = 80
        self.manual_baskets = False
        self.setting_basket = None

        # ═══════════════════════════════════════════
        # BALL HSV - TIGHTENED based on real data
        # ═══════════════════════════════════════════
        # Purple: H=125-180 + H=0-10, needs S>30, V>35
        # But purple also overlaps with dark blue backgrounds
        # So we require HIGHER saturation for purple
        self.purple_ranges = [
            (np.array([140, 50, 50]), np.array([180, 255, 220])),
            (np.array([0, 50, 50]),   np.array([10, 255, 220])),
        ]
        # Green: H=40-90, S>40, V>40
        # KEY: require higher S and V to reject gray walls/floor
        self.green_ranges = [
            (np.array([40, 45, 40]), np.array([90, 255, 200])),
        ]

        # ═══════════════════════════════════════════
        # STRICT BALL FILTERING
        # ═══════════════════════════════════════════
        self.min_ball_area = 300      # BIGGER minimum - reject tiny noise
        self.max_ball_area = 8000
        self.min_circularity = 0.50   # STRICTER - must be round
        self.min_solidity = 0.70      # Must be solid (not hollow)

        # ═══════════════════════════════════════════
        # ROI - ONLY the field floor area
        # ═══════════════════════════════════════════
        # Your field is in the bottom-center of the frame
        # We need to EXCLUDE: walls, ceiling, posters, people sitting on sides
        self.roi_y_top = 0.50     # ONLY bottom 50% of frame
        self.roi_y_bot = 0.95     # exclude very bottom (your desk/hands)
        self.roi_x_left = 0.05    # slight margin
        self.roi_x_right = 0.95

        # Processing
        self.process_width = 640  # downscale for detection (FAST!)
        self.frame_skip = 2       # process every 2nd frame
        self.frame_n = 0

        # Tracking
        self.tracker = BallTracker(max_gone=15, max_dist=60)
        self.scored_ids = set()
        self.events = []

        # UI
        self.show_masks = False
        self.show_debug = False

    def detect_balls(self, frame):
        """Detect balls in downscaled ROI only."""
        h_f, w_f = frame.shape[:2]

        # Extract ROI
        y1 = int(h_f * self.roi_y_top)
        y2 = int(h_f * self.roi_y_bot)
        x1 = int(w_f * self.roi_x_left)
        x2 = int(w_f * self.roi_x_right)
        roi = frame[y1:y2, x1:x2]

        # Downscale ROI for speed
        rh, rw = roi.shape[:2]
        scale = self.process_width / rw if rw > self.process_width else 1.0
        if scale < 1.0:
            small = cv2.resize(roi, (int(rw * scale), int(rh * scale)))
        else:
            small = roi
            scale = 1.0

        hsv = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), (5, 5), 0)
        dets = []

        for cname, ranges in [('purple', self.purple_ranges), ('green', self.green_ranges)]:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Scale area thresholds for downscaled image
            min_a = self.min_ball_area * (scale * scale)
            max_a = self.max_ball_area * (scale * scale)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_a or area > max_a:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circ = 4 * math.pi * area / (peri * peri)
                if circ < self.min_circularity:
                    continue

                # Solidity
                hull = cv2.convexHull(cnt)
                ha = cv2.contourArea(hull)
                if ha > 0 and (area / ha) < self.min_solidity:
                    continue

                # Aspect ratio
                bx, by, bw, bh = cv2.boundingRect(cnt)
                ar = bw / bh if bh > 0 else 0
                if ar < 0.5 or ar > 2.0:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx_s = int(M["m10"] / M["m00"])
                cy_s = int(M["m01"] / M["m00"])

                # Scale back to full frame coordinates
                cx_full = int(cx_s / scale) + x1
                cy_full = int(cy_s / scale) + y1
                area_full = area / (scale * scale)

                dets.append((cx_full, cy_full, cname, area_full))

        return dets

    def detect_baskets(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        corners, ids, _ = self.aruco_det.detectMarkers(gray)
        detected = []

        if ids is not None:
            mid_x = frame.shape[1] // 2
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                # Auto-assign by position: left=blue, right=red
                alli = 'blue' if cx < mid_x else 'red'
                self.baskets[alli] = (cx, cy, self.basket_radius)
                detected.append((alli, int(marker_id), cx, cy, corners[i]))
        return detected

    def check_scoring(self, tracked):
        if self.state not in ('AUTO', 'TELEOP'):
            return
        for bid, (cx, cy, color, area) in tracked.items():
            if bid in self.scored_ids: continue
            for alli, (bx, by, br) in self.baskets.items():
                if math.hypot(cx - bx, cy - by) < br:
                    self.scored_ids.add(bid)
                    per = 'AUTO' if self.state == 'AUTO' else 'TELEOP'
                    if alli == 'red':
                        if self.state == 'AUTO': self.red_auto += self.PTS
                        else: self.red_teleop += self.PTS
                    else:
                        if self.state == 'AUTO': self.blue_auto += self.PTS
                        else: self.blue_teleop += self.PTS
                    self.events.append({'t': round(self.match_time, 1), 'p': per,
                                        'a': alli, 'c': color, 'id': bid})
                    print(f"  ⚽ {color} #{bid} → {alli.upper()} (+{self.PTS}, {per})")
                    break

    def update_state(self):
        if self.state in ('STOPPED', 'ENDED'): return
        el = time.time() - self.start_time
        self.match_time = el
        if self.state == 'AUTO' and el >= self.AUTO_DUR:
            self.state = 'TRANSITION'; print("\n⏸ AUTO → TRANSITION")
        elif self.state == 'TRANSITION' and el >= self.AUTO_DUR + self.TRANS_DUR:
            self.state = 'TELEOP'; print("\n🎮 TELEOP!")
        elif self.state == 'TELEOP' and el >= self.AUTO_DUR + self.TRANS_DUR + self.TELEOP_DUR:
            self.state = 'ENDED'; print("\n🏁 ENDED!"); self.print_scores()

    def time_str(self):
        if not self.start_time: return "0:00"
        el = time.time() - self.start_time
        if el < self.AUTO_DUR:
            return f"AUTO 0:{max(0,int(self.AUTO_DUR-el)):02d}"
        elif el < self.AUTO_DUR + self.TRANS_DUR:
            return f"TRANS {max(0,int(self.AUTO_DUR+self.TRANS_DUR-el))}"
        else:
            rem = max(0, self.TELEOP_DUR - (el - self.AUTO_DUR - self.TRANS_DUR))
            return f"TELEOP {int(rem)//60}:{int(rem)%60:02d}"

    # ─── DRAWING (minimal for speed) ──────────
    def draw_all(self, frame, tracked, markers):
        h, w = frame.shape[:2]

        # HUD background
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        sc = {'STOPPED': (128, 128, 128), 'AUTO': (0, 255, 255),
              'TRANSITION': (0, 165, 255), 'TELEOP': (0, 255, 0),
              'ENDED': (0, 0, 255)}.get(self.state, (255, 255, 255))

        cv2.putText(frame, f"{self.state}  {self.time_str()}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)

        nb = len(tracked)
        nbk = len(self.baskets)
        ns = len(self.scored_ids)
        cv2.putText(frame, f"Balls:{nb}  Baskets:{nbk}/2  Scored:{ns}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Scores
        rt = self.red_auto + self.red_teleop
        bt = self.blue_auto + self.blue_teleop
        cv2.putText(frame, "RED", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"A:{self.red_auto} T:{self.red_teleop} = {rt}",
                    (65, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, "BLUE", (w // 2, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(frame, f"A:{self.blue_auto} T:{self.blue_teleop} = {bt}",
                    (w // 2 + 70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Big totals
        cv2.putText(frame, f"RED: {rt}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"BLUE: {bt}", (w // 2, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)

        # ROI box
        y1 = int(h * self.roi_y_top)
        y2 = int(h * self.roi_y_bot)
        x1 = int(w * self.roi_x_left)
        x2 = int(w * self.roi_x_right)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 1)

        # Baskets
        for alli, (bx, by, br) in self.baskets.items():
            c = (0, 0, 255) if alli == 'red' else (255, 100, 0)
            cv2.circle(frame, (bx, by), br, c, 2)
            cv2.putText(frame, f"{alli[0].upper()} GOAL", (bx - 30, by - br - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

        # Markers
        for alli, mid, cx, cy, corn in markers:
            c = (0, 0, 255) if alli == 'red' else (255, 100, 0)
            pts = corn[0].astype(int)
            cv2.polylines(frame, [pts], True, c, 2)

        # Balls (minimal drawing)
        for bid, (cx, cy, color, area) in tracked.items():
            dc = (200, 0, 200) if color == 'purple' else (0, 200, 0)
            r = max(8, min(int(math.sqrt(area / math.pi)), 30))
            scored = bid in self.scored_ids
            if scored:
                cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
            else:
                cv2.circle(frame, (cx, cy), r, dc, 2)
                cv2.putText(frame, f"#{bid}", (cx - 8, cy - r - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, dc, 1)

        # Controls hint
        cv2.putText(frame, "S:start R:reset M:masks 1/2:baskets +/-:radius F:speed Q:quit",
                    (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        return frame

    def show_color_masks(self, frame):
        h_f, w_f = frame.shape[:2]
        y1 = int(h_f * self.roi_y_top)
        y2 = int(h_f * self.roi_y_bot)
        x1 = int(w_f * self.roi_x_left)
        x2 = int(w_f * self.roi_x_right)
        roi = frame[y1:y2, x1:x2]

        # Downscale
        rh, rw = roi.shape[:2]
        scale = self.process_width / rw if rw > self.process_width else 1.0
        if scale < 1.0:
            small = cv2.resize(roi, (int(rw * scale), int(rh * scale)))
        else:
            small = roi

        hsv = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), (5, 5), 0)

        pm = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.purple_ranges:
            pm = cv2.bitwise_or(pm, cv2.inRange(hsv, lo, hi))
        gm = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.green_ranges:
            gm = cv2.bitwise_or(gm, cv2.inRange(hsv, lo, hi))

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pm = cv2.morphologyEx(cv2.morphologyEx(pm, cv2.MORPH_OPEN, k, iterations=2), cv2.MORPH_CLOSE, k, iterations=2)
        gm = cv2.morphologyEx(cv2.morphologyEx(gm, cv2.MORPH_OPEN, k, iterations=2), cv2.MORPH_CLOSE, k, iterations=2)

        sh, sw = small.shape[:2]
        out = np.zeros((sh, sw * 3, 3), dtype=np.uint8)
        out[:, :sw] = small  # original
        out[:, sw:sw*2, 0] = pm; out[:, sw:sw*2, 2] = pm
        out[:, sw*2:, 1] = gm

        cv2.putText(out, "ORIGINAL", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(out, "PURPLE", (sw + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(out, "GREEN", (sw * 2 + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return out

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.setting_basket:
            self.baskets[self.setting_basket] = (x, y, self.basket_radius)
            self.manual_baskets = True
            print(f"  ✅ {self.setting_basket.upper()} at ({x},{y})")
            self.setting_basket = None

    def print_scores(self):
        rt, bt = self.red_auto + self.red_teleop, self.blue_auto + self.blue_teleop
        print(f"\n{'=' * 45}")
        print(f"  RED:  A:{self.red_auto} T:{self.red_teleop} = {rt}")
        print(f"  BLUE: A:{self.blue_auto} T:{self.blue_teleop} = {bt}")
        r = "RED" if rt > bt else "BLUE" if bt > rt else "TIE"
        print(f"  WINNER: {r} ({rt}-{bt})")
        print(f"{'=' * 45}\n")

    def save_report(self):
        rpt = {'game': 'FTC DECODE 2025-2026',
               'red': self.red_auto + self.red_teleop,
               'blue': self.blue_auto + self.blue_teleop,
               'events': self.events}
        fn = f'match_{int(time.time())}.json'
        with open(fn, 'w') as f:
            json.dump(rpt, f, indent=2)
        print(f"  Saved: {fn}")

    def reset(self):
        self.red_auto = self.red_teleop = self.blue_auto = self.blue_teleop = 0
        self.state = 'STOPPED'; self.start_time = None; self.match_time = 0
        self.tracker = BallTracker(max_gone=15, max_dist=60)
        self.scored_ids.clear(); self.events.clear()
        if not self.manual_baskets: self.baskets.clear()

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open {source}"); return

        # Camera settings for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Reduce buffer to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        win = 'FTC DECODE 2025-2026 v5'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.mouse_cb)

        print(f"\n{'=' * 45}")
        print("  FTC DECODE 2025-2026 - Judge v5")
        print(f"{'=' * 45}")
        print("  S:start R:reset P:scores Q:quit")
        print("  M:masks 1:RED basket 2:BLUE basket")
        print("  +/-:radius F:speed(skip frames)")
        print(f"{'=' * 45}\n")

        tracked = {}
        markers = []

        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            key = cv2.waitKey(1) & 0xFF
            self.frame_n += 1

            if key == ord('q'): self.save_report(); break
            elif key == ord('s') and self.state in ('STOPPED', 'ENDED'):
                self.reset(); self.state = 'AUTO'; self.start_time = time.time()
                print("\n🚀 MATCH → AUTO (30s)")
            elif key == ord('r'): self.reset(); print("  Reset")
            elif key == ord('p'): self.print_scores()
            elif key == ord('m'): self.show_masks = not self.show_masks
            elif key == ord('d'): self.show_debug = not self.show_debug
            elif key == ord('f'):
                self.frame_skip = 4 if self.frame_skip <= 2 else 1
                print(f"  Skip: {self.frame_skip}")
            elif key == ord('1'): self.setting_basket = 'red'; print("  Click RED...")
            elif key == ord('2'): self.setting_basket = 'blue'; print("  Click BLUE...")
            elif key in (ord('+'), ord('=')):
                self.basket_radius = min(200, self.basket_radius + 10)
                for k in self.baskets:
                    bx, by, _ = self.baskets[k]; self.baskets[k] = (bx, by, self.basket_radius)
                print(f"  Radius: {self.basket_radius}")
            elif key == ord('-'):
                self.basket_radius = max(20, self.basket_radius - 10)
                for k in self.baskets:
                    bx, by, _ = self.baskets[k]; self.baskets[k] = (bx, by, self.basket_radius)
                print(f"  Radius: {self.basket_radius}")

            self.update_state()

            # Only process every Nth frame
            if self.frame_n % self.frame_skip == 0:
                markers = self.detect_baskets(frame)
                balls = self.detect_balls(frame)
                tracked = self.tracker.update(balls)
                self.check_scoring(tracked)

            # Draw
            disp = self.draw_all(frame, tracked, markers)

            if self.setting_basket:
                cv2.putText(disp, f"CLICK {self.setting_basket.upper()}",
                            (10, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(win, disp)

            if self.show_masks:
                cv2.imshow('Masks', self.show_color_masks(frame))
            else:
                try:
                    if cv2.getWindowProperty('Masks', cv2.WND_PROP_VISIBLE) > 0:
                        cv2.destroyWindow('Masks')
                except:
                    pass

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    j = FTCDecodeJudge()

    # ╔══════════════════════════════════════════════╗
    # ║  QUICK TUNING GUIDE                          ║
    # ╠══════════════════════════════════════════════╣
    # ║  Too few balls? → Lower min_ball_area,       ║
    # ║    lower min_circularity, widen HSV           ║
    # ║  Too many false? → Raise min_ball_area,      ║
    # ║    raise min_circularity, narrow ROI          ║
    # ║  Slow FPS? → Increase frame_skip,            ║
    # ║    lower process_width                        ║
    # ║  Baskets wrong? → Press 1/2 to click set     ║
    # ╚══════════════════════════════════════════════╝

    # j.min_ball_area = 200       # lower = more balls detected
    # j.min_circularity = 0.40    # lower = accept less round shapes
    # j.basket_radius = 100       # bigger = easier scoring
    # j.frame_skip = 3            # higher = faster but less responsive
    # j.process_width = 480       # lower = faster processing
    # j.roi_y_top = 0.45          # adjust detection zone top
    # j.roi_y_bot = 0.92          # adjust detection zone bottom

    src = 0
    if len(sys.argv) > 1:
        try:
            src = int(sys.argv[1])
        except:
            src = sys.argv[1]

    j.run(source=src)

"""
FTC 2025-2026 DECODE presented by RTX — Computer Vision Judge System v5

NEW in v5:
  - Complete FOUL system: MINOR (5pts) and MAJOR (15pts) per official manual
  - Keyboard shortcuts to assign fouls with rule codes
  - BASE scoring: partial (5pts), full (10pts), both-robots bonus (10pts)
  - LEAVE scoring (3pts per robot at end of AUTO)
  - Claude AI Agent mode: async review of ambiguous events
  - All v4 entry-based scoring preserved (no phantom scoring)

Official Foul Rules (Section 10.6 / 11):
  MINOR FOUL = 5 pts credited to OPPONENT
  MAJOR FOUL = 15 pts credited to OPPONENT
  Key rules:
    G301  Delay              VERBAL -> MAJOR FOUL
    G403  Movement in transition  MAJOR FOUL
    G404  Movement after match    MINOR -> MAJOR FOUL
    G408  >3 artifacts control    MINOR FOUL per extra
    G416  Launch outside zone     MINOR -> MAJOR FOUL
    G417  Opponent gate contact   MAJOR FOUL + PATTERN RP
    G418  Meddle with ramp        MAJOR FOUL per artifact
    G419  Launch into own ramp    MAJOR FOUL per artifact
    G420  Combat robotics         MAJOR FOUL + YELLOW CARD

Official Scoring (Table 10-2):
  CLASSIFIED=3 | OVERFLOW=1 | PATTERN=2/match | DEPOT=1 | LEAVE=3
  BASE partial=5 | BASE full=10 | Both robots full=+10
  MINOR FOUL=5 to opponent | MAJOR FOUL=15 to opponent
  Timer: 30s AUTO + 8s transition + 2:00 TELEOP = 2:38
"""

import os

_font_dirs = ["/usr/share/fonts", "/usr/share/fonts/truetype/dejavu",
              os.path.expanduser("~/.fonts")]
for _fp in _font_dirs:
    if os.path.isdir(_fp):
        os.environ.setdefault("QT_QPA_FONTDIR", _fp)
        break

import cv2
import cv2.aruco as aruco
import numpy as np
from collections import defaultdict
import time
import json
import threading

# ─── Optional: Claude API for smart scoring validation ───────────────
ENABLE_CLAUDE_VALIDATION = False  # Set True if you have an API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
try:
    if ENABLE_CLAUDE_VALIDATION and ANTHROPIC_API_KEY:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("[Claude API] Validation enabled.")
    else:
        claude_client = None
except ImportError:
    claude_client = None
    print("[Claude API] anthropic package not installed. Running without AI validation.")


def validate_scoring_with_claude(event_description: str, callback=None):
    """Send scoring event to Claude for validation (async, non-blocking)."""
    if not claude_client:
        return

    def _call():
        try:
            msg = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        f"FTC DECODE game judge. A scoring event was detected by computer vision:\n"
                        f"{event_description}\n\n"
                        f"Is this likely a legitimate score? Reply with just YES or NO and a brief reason."
                    )
                }]
            )
            result = msg.content[0].text.strip()
            print(f"  [Claude validation] {result}")
            if callback:
                callback(result)
        except Exception as e:
            print(f"  [Claude validation error] {e}")

    threading.Thread(target=_call, daemon=True).start()


class DecodeFTCJudge:
    """FTC 2025-2026 DECODE CV Judge — v5 (fouls + AI agent)."""

    # ── Official Point Values ────────────────────────────────────────
    PTS_CLASSIFIED = 3
    PTS_OVERFLOW   = 1
    PTS_PATTERN    = 2
    PTS_DEPOT      = 1
    PTS_LEAVE      = 3
    PTS_BASE_PARTIAL = 5
    PTS_BASE_FULL    = 10
    PTS_BASE_BOTH    = 10   # bonus: both robots fully returned

    # ── Foul Point Values (credited to OPPONENT) ─────────────────────
    PTS_MINOR_FOUL = 5
    PTS_MAJOR_FOUL = 15

    # ── Foul Catalog (from Section 11 Game Rules) ────────────────────
    FOUL_CATALOG = {
        # key: (severity, rule_code, description)
        'delay':       ('MAJOR', 'G301', 'Delay of match start'),
        'auto_move':   ('MAJOR', 'G403', 'Robot movement during AUTO-TELEOP transition'),
        'end_move':    ('MINOR', 'G404', 'Robot movement after match end'),
        'end_launch':  ('MAJOR', 'G404', 'Robot launched artifact after match end'),
        'end_gate':    ('MAJOR', 'G404', 'Robot contacted gate after match end'),
        'control4':    ('MINOR', 'G408', 'Controlling >3 artifacts (per extra)'),
        'control5':    ('MINOR', 'G408', 'Controlling 5+ artifacts (YELLOW CARD)'),
        'launch_zone': ('MINOR', 'G416', 'Launching outside launch zone'),
        'launch_goal': ('MAJOR', 'G416', 'Launching outside zone into goal'),
        'opp_gate':    ('MAJOR', 'G417', 'Contacted opponent gate (+PATTERN RP)'),
        'ramp_own':    ('MAJOR', 'G418', 'Meddled with own ramp artifacts'),
        'ramp_opp':    ('MAJOR', 'G418', 'Meddled with opponent ramp artifacts'),
        'own_ramp':    ('MAJOR', 'G419', 'Launched onto own ramp'),
        'opp_goal':    ('MAJOR', 'G419', 'Launched into opponent goal/ramp'),
        'combat':      ('MAJOR', 'G420', 'Deliberately impaired opponent robot'),
        'pin':         ('MINOR', 'G421', 'Pinning opponent >3 seconds'),
        'pin_long':    ('MAJOR', 'G421', 'Pinning opponent >10 seconds'),
        'tunnel':      ('MAJOR', 'G425', 'Contact in opponent secret tunnel'),
        'human_reach': ('MAJOR', 'G431', 'Drive team member contact with robot/field'),
        'storage':     ('MINOR', 'G434', 'Alliance area >6 artifacts'),
        'general_minor': ('MINOR', 'G---', 'General minor foul'),
        'general_major': ('MAJOR', 'G---', 'General major foul'),
    }

    MOTIFS = {
        'GPP': ['G', 'P', 'P'],
        'PGP': ['P', 'G', 'P'],
        'PPG': ['P', 'P', 'G'],
    }

    AUTO_DURATION       = 30
    TRANSITION_DURATION = 8
    TELEOP_DURATION     = 120
    TOTAL_DURATION      = AUTO_DURATION + TRANSITION_DURATION + TELEOP_DURATION

    def __init__(self):
        self.scores = {
            a: {'classified': 0, 'overflow': 0, 'pattern': 0, 'depot': 0,
                'leave': 0, 'base': 0, 'fouls_received': 0}
            for a in ('red', 'blue')
        }
        self.active_motif_name = None
        self.active_motif = None
        self.ramp = {'red': [], 'blue': []}

        # ── Foul tracking ────────────────────────────────────────────
        self.fouls = {'red': [], 'blue': []}  # list of (foul_key, time, count)
        self.yellow_cards = {'red': set(), 'blue': set()}  # team numbers with YELLOW
        self.red_cards = {'red': set(), 'blue': set()}

        self.game_state = 'STOPPED'
        self.start_time = None
        self.match_elapsed = 0.0
        self.auto_pattern_assessed = False

        # ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.ROBOT_IDS = {1, 2, 11, 12}
        self.detected_marker_ids = []

        # Color ranges — TUNED for real match conditions
        # Purple: narrowed from v5 (was too wide), raised sat/val to reject
        #         skin tones, blue objects, shadows. Press 'd' to debug.
        # Green: unchanged, works well.
        self.COLOR_RANGES = {
            'purple': {
                'lower': np.array([120, 50, 50]),   # hue 120-160, sat/val >= 50
                'upper': np.array([160, 255, 255]),
            },
            'green': {
                'lower': np.array([35, 50, 50]),
                'upper': np.array([85, 255, 255]),
            },
        }
        self.show_hsv_debug = False  # Toggle with 'd' key

        # ── Ball Tracking ────────────────────────────────────────────
        self.tracked_balls = {}       # bid -> list of {center, timestamp, ...}
        self.next_ball_id = 0
        self.min_ball_area = 300      # raised: reject tiny noise blobs
        self.max_ball_area = 8000     # raised: allow closer balls
        self.min_circularity = 0.6    # raised: reject non-round shapes
        self.TRACK_STALE_SEC = 8.0

        # ── ENTRY-BASED SCORING STATE ────────────────────────────────
        # For each ball ID, track whether it's currently "inside" each zone.
        # A score only triggers on the transition: outside → inside.
        self.ball_zone_state = defaultdict(dict)  # bid -> {zone_name: bool}
        self.scored_ball_ids = set()               # permanently scored

        # ── Zones ────────────────────────────────────────────────────
        self.field_zones = {}
        self.manual_scoring_zones = []   # (name, center, radius, alliance)

        self.frame_width = 640
        self.frame_height = 480
        self.event_log = []

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _code(color_name):
        return 'P' if color_name == 'purple' else 'G'

    # ── Detection ────────────────────────────────────────────────────

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        markers = {}
        self.detected_marker_ids = []
        if ids is not None:
            self.detected_marker_ids = ids.flatten().tolist()
            for i, mid in enumerate(ids.flatten()):
                corner = corners[i][0]
                center = corner.mean(axis=0).astype(int)
                markers[int(mid)] = {
                    'corners': corner, 'center': tuple(center), 'id': int(mid),
                }
        return markers

    def detect_colored_balls(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Blur to reduce noise before color segmentation
        hsv = cv2.GaussianBlur(hsv, (9, 9), 2)
        detected = []
        for color_name, cr in self.COLOR_RANGES.items():
            mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (self.min_ball_area < area < self.max_ball_area):
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circ = 4 * np.pi * area / (peri * peri)
                if circ < self.min_circularity:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected.append({
                    'color': color_name,
                    'center': (cx, cy),
                    'radius': int(np.sqrt(area / np.pi)),
                    'area': area,
                })
        return detected

    # ── Ball Tracking ────────────────────────────────────────────────

    def track_balls(self, balls, frame_time):
        MAX_DIST = 60
        used_ids = set()
        result = []

        for ball in balls:
            bc = np.array(ball['center'])
            best_id, best_d = None, MAX_DIST

            for bid, hist in self.tracked_balls.items():
                if bid in used_ids or not hist:
                    continue
                d = np.linalg.norm(bc - np.array(hist[-1]['center']))
                if d < best_d:
                    best_d, best_id = d, bid

            if best_id is not None:
                ball['id'] = best_id
                used_ids.add(best_id)
            else:
                ball['id'] = self.next_ball_id
                self.tracked_balls[self.next_ball_id] = []
                self.next_ball_id += 1

            ball['timestamp'] = frame_time
            hist = self.tracked_balls[ball['id']]
            if hist:
                dt = frame_time - hist[-1]['timestamp']
                ball['velocity'] = best_d / dt if (best_id is not None and dt > 0) else 0
            else:
                ball['velocity'] = 0

            hist.append(ball)
            if len(hist) > 40:
                self.tracked_balls[ball['id']] = hist[-40:]
            result.append(ball)

        # Prune stale (LONGER timeout for stationary balls)
        cutoff = frame_time - self.TRACK_STALE_SEC
        for bid in [b for b, h in self.tracked_balls.items() if h and h[-1]['timestamp'] < cutoff]:
            del self.tracked_balls[bid]
            # Also clean zone state for this ball
            self.ball_zone_state.pop(bid, None)

        return result

    # ── Field Zones ──────────────────────────────────────────────────

    def update_field_zones(self, markers):
        for mid, md in markers.items():
            if mid not in self.ROBOT_IDS:
                side = "blue" if md['center'][0] < self.frame_width / 2 else "red"
                self.field_zones[f"marker_{side}_{mid}"] = md['center']

    # ── ENTRY-BASED SCORING ──────────────────────────────────────────

    def _get_all_zones(self):
        """Return list of (zone_name, center, radius, alliance) for all zones."""
        zones = []
        # ArUco-detected zones
        for zname, zcenter in self.field_zones.items():
            alliance = 'red' if 'red' in zname else 'blue'
            zones.append((zname, zcenter, 120, alliance))
        # Manual zones
        for name, center, radius, alliance in self.manual_scoring_zones:
            zones.append((name, center, radius, alliance))
        return zones

    def detect_and_score(self, tracked_balls):
        """ENTRY-BASED scoring: only triggers when a ball ENTERS a zone.

        For each ball, for each zone, we track: is it inside right now?
        A score only fires on the transition from outside → inside.
        Once a ball has scored, it's permanently locked out.
        """
        if self.game_state not in ('AUTO', 'TELEOP', 'STOPPED'):
            return

        zones = self._get_all_zones()
        if not zones:
            return

        for ball in tracked_balls:
            bid = ball['id']
            if bid in self.scored_ball_ids:
                continue

            bc = np.array(ball['center'])

            for zname, zcenter, zradius, alliance in zones:
                dist = np.linalg.norm(bc - np.array(zcenter))
                currently_inside = dist < zradius

                # What was the previous state?
                was_inside = self.ball_zone_state[bid].get(zname, False)

                # Update state
                self.ball_zone_state[bid][zname] = currently_inside

                # TRANSITION: outside → inside = SCORE!
                if currently_inside and not was_inside:
                    # Ball must have at least 1 prior observation (not brand-new noise)
                    hist = self.tracked_balls.get(bid, [])
                    if len(hist) < 2:
                        continue

                    self._score_artifact(ball, alliance, zname)
                    break  # one score per ball

    def _score_artifact(self, ball, alliance, zone_name):
        bid = ball['id']
        pts = self.PTS_CLASSIFIED
        code = self._code(ball['color'])

        self.scored_ball_ids.add(bid)
        self.scores[alliance]['classified'] += pts

        if len(self.ramp[alliance]) < 9:
            self.ramp[alliance].append(code)

        self.log_event(alliance.upper(),
                       f'{ball["color"].capitalize()} CLASSIFIED in {zone_name}', pts)
        ramp_str = ''.join(self.ramp[alliance])
        print(f"SCORED! {alliance.upper()} -- {ball['color']} artifact CLASSIFIED: +{pts}  "
              f"(Ramp: {ramp_str})")

        # Optional Claude validation
        if claude_client:
            vel = ball.get('velocity', 0)
            desc = (f"A {ball['color']} ball entered zone '{zone_name}' "
                    f"({alliance} alliance) with velocity {vel:.1f} px/s. "
                    f"Ball was tracked for {len(self.tracked_balls.get(bid, []))} frames.")
            validate_scoring_with_claude(desc)

    # ── Pattern Assessment ───────────────────────────────────────────

    def assess_patterns(self, period_label):
        if self.active_motif is None:
            print(f"[{period_label}] No MOTIF set -- skipping PATTERN assessment.")
            return
        full_pattern = self.active_motif * 3
        for alliance in ('red', 'blue'):
            ramp = self.ramp[alliance]
            matches = sum(1 for i, c in enumerate(ramp)
                          if i < len(full_pattern) and c == full_pattern[i])
            pts = matches * self.PTS_PATTERN
            self.scores[alliance]['pattern'] += pts
            ramp_str = ''.join(ramp) or '(empty)'
            expected = ''.join(full_pattern[:len(ramp)]) or '(n/a)'
            self.log_event(alliance.upper(),
                           f'PATTERN ({period_label}): {matches}/{len(ramp)} match', pts)
            print(f"  PATTERN [{period_label}] {alliance.upper()}: "
                  f"{ramp_str} vs {expected} -> {matches} matches = +{pts}")

    # ─────────────────────────────────────────────────────────────────
    #  FOUL SYSTEM
    # ─────────────────────────────────────────────────────────────────

    def assign_foul(self, violating_alliance, foul_key, count=1):
        """Assign a foul to an alliance. Points go to the OPPONENT.

        Args:
            violating_alliance: 'red' or 'blue' — the alliance that committed the foul
            foul_key: key from FOUL_CATALOG
            count: number of fouls (e.g., G408 is per extra artifact)
        """
        if foul_key not in self.FOUL_CATALOG:
            print(f"Unknown foul key: {foul_key}")
            return

        severity, rule, desc = self.FOUL_CATALOG[foul_key]
        opponent = 'blue' if violating_alliance == 'red' else 'red'
        pts_per = self.PTS_MAJOR_FOUL if severity == 'MAJOR' else self.PTS_MINOR_FOUL
        total_pts = pts_per * count

        # Credit points to opponent
        self.scores[opponent]['fouls_received'] += total_pts

        # Record the foul
        self.fouls[violating_alliance].append({
            'key': foul_key, 'rule': rule, 'severity': severity,
            'desc': desc, 'points': total_pts, 'count': count,
            'time': round(self.match_elapsed, 2),
        })

        self.log_event(violating_alliance.upper(),
                       f'FOUL {severity} {rule}: {desc} (x{count})', -total_pts)

        card_str = ''
        # Check for YELLOW CARD triggers
        if foul_key in ('control5', 'combat', 'tunnel'):
            self.yellow_cards[violating_alliance].add('team')
            card_str = ' + YELLOW CARD'

        print(f"{'!'*3} FOUL: {violating_alliance.upper()} — {severity} {rule}: {desc} "
              f"(x{count} = {total_pts}pts to {opponent.upper()}){card_str}")

        # Claude AI review for context
        if claude_client:
            review_desc = (
                f"FTC DECODE foul called: {severity} {rule} ({desc}) against {violating_alliance} "
                f"alliance at {self.match_elapsed:.1f}s into the match. State: {self.game_state}. "
                f"This is foul #{len(self.fouls[violating_alliance])} for {violating_alliance} this match."
            )
            validate_scoring_with_claude(review_desc)

    def assign_leave(self, alliance):
        """Award LEAVE points (robot off LAUNCH LINE at end of AUTO)."""
        self.scores[alliance]['leave'] += self.PTS_LEAVE
        self.log_event(alliance.upper(), 'LEAVE (robot off launch line)', self.PTS_LEAVE)
        print(f"LEAVE: {alliance.upper()} +{self.PTS_LEAVE}")

    def assign_base(self, alliance, level='full'):
        """Award BASE points at end of match.
        level: 'partial' (5pts), 'full' (10pts), 'both' (10pts bonus)
        """
        if level == 'partial':
            pts = self.PTS_BASE_PARTIAL
            desc = 'BASE partially returned'
        elif level == 'full':
            pts = self.PTS_BASE_FULL
            desc = 'BASE fully returned'
        elif level == 'both':
            pts = self.PTS_BASE_BOTH
            desc = 'BASE both robots fully returned bonus'
        else:
            return
        self.scores[alliance]['base'] += pts
        self.log_event(alliance.upper(), desc, pts)
        print(f"BASE: {alliance.upper()} — {desc}: +{pts}")

    def get_foul_summary(self, alliance):
        """Get summary of fouls for an alliance."""
        minors = sum(1 for f in self.fouls[alliance] if f['severity'] == 'MINOR')
        majors = sum(1 for f in self.fouls[alliance] if f['severity'] == 'MAJOR')
        total_pts = sum(f['points'] for f in self.fouls[alliance])
        return minors, majors, total_pts

    # ── Timer ────────────────────────────────────────────────────────

    def update_timer(self):
        if self.game_state == 'STOPPED' or self.start_time is None:
            return 0, self.TOTAL_DURATION, 'STOPPED'
        if self.game_state == 'FINISHED':
            return self.match_elapsed, 0, 'FINISHED'

        elapsed = time.time() - self.start_time
        self.match_elapsed = min(elapsed, self.TOTAL_DURATION)

        if elapsed < self.AUTO_DURATION:
            state = 'AUTO'
        elif elapsed < self.AUTO_DURATION + self.TRANSITION_DURATION:
            state = 'TRANSITION'
            if not self.auto_pattern_assessed:
                self.auto_pattern_assessed = True
                print("\n--- END OF AUTO ---")
                self.assess_patterns('AUTO')
        elif elapsed < self.TOTAL_DURATION:
            state = 'TELEOP'
        else:
            state = 'FINISHED'
            if self.game_state != 'FINISHED':
                print("\n--- MATCH OVER ---")
                self.assess_patterns('TELEOP_END')
                self.print_scores()

        self.game_state = state
        remaining = max(0, self.TOTAL_DURATION - elapsed)
        return elapsed, remaining, state

    # ── Log ───────────────────────────────────────────────────────────

    def log_event(self, alliance, desc, points):
        self.event_log.append({
            'time': round(self.match_elapsed, 2), 'alliance': alliance,
            'event': desc, 'points': points, 'state': self.game_state,
        })

    # ── Drawing ──────────────────────────────────────────────────────

    def draw_markers(self, frame, markers):
        for mid, md in markers.items():
            cv2.polylines(frame, [md['corners'].astype(int)], True, (0, 255, 0), 2)
            cv2.putText(frame, str(mid), md['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def draw_balls(self, frame, tracked_balls):
        # Precompute HSV for debug overlay
        hsv_frame = None
        if self.show_hsv_debug:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for ball in tracked_balls:
            c, r = ball['center'], ball['radius']
            dc = (200, 0, 200) if ball['color'] == 'purple' else (0, 200, 0)
            scored = ball['id'] in self.scored_ball_ids

            if scored:
                cv2.circle(frame, c, r, (128, 128, 128), 1)
                cv2.putText(frame, "done", (c[0]-15, c[1]-r-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            else:
                cv2.circle(frame, c, r, dc, 2)
                cv2.circle(frame, c, 3, dc, -1)
                label = f"ID:{ball['id']} {self._code(ball['color'])}"
                cv2.putText(frame, label, (c[0]-20, c[1]-r-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, dc, 1)

            # HSV debug: show actual HSV values at ball center
            if hsv_frame is not None:
                y_px = min(max(c[1], 0), frame.shape[0]-1)
                x_px = min(max(c[0], 0), frame.shape[1]-1)
                h, s, v = hsv_frame[y_px, x_px]
                hsv_label = f"H:{h} S:{s} V:{v}"
                cv2.putText(frame, hsv_label, (c[0]-30, c[1]+r+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            # Trail
            hist = self.tracked_balls.get(ball['id'], [])
            if len(hist) > 1:
                pts = np.array([h['center'] for h in hist[-20:]], np.int32)
                cv2.polylines(frame, [pts], False, dc, 1)
        return frame

    def draw_hsv_masks(self, frame):
        """Show purple and green detection masks side-by-side (debug mode)."""
        if not self.show_hsv_debug:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        mask_h = h // 4  # small overlay

        for i, (cname, cr) in enumerate(self.COLOR_RANGES.items()):
            mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_resized = cv2.resize(mask_color, (w // 4, mask_h))
            x_off = i * (w // 4)
            y_off = h - mask_h
            frame[y_off:y_off+mask_h, x_off:x_off+mask_resized.shape[1]] = mask_resized
            cv2.putText(frame, f"{cname} mask", (x_off+5, y_off+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return frame

    def draw_zones(self, frame):
        for zname, zcenter in self.field_zones.items():
            color = (0, 0, 255) if 'red' in zname else (255, 0, 0)
            cv2.circle(frame, zcenter, 120, color, 2)
            cv2.putText(frame, zname.split('_', 1)[-1],
                        (zcenter[0]-30, zcenter[1]-125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        for name, center, radius, alliance in self.manual_scoring_zones:
            color = (0, 165, 255) if alliance == 'red' else (255, 165, 0)
            cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{alliance.upper()} GOAL",
                        (center[0]-40, center[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_scoreboard(self, frame):
        elapsed, remaining, state = self.update_timer()
        h, w = frame.shape[:2]
        overlay = frame.copy()

        panel_h = 260
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (255, 255, 255), 2)

        # State
        sc = {'STOPPED': (128,128,128), 'AUTO': (0,200,255), 'TRANSITION': (0,128,255),
              'TELEOP': (0,255,0), 'FINISHED': (0,0,255)}.get(state, (255,255,255))
        cv2.putText(overlay, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2)

        # Timer (countdown)
        mins, secs = int(remaining // 60), int(remaining % 60)
        tc = (0, 0, 255) if remaining < 30 else (255, 255, 255)
        cv2.putText(overlay, f'{mins}:{secs:02d}', (10, 68),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, tc, 2)

        # MOTIF
        motif_str = self.active_motif_name or 'NOT SET (1/2/3)'
        cv2.putText(overlay, f'MOTIF: {motif_str}', (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Info
        active = sum(1 for h in self.tracked_balls.values() if h)
        cv2.putText(overlay, f'Balls:{active} Scored:{len(self.scored_ball_ids)} Markers:{self.detected_marker_ids}',
                    (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)

        # Per-alliance
        for idx, a in enumerate(('red', 'blue')):
            s = self.scores[a]
            total = sum(s.values())
            x = 10 if idx == 0 else w // 2 + 30
            y0 = 115
            ac = (0, 0, 255) if a == 'red' else (255, 0, 0)
            opp = 'blue' if a == 'red' else 'red'

            cv2.putText(overlay, f'{a.upper()} ALLIANCE', (x, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ac, 2)
            cv2.putText(overlay, f"Class:{s['classified']} Pat:{s['pattern']} Base:{s['base']}",
                        (x, y0+22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

            # Foul info: fouls committed BY this alliance
            mi, ma, fpts = self.get_foul_summary(a)
            fouls_received_pts = s['fouls_received']
            cv2.putText(overlay, f"Fouls rcvd:+{fouls_received_pts} Committed:{mi}m+{ma}M",
                        (x, y0+42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,180,255), 1)

            ramp_str = ''.join(self.ramp[a]) or '-'
            cv2.putText(overlay, f"Ramp:[{ramp_str}]  Leave:{s['leave']}",
                        (x, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

            cv2.putText(overlay, f'TOTAL: {total}',
                        (x, y0+85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ac, 2)

            # Yellow card indicator
            if self.yellow_cards[a]:
                cv2.putText(overlay, 'YC', (x + 200, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # AI validation indicator
        if claude_client:
            cv2.putText(overlay, '[AI]', (w - 50, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return overlay

    # ── Mouse ────────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            alliance = 'red' if x > self.frame_width // 2 else 'blue'
            zid = len(self.manual_scoring_zones) + 1
            self.manual_scoring_zones.append(
                (f'manual_{alliance}_{zid}', (x, y), 100, alliance)
            )
            print(f"+ Manual GOAL zone at ({x},{y}) for {alliance.upper()}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.manual_scoring_zones.clear()
            self.field_zones.clear()
            print("Cleared ALL scoring zones")

    # ── Match Controls ───────────────────────────────────────────────

    def start_match(self):
        self.reset_match(keep_motif=True, keep_zones=True)
        self.game_state = 'AUTO'
        self.start_time = time.time()
        motif_display = self.active_motif_name or '(none — set with 1/2/3 during match)'
        print(f"\n>>> MATCH STARTED -- MOTIF: {motif_display} <<<")
        print(f"    30s AUTO + 8s transition + 2:00 TELEOP = 2:38\n")

    def reset_match(self, keep_motif=False, keep_zones=False):
        self.scores = {
            a: {k: 0 for k in self.scores[a]} for a in ('red', 'blue')
        }
        self.ramp = {'red': [], 'blue': []}
        self.game_state = 'STOPPED'
        self.start_time = None
        self.match_elapsed = 0.0
        self.auto_pattern_assessed = False
        self.scored_ball_ids.clear()
        self.tracked_balls.clear()
        self.ball_zone_state.clear()
        self.event_log.clear()
        self.next_ball_id = 0
        self.fouls = {'red': [], 'blue': []}
        self.yellow_cards = {'red': set(), 'blue': set()}
        self.red_cards = {'red': set(), 'blue': set()}
        if not keep_motif:
            self.active_motif = None
            self.active_motif_name = None
        if not keep_zones:
            self.manual_scoring_zones.clear()
            self.field_zones.clear()

    def set_motif(self, name):
        if name in self.MOTIFS:
            self.active_motif_name = name
            self.active_motif = self.MOTIFS[name]
            print(f"MOTIF set: {name} -> RAMP: {''.join(self.active_motif * 3)}")

    def print_scores(self):
        print("\n" + "=" * 65)
        print("DECODE MATCH SCORES")
        print("=" * 65)
        for a in ('red', 'blue'):
            s = self.scores[a]
            total = sum(s.values())
            ramp_str = ''.join(self.ramp[a]) or '(empty)'
            mi, ma, fpts = self.get_foul_summary(a)
            print(f"\n{a.upper()} ALLIANCE:")
            print(f"  Classified: {s['classified']}  Overflow: {s['overflow']}  Pattern: {s['pattern']}")
            print(f"  Depot: {s['depot']}  Leave: {s['leave']}  Base: {s['base']}")
            print(f"  Fouls received (from opponent): +{s['fouls_received']}")
            print(f"  Fouls committed: {mi} minor + {ma} major = {fpts}pts to opponent")
            print(f"  Ramp: [{ramp_str}]")
            if self.yellow_cards[a]:
                print(f"  ** YELLOW CARD **")
            print(f"  TOTAL: {total}")
        rt, bt = sum(self.scores['red'].values()), sum(self.scores['blue'].values())
        print(f"\n{'RED LEADS' if rt > bt else 'BLUE LEADS' if bt > rt else 'TIED'}")
        print("=" * 65 + "\n")

    def save_report(self, filename='decode_match_report.json'):
        report = {'motif': self.active_motif_name, 'match_elapsed': round(self.match_elapsed, 2)}
        for a in ('red', 'blue'):
            s = self.scores[a]
            mi, ma, fpts = self.get_foul_summary(a)
            report[f'{a}_alliance'] = {
                **s, 'total': sum(s.values()), 'ramp': self.ramp[a],
                'fouls_committed': self.fouls[a],
                'minor_fouls': mi, 'major_fouls': ma,
                'yellow_cards': bool(self.yellow_cards[a]),
            }
        report['event_log'] = self.event_log
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved -> {filename}")

    # ── Manual score adjustments ─────────────────────────────────────

    def manual_add_score(self, alliance, color_code):
        """Manually add a scored artifact (keyboard shortcut)."""
        self.scores[alliance]['classified'] += self.PTS_CLASSIFIED
        if len(self.ramp[alliance]) < 9:
            self.ramp[alliance].append(color_code)
        cname = 'purple' if color_code == 'P' else 'green'
        self.log_event(alliance.upper(), f'Manual {cname} CLASSIFIED', self.PTS_CLASSIFIED)
        print(f"MANUAL SCORE: {alliance.upper()} -- {cname} +{self.PTS_CLASSIFIED}  "
              f"(Ramp: {''.join(self.ramp[alliance])})")

    def manual_undo_last(self, alliance):
        """Undo last scored artifact for an alliance."""
        if self.ramp[alliance]:
            removed = self.ramp[alliance].pop()
            self.scores[alliance]['classified'] -= self.PTS_CLASSIFIED
            cname = 'purple' if removed == 'P' else 'green'
            print(f"UNDO: {alliance.upper()} removed {cname} "
                  f"(Ramp: {''.join(self.ramp[alliance]) or '-'})")
        else:
            print(f"Nothing to undo for {alliance.upper()}")

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("ERROR: Cannot open video source", video_source)
            return

        win = 'FTC DECODE Judge v4'
        ret, first = cap.read()
        if not ret:
            print("ERROR: Cannot read video")
            cap.release()
            return
        self.frame_height, self.frame_width = first.shape[:2]
        cv2.imshow(win, first)
        cv2.waitKey(1)
        cv2.setMouseCallback(win, self.on_mouse)

        print("\nFTC 2025-2026 DECODE Judge System v5 — Fouls + AI Agent")
        print("=" * 65)
        print("SCORING: CLASSIFIED=3  OVERFLOW=1  PATTERN=2/match  DEPOT=1")
        print("         LEAVE=3  BASE partial=5/full=10/both=+10")
        print("FOULS:   MINOR=5pts to opponent  MAJOR=15pts to opponent")
        print("TIMER:   30s AUTO + 8s transition + 2:00 TELEOP = 2:38")
        print()
        print("MATCH CONTROLS:")
        print("  1/2/3  -- Set MOTIF (1=GPP, 2=PGP, 3=PPG)")
        print("  s      -- Start match")
        print("  r      -- Reset match")
        print("  p      -- Print scores")
        print("  a      -- Assess patterns now")
        print()
        print("MANUAL SCORING:")
        print("  F1/F2  -- Red: add Purple/Green artifact")
        print("  F3/F4  -- Blue: add Purple/Green artifact")
        print("  z/x    -- Undo last Red/Blue artifact")
        print()
        print("FOULS (hold key for alliance, then foul type):")
        print("  4      -- MINOR foul on RED  (5pts to BLUE)")
        print("  5      -- MAJOR foul on RED  (15pts to BLUE)")
        print("  6      -- MINOR foul on BLUE (5pts to RED)")
        print("  7      -- MAJOR foul on BLUE (15pts to RED)")
        print("  f      -- Open foul menu (pick specific rule)")
        print()
        print("ENDGAME (assign at end of match):")
        print("  F5/F6  -- LEAVE: Red/Blue robot off launch line")
        print("  F7/F8  -- BASE FULL: Red/Blue robot returned")
        print("  F9/F10 -- BASE PARTIAL: Red/Blue robot partial")
        print("  F11    -- BASE BOTH BONUS: Red  (+10)")
        print("  F12    -- BASE BOTH BONUS: Blue (+10)")
        print()
        print("CV ZONES & DEBUG:")
        print("  Left-click   -- Add manual GOAL zone")
        print("  Right-click  -- Clear ALL zones")
        print("  c            -- Print HSV color ranges")
        print("  d            -- Toggle HSV debug overlay (see masks + values)")
        print("  [ / ]        -- Adjust purple lower hue (-5/+5)")
        print("  q            -- Quit & save report")
        if claude_client:
            print("\n  [Claude AI Agent ENABLED for scoring validation]")
        print("=" * 65 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = time.time()
            key = cv2.waitKey(1) & 0xFF

            # Detect
            markers = self.detect_markers(frame)
            self.update_field_zones(markers)
            balls = self.detect_colored_balls(frame)
            tracked = self.track_balls(balls, frame_time)

            # ── Match Controls ──
            if key == ord('1'):   self.set_motif('GPP')
            elif key == ord('2'): self.set_motif('PGP')
            elif key == ord('3'): self.set_motif('PPG')
            elif key == ord('s'): self.start_match()
            elif key == ord('r'):
                self.reset_match()
                print("Match reset!")
            elif key == ord('p'): self.print_scores()
            elif key == ord('a'):
                print("\nManual PATTERN assessment:")
                self.assess_patterns('MANUAL')
            elif key == ord('c'):
                for cn, cr in self.COLOR_RANGES.items():
                    print(f"  {cn}: {cr}")
            elif key == ord('d'):
                self.show_hsv_debug = not self.show_hsv_debug
                print(f"HSV debug: {'ON' if self.show_hsv_debug else 'OFF'}")
            elif key == ord('q'):
                self.save_report()
                break

            # ── Live purple range tuning ([ and ] adjust lower hue) ──
            elif key == ord('['):
                self.COLOR_RANGES['purple']['lower'][0] = max(0,
                    self.COLOR_RANGES['purple']['lower'][0] - 5)
                print(f"Purple lower hue: {self.COLOR_RANGES['purple']['lower'][0]}")
            elif key == ord(']'):
                self.COLOR_RANGES['purple']['lower'][0] = min(179,
                    self.COLOR_RANGES['purple']['lower'][0] + 5)
                print(f"Purple lower hue: {self.COLOR_RANGES['purple']['lower'][0]}")

            # ── Manual scoring (F1-F4 + undo) ──
            elif key == 190:  self.manual_add_score('red', 'P')   # F1
            elif key == 191:  self.manual_add_score('red', 'G')   # F2
            elif key == 192:  self.manual_add_score('blue', 'P')  # F3
            elif key == 193:  self.manual_add_score('blue', 'G')  # F4
            elif key == ord('z'): self.manual_undo_last('red')
            elif key == ord('x'): self.manual_undo_last('blue')

            # ── Quick Fouls ──
            elif key == ord('4'):  self.assign_foul('red', 'general_minor')
            elif key == ord('5'):  self.assign_foul('red', 'general_major')
            elif key == ord('6'):  self.assign_foul('blue', 'general_minor')
            elif key == ord('7'):  self.assign_foul('blue', 'general_major')

            # ── Specific Foul Menu ──
            elif key == ord('f'):
                self._foul_menu()

            # ── Endgame: LEAVE (F5/F6) ──
            elif key == 194:  self.assign_leave('red')    # F5
            elif key == 195:  self.assign_leave('blue')   # F6

            # ── Endgame: BASE FULL (F7/F8) ──
            elif key == 196:  self.assign_base('red', 'full')     # F7
            elif key == 197:  self.assign_base('blue', 'full')    # F8

            # ── Endgame: BASE PARTIAL (F9/F10) ──
            elif key == 198:  self.assign_base('red', 'partial')  # F9
            elif key == 199:  self.assign_base('blue', 'partial') # F10

            # ── Endgame: BASE BOTH BONUS (F11/F12) ──
            elif key == 200:  self.assign_base('red', 'both')     # F11
            elif key == 201:  self.assign_base('blue', 'both')    # F12

            # Scoring (entry-based)
            if self.game_state in ('AUTO', 'TELEOP', 'STOPPED'):
                self.detect_and_score(tracked)

            # Draw
            display = self.draw_markers(frame, markers)
            display = self.draw_balls(display, tracked)
            display = self.draw_zones(display)
            display = self.draw_hsv_masks(display)
            display = self.draw_scoreboard(display)

            cv2.imshow(win, display)

        cap.release()
        cv2.destroyAllWindows()

    def _foul_menu(self):
        """Interactive terminal foul menu for specific rule codes."""
        print("\n--- FOUL ASSIGNMENT MENU ---")
        print("Alliance? (r)ed / (b)lue: ", end='', flush=True)
        a_key = input().strip().lower()
        alliance = 'red' if a_key == 'r' else 'blue' if a_key == 'b' else None
        if not alliance:
            print("Cancelled.")
            return

        print(f"\nFoul types for {alliance.upper()}:")
        keys = list(self.FOUL_CATALOG.keys())
        for i, k in enumerate(keys):
            sev, rule, desc = self.FOUL_CATALOG[k]
            print(f"  {i:2d}. [{sev:5s}] {rule}: {desc}")

        print(f"\nEnter foul number (0-{len(keys)-1}): ", end='', flush=True)
        try:
            idx = int(input().strip())
            foul_key = keys[idx]
        except (ValueError, IndexError):
            print("Cancelled.")
            return

        print("Count (default 1): ", end='', flush=True)
        count_str = input().strip()
        count = int(count_str) if count_str.isdigit() else 1

        self.assign_foul(alliance, foul_key, count)


if __name__ == "__main__":
    judge = DecodeFTCJudge()
    judge.run(video_source=0)

import argparse
import math
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpack


@dataclass(frozen=True)
class ExpectedEvent:
    idx: int
    frame: int
    t: float
    kind: str


@dataclass(frozen=True)
class VisualNote:
    start_t: float
    end_t: float
    is_hold: bool


@dataclass
class ResultRow:
    idx: int
    kind: str
    expected_t: float
    expected_frame: int
    actual_t: Optional[float]
    offset_ms: Optional[float]
    verdict: str


def load_gdr(path: str) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    obj = msgpack.unpackb(raw, raw=False)
    if not isinstance(obj, dict) or "inputs" not in obj:
        raise ValueError("Not a valid .gdr (expected msgpack dict with 'inputs')")
    return obj


def infer_fps(gdr: Dict[str, Any]) -> float:
    duration = float(gdr.get("duration") or 0.0)
    inputs = gdr.get("inputs") or []
    frames = [int(e.get("frame", 0)) for e in inputs if isinstance(e, dict) and "frame" in e]
    max_frame = max(frames) if frames else 0
    if duration <= 0.0 or max_frame <= 0:
        return 240.0
    fps = max_frame / duration
    if fps < 30:
        return 60.0
    if fps > 1000:
        return 240.0
    return float(fps)


def build_expected_events(gdr: Dict[str, Any], fps: float, btn: int, second_player: bool) -> List[ExpectedEvent]:
    expected: List[ExpectedEvent] = []
    idx = 0
    for e in gdr["inputs"]:
        if not isinstance(e, dict):
            continue
        if int(e.get("btn", -1)) != btn:
            continue
        if bool(e.get("2p", False)) != second_player:
            continue
        frame = int(e.get("frame", 0))
        down = bool(e.get("down", False))
        t = frame / fps
        expected.append(ExpectedEvent(idx=idx, frame=frame, t=t, kind="down" if down else "up"))
        idx += 1
    expected.sort(key=lambda x: x.frame)
    return [ExpectedEvent(idx=i, frame=ev.frame, t=ev.t, kind=ev.kind) for i, ev in enumerate(expected)]


def build_visual_notes(expected: List[ExpectedEvent], hold_min_frames: int) -> List[VisualNote]:
    notes: List[VisualNote] = []
    i = 0
    while i < len(expected):
        ev = expected[i]
        if ev.kind != "down":
            notes.append(VisualNote(start_t=ev.t, end_t=ev.t, is_hold=False))
            i += 1
            continue
        j = i + 1
        while j < len(expected) and expected[j].kind != "up":
            j += 1
        if j < len(expected):
            up = expected[j]
            is_hold = (up.frame - ev.frame) >= hold_min_frames
            notes.append(VisualNote(start_t=ev.t, end_t=up.t, is_hold=is_hold))
            i = j + 1
        else:
            notes.append(VisualNote(start_t=ev.t, end_t=ev.t, is_hold=False))
            i += 1
    return notes


def export_results_text(results: List[ResultRow], out_path: Path, meta: Dict[str, str]) -> None:
    expected_misses = [r for r in results if r.verdict == "miss" and not math.isnan(r.expected_t)]
    extras = [r for r in results if math.isnan(r.expected_t)]
    hits = [r for r in results if r.verdict == "hit" and r.offset_ms is not None and not math.isnan(r.expected_t)]
    hit_offsets = [r.offset_ms for r in hits if r.offset_ms is not None]

    def fmt_ms(x: Optional[float]) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f} ms"

    if hit_offsets:
        mean = sum(hit_offsets) / len(hit_offsets)
        mae = sum(abs(x) for x in hit_offsets) / len(hit_offsets)
        worst = max(hit_offsets, key=lambda v: abs(v))
    else:
        mean = float("nan")
        mae = float("nan")
        worst = float("nan")

    lines: List[str] = []
    lines.append("GD Rhythm Trainer")
    lines.append("=" * 60)
    for k in ["macro_file", "fps", "window_ms", "expected_events", "inputs_captured", "generated_at", "target_x_frac"]:
        if k in meta:
            label = {
                "macro_file": "Macro file",
                "fps": "Detected FPS",
                "window_ms": "Hit window",
                "expected_events": "Expected events",
                "inputs_captured": "Inputs captured",
                "generated_at": "Generated at",
                "target_x_frac": "Target position",
            }[k]
            suffix = " ms" if k == "window_ms" else ""
            if k == "target_x_frac":
                lines.append(f"{label}: {float(meta[k]) * 100:.1f}% of screen width from left")
            else:
                lines.append(f"{label}: {meta[k]}{suffix}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 60)
    lines.append(f"Hits: {len(hits)}")
    lines.append(f"Misses: {len(expected_misses)}")
    lines.append(f"Unexpected inputs: {len(extras)}")
    lines.append(f"Mean hit offset: {fmt_ms(mean)}   (negative=early, positive=late)")
    lines.append(f"Mean abs error: {('n/a' if math.isnan(mae) else f'{mae:.2f} ms')}")
    lines.append(f"Worst hit offset: {fmt_ms(worst)}")
    lines.append("")
    lines.append("Misses (detailed)")
    lines.append("-" * 60)
    if not expected_misses:
        lines.append("No misses!")
    else:
        lines.append("Format: \n[KIND] expected_time  frame    your_action           offset\n")
        for r in expected_misses:
            if r.actual_t is None:
                your_action = "NO INPUT"
                off = ""
            else:
                your_action = f"INPUT @ {r.actual_t:.6f}s"
                off = fmt_ms(r.offset_ms)
            lines.append(f"[{r.kind.upper():4}] t={r.expected_t:.6f}s  f={r.expected_frame:7d}  {your_action:20}  {off}")

    if extras:
        lines.append("")
        lines.append("Unexpected inputs")
        lines.append("-" * 60)
        lines.append("These are presses/releases that did not match any expected event (extra or too far off)")
        for r in extras[:250]:
            if r.actual_t is None:
                continue
            lines.append(f"[{r.kind.upper():4}] input_time={r.actual_t:.6f}s")
        if len(extras) > 250:
            lines.append(f"... ({len(extras) - 250} more omitted)")

    lines.append("")
    lines.append("How offsets work")
    lines.append("-" * 60)
    lines.append("Offset = (your input time - expected time)")
    lines.append("  Negative offset -> you were early")
    lines.append("  Positive offset -> you were late")
    lines.append("If an expected event shows NO INPUT, you never pressed/releases within the hit window for that event")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_compact_stats(results: List[ResultRow], expected_count: int) -> Dict[str, float]:
    hits = [r for r in results if r.verdict == "hit" and r.offset_ms is not None and not math.isnan(r.expected_t)]
    misses = [r for r in results if r.verdict == "miss" and not math.isnan(r.expected_t)]
    extras = [r for r in results if math.isnan(r.expected_t)]
    hit_offsets = [r.offset_ms for r in hits if r.offset_ms is not None]

    if hit_offsets:
        mean = sum(hit_offsets) / len(hit_offsets)
        mae = sum(abs(x) for x in hit_offsets) / len(hit_offsets)
        worst = max(hit_offsets, key=lambda v: abs(v))
    else:
        mean = 0.0
        mae = 0.0
        worst = 0.0

    judged = len([r for r in results if not math.isnan(r.expected_t)])
    completion = (judged / expected_count) if expected_count > 0 else 0.0

    return {
        "hits": float(len(hits)),
        "misses": float(len(misses)),
        "extras": float(len(extras)),
        "mean": float(mean),
        "mae": float(mae),
        "worst": float(worst),
        "completion": float(completion),
    }


class Button:
    def __init__(self, rect, label: str):
        self.rect = rect
        self.label = label

    def draw(self, screen, font, *, hovered: bool):
        import pygame
        bg = (55, 55, 70) if hovered else (40, 40, 55)
        pygame.draw.rect(screen, bg, self.rect, border_radius=10)
        pygame.draw.rect(screen, (200, 200, 210), self.rect, 2, border_radius=10)
        txt = font.render(self.label, True, (240, 240, 240))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, mx: int, my: int) -> bool:
        return self.rect.collidepoint(mx, my)


def list_maps(scan_dir: Path) -> List[Path]:
    if not scan_dir.exists():
        return []
    return sorted(scan_dir.glob("*.gdr"), key=lambda p: p.name.lower())


def preview_map_info(path: Path, btn: int, p2: bool) -> Tuple[int, float, float]:
    gdr = load_gdr(str(path))
    fps = infer_fps(gdr)
    expected = build_expected_events(gdr, fps=fps, btn=btn, second_player=p2)
    duration = float(gdr.get("duration") or 0.0)
    if duration <= 0.0 and expected:
        duration = expected[-1].t
    return len(expected), fps, duration


def run_app(
    scan_dir: Path,
    btn: int,
    second_player: bool,
    fps_override: float,
    hit_window_ms: float,
    lead_in_s: float,
    width: int,
    height: int,
    scroll_s: float,
    lane_y_frac: float,
    target_frac: float,
    hold_min_frames: int,
) -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption("GD Rhythm Trainer")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 18)
    big = pygame.font.SysFont("Consolas", 30, bold=True)
    small = pygame.font.SysFont("Consolas", 16)

    state = "home"
    selected: Optional[Path] = None
    maps: List[Path] = []
    map_infos: Dict[str, Tuple[int, float, float]] = {}
    scroll_offset = 0

    row_h = 64
    list_top = 110
    list_left = 30
    list_w = width - 60
    list_h = height - 190
    visible_rows = max(1, list_h // row_h)

    scrollbar_width = 12
    scrollbar_margin = 6
    scroll_dragging = False

    expected: List[ExpectedEvent] = []
    notes: List[VisualNote] = []
    results: List[ResultRow] = []
    run_meta: Dict[str, str] = {}

    play_start_perf = 0.0
    win_s = hit_window_ms / 1000.0
    next_idx = 0

    lane_y = int(height * lane_y_frac)
    target_x = int(width * target_frac)

    toast_text = ""
    toast_until = 0.0

    def toast(msg: str, seconds: float = 2.0) -> None:
        nonlocal toast_text, toast_until
        toast_text = msg
        toast_until = time.perf_counter() + seconds

    def refresh_maps() -> None:
        nonlocal maps, selected, scroll_offset, map_infos
        maps = list_maps(scan_dir)
        scroll_offset = 0
        selected = None if (selected is None or selected not in maps) else selected
        map_infos = {}

    def home_buttons():
        y = height - 60
        return (
            Button(pygame.Rect(30, y, 150, 40), "Refresh"),
            Button(pygame.Rect(width - 180, y, 150, 40), "Quit"),
        )

    def results_buttons():
        y = height - 70
        return (
            Button(pygame.Rect(30, y, 190, 46), "Back to Home"),
            Button(pygame.Rect(width - 240, y, 210, 46), "Export Results"),
        )

    def now_t() -> float:
        return time.perf_counter() - play_start_perf

    def record_input(kind: str, actual_t: float) -> None:
        nonlocal next_idx
        while next_idx < len(expected) and actual_t > expected[next_idx].t + win_s:
            ev = expected[next_idx]
            results.append(ResultRow(idx=ev.idx, kind=ev.kind, expected_t=ev.t, expected_frame=ev.frame, actual_t=None, offset_ms=None, verdict="miss"))
            next_idx += 1

        if next_idx >= len(expected):
            results.append(ResultRow(idx=len(results), kind=kind, expected_t=float("nan"), expected_frame=-1, actual_t=actual_t, offset_ms=None, verdict="miss"))
            return

        ev = expected[next_idx]
        if ev.kind != kind:
            results.append(ResultRow(idx=len(results), kind=kind, expected_t=float("nan"), expected_frame=-1, actual_t=actual_t, offset_ms=None, verdict="miss"))
            return

        dt = actual_t - ev.t
        if abs(dt) <= win_s:
            results.append(ResultRow(idx=ev.idx, kind=ev.kind, expected_t=ev.t, expected_frame=ev.frame, actual_t=actual_t, offset_ms=dt * 1000.0, verdict="hit"))
        else:
            results.append(ResultRow(idx=ev.idx, kind=ev.kind, expected_t=ev.t, expected_frame=ev.frame, actual_t=actual_t, offset_ms=dt * 1000.0, verdict="miss"))
        next_idx += 1

    def finalize_misses(up_to_t: float) -> None:
        nonlocal next_idx
        while next_idx < len(expected) and up_to_t > expected[next_idx].t + win_s:
            ev = expected[next_idx]
            results.append(ResultRow(idx=ev.idx, kind=ev.kind, expected_t=ev.t, expected_frame=ev.frame, actual_t=None, offset_ms=None, verdict="miss"))
            next_idx += 1

    margin = 20

    def time_to_x(current_t: float, event_t: float) -> int:
        return target_x + int((event_t - current_t) / scroll_s * (width - target_x - margin))

    def go_results() -> None:
        nonlocal state
        state = "results"

    def start_map(path: Path) -> None:
        nonlocal selected, expected, notes, results, run_meta, next_idx, play_start_perf, state, win_s, lane_y, target_x
        selected = path
        try:
            gdr = load_gdr(str(path))
        except Exception:
            toast(f"Failed to load: {path.name}")
            return

        fps = fps_override if fps_override > 0 else infer_fps(gdr)
        expected = build_expected_events(gdr, fps=fps, btn=btn, second_player=second_player)
        if not expected:
            toast("No inputs found (btn/player filter)")
            return

        notes = build_visual_notes(expected, hold_min_frames=hold_min_frames)
        results = []
        next_idx = 0

        lane_y = int(height * lane_y_frac)
        target_x = int(width * target_frac)
        win_s = hit_window_ms / 1000.0

        run_meta = {
            "macro_file": str(path),
            "fps": f"{fps:.6f}",
            "window_ms": f"{hit_window_ms:.2f}",
            "expected_events": str(len(expected)),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_x_frac": str(target_frac),
        }

        play_start_perf = time.perf_counter() + max(0.0, lead_in_s)
        state = "lead_in"

    def export_results() -> Optional[Path]:
        if selected is None:
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).resolve().parent
        else:
            base_dir = Path(__file__).resolve().parent

        results_dir = base_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        out_path = results_dir / f"{selected.stem}_results_{timestamp}.txt"
        run_meta["inputs_captured"] = str(len([r for r in results if r.actual_t is not None]))
        export_results_text(results, out_path, run_meta)
        return out_path

    def get_scrollbar_geometry():
        import pygame

        nonlocal scroll_offset

        if len(maps) <= visible_rows:
            return None, None, 0, 0, 0

        list_rect = pygame.Rect(list_left, list_top, list_w, list_h)
        track_margin = scrollbar_margin
        track_w = scrollbar_width
        track_x = list_rect.right - track_w - track_margin
        track_y = list_rect.top + track_margin
        track_h = list_rect.height - 2 * track_margin
        track_rect = pygame.Rect(track_x, track_y, track_w, track_h)

        max_scroll = max(1, len(maps) - visible_rows)
        thumb_h = max(24, int(track_h * (visible_rows / len(maps))))
        thumb_range = max(1, track_h - thumb_h)
        thumb_y = track_y + int(thumb_range * (scroll_offset / max_scroll))
        thumb_rect = pygame.Rect(track_x + 2, thumb_y, track_w - 4, thumb_h)
        return track_rect, thumb_rect, track_h, thumb_h, max_scroll

    refresh_maps()

    running = True
    while running:
        clock.tick(240)

        mx, my = pygame.mouse.get_pos()
        clicked = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if state in ("play", "lead_in"):
                    go_results()
            elif event.type == pygame.MOUSEWHEEL and state == "home":
                if len(maps) > visible_rows:
                    max_offset = max(0, len(maps) - visible_rows)
                    scroll_offset = max(0, min(max_offset, scroll_offset - event.y))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked = True
                    if state == "home":
                        track_rect, thumb_rect, track_h, thumb_h, max_scroll = get_scrollbar_geometry()
                        if thumb_rect is not None and thumb_rect.collidepoint(event.pos):
                            scroll_dragging = True
                elif state == "home" and event.button in (4, 5):
                    if len(maps) > visible_rows:
                        max_offset = max(0, len(maps) - visible_rows)
                        delta = -1 if event.button == 4 else 1
                        scroll_offset = max(0, min(max_offset, scroll_offset + delta))
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    scroll_dragging = False
            elif event.type == pygame.MOUSEMOTION and scroll_dragging and state == "home":
                track_rect, thumb_rect, track_h, thumb_h, max_scroll = get_scrollbar_geometry()
                if track_rect is not None and max_scroll > 0:
                    track_y = track_rect.y
                    track_h = track_rect.height
                    thumb_range = max(1, track_h - thumb_h)
                    mouse_y = event.pos[1]
                    center_min = track_y + thumb_h / 2
                    center_max = track_y + track_h - thumb_h / 2
                    center_y = min(max(mouse_y, center_min), center_max)
                    normalized = (center_y - center_min) / thumb_range
                    new_offset = int(round(normalized * max_scroll))
                    max_offset = max(0, len(maps) - visible_rows)
                    scroll_offset = max(0, min(max_offset, new_offset))
            elif state == "play":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    record_input("down", now_t())
                elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                    record_input("up", now_t())

        screen.fill((18, 18, 24))

        if state == "home":
            screen.blit(big.render("GD Rhythm Trainer", True, (240, 240, 240)), (30, 24))
            screen.blit(small.render(f"Maps folder: {str(scan_dir)}", True, (185, 185, 185)), (30, 56))
            screen.blit(small.render("Click a map to play. SPACE: press/release. ESC: quit.", True, (185, 185, 185)), (30, 76))


            list_rect = pygame.Rect(list_left, list_top, list_w, list_h)
            pygame.draw.rect(screen, (28, 28, 36), list_rect, border_radius=12)
            pygame.draw.rect(screen, (95, 95, 110), list_rect, 2, border_radius=12)

            if not maps:
                screen.blit(font.render("No .gdr files found in this folder", True, (210, 210, 210)), (46, 120))
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    scroll_offset = max(0, scroll_offset - 1)
                if keys[pygame.K_DOWN]:
                    scroll_offset = min(max(0, len(maps) - visible_rows), scroll_offset + 1)

                start = scroll_offset
                end = min(len(maps), start + visible_rows)

                row_area_w = list_w - 24 - (scrollbar_width + scrollbar_margin)

                for i in range(start, end):
                    p = maps[i]
                    row_y = list_top + 6 + (i - start) * row_h
                    row = pygame.Rect(list_left + 12, row_y, row_area_w, row_h - 6)
                    hovered = row.collidepoint(mx, my)
                    bg = (55, 55, 70) if hovered else (40, 40, 52)
                    pygame.draw.rect(screen, bg, row, border_radius=10)
                    pygame.draw.rect(screen, (110, 110, 128), row, 1, border_radius=10)

                    screen.blit(font.render(p.name, True, (240, 240, 240)), (row.x + 12, row.y + 10))

                    if p.name not in map_infos:
                        try:
                            map_infos[p.name] = preview_map_info(p, btn=btn, p2=second_player)
                        except Exception:
                            map_infos[p.name] = (0, 0.0, 0.0)

                    evn, fp, dur = map_infos[p.name]
                    screen.blit(small.render(f"{evn} events | {fp:.0f} fps | {dur:.1f}s", True, (190, 190, 190)), (row.x + 12, row.y + 32))

                    if clicked and hovered:
                        start_map(p)

                if len(maps) > visible_rows:
                    track_rect, thumb_rect, track_h, thumb_h, max_scroll = get_scrollbar_geometry()
                    if track_rect is not None and thumb_rect is not None:
                        pygame.draw.rect(screen, (20, 20, 30), track_rect, border_radius=6)
                        pygame.draw.rect(screen, (80, 80, 100), track_rect, 1, border_radius=6)
                        pygame.draw.rect(screen, (140, 140, 170), thumb_rect, border_radius=6)

            b_refresh = Button(pygame.Rect(30, height - 60, 150, 40), "Refresh")
            b_quit = Button(pygame.Rect(width - 180, height - 60, 150, 40), "Quit")
            for b in (b_refresh, b_quit):
                b.draw(screen, font, hovered=b.hit(mx, my))
            if clicked and b_refresh.hit(mx, my):
                refresh_maps()
                toast("Refreshed map list")
            if clicked and b_quit.hit(mx, my):
                running = False


        elif state == "lead_in":
            left = play_start_perf - time.perf_counter()
            screen.blit(big.render(f"Starting in {max(0.0, left):.2f}s", True, (240, 240, 240)), (30, 30))
            screen.blit(font.render("SPACE: press/release. ESC: quit.", True, (190, 190, 190)), (30, 72))
            if selected is not None:
                screen.blit(small.render(f"Map: {selected.name}", True, (175, 175, 175)), (30, 100))
            if time.perf_counter() >= play_start_perf:
                state = "play"

        elif state == "play":
            t = now_t()
            finalize_misses(t)

            pygame.draw.line(screen, (120, 120, 140), (0, lane_y), (width, lane_y), 1)
            pygame.draw.line(screen, (240, 240, 240), (target_x, lane_y - 90), (target_x, lane_y + 90), 2)
            target_box = pygame.Rect(0, 0, 52, 52)
            target_box.center = (target_x, lane_y)
            pygame.draw.rect(screen, (240, 240, 240), target_box, 2)

            view_left = t
            view_right = t + scroll_s + 0.1

            for n in notes:
                if n.end_t < view_left - 0.2:
                    continue
                if n.start_t > view_right + 0.2:
                    continue

                x0 = time_to_x(t, n.start_t)
                s = 32
                head = pygame.Rect(0, 0, s, s)
                head.center = (x0, lane_y)

                if n.is_hold:
                    x1 = time_to_x(t, n.end_t)
                    bar_thick = max(6, s // 5)
                    left = x0 + s // 2
                    right = x1 - s // 2
                    if right > left:
                        pygame.draw.line(screen, (110, 220, 160), (left, lane_y), (right, lane_y), bar_thick)

                    pygame.draw.rect(screen, (90, 180, 255), head, border_radius=4)

                    tri_h = s
                    tri = [(x1, lane_y), (x1 - tri_h // 2, lane_y - tri_h // 2), (x1 - tri_h // 2, lane_y + tri_h // 2)]
                    pygame.draw.polygon(screen, (255, 170, 80), tri)
                else:
                    pygame.draw.rect(screen, (90, 180, 255), head, border_radius=4)

            screen.blit(font.render(f"t={max(0.0, t):.3f}s", True, (200, 200, 200)), (10, 10))
            judged = len([r for r in results if not math.isnan(r.expected_t)])
            miss_n = len([r for r in results if r.verdict == "miss" and not math.isnan(r.expected_t)])
            screen.blit(font.render(f"judged={judged}/{len(expected)}   misses={miss_n}", True, (200, 200, 200)), (10, 34))

            if next_idx < len(expected):
                ev = expected[next_idx]
                screen.blit(font.render(f"Next: {ev.kind.upper()} @ {ev.t:.3f}s (f {ev.frame})", True, (200, 200, 200)), (10, 58))

            if next_idx >= len(expected) and t > expected[-1].t + 1.0:
                go_results()

        elif state == "results":
            screen.blit(big.render("Run Results", True, (240, 240, 240)), (30, 24))
            if selected is not None:
                screen.blit(small.render(f"Map: {selected.name}", True, (185, 185, 185)), (30, 56))

            stats = compute_compact_stats(results, expected_count=len(expected) if expected else 0)
            hits = int(stats["hits"])
            misses = int(stats["misses"])
            extras = int(stats["extras"])

            lines = [
                f"Completion: {stats['completion']*100.0:.1f}%",
                f"Hits: {hits}",
                f"Misses: {misses}",
                f"Unexpected inputs: {extras}",
                f"Mean hit offset: {stats['mean']:+.2f} ms (negative=early, positive=late)",
                f"Worst hit offset: {stats['worst']:.2f} ms",
                "",
            ]
            y = 90
            for s in lines:
                screen.blit(font.render(s, True, (205, 205, 205)), (30, y))
                y += 22

            b_home = Button(pygame.Rect(30, height - 70, 190, 46), "Back to Home")
            b_export = Button(pygame.Rect(width - 240, height - 70, 210, 46), "Export Results")
            play_again_w = 210
            play_again_h = 46
            play_again_x = (width - play_again_w) // 2
            play_again_y = height - 70
            b_play_again = Button(pygame.Rect(play_again_x, play_again_y, play_again_w, play_again_h), "Play Again")

            b_home.draw(screen, font, hovered=b_home.hit(mx, my))
            b_export.draw(screen, font, hovered=b_export.hit(mx, my))
            b_play_again.draw(screen, font, hovered=b_play_again.hit(mx, my))

            if clicked and b_home.hit(mx, my):
                refresh_maps()
                state = "home"
            if clicked and b_export.hit(mx, my):
                out = export_results()
                toast(f"Exported: {out.name}" if out else "No map selected", seconds=2.3)
            if clicked and b_play_again.hit(mx, my):
                if selected is not None:
                    start_map(selected)

        if toast_text and time.perf_counter() < toast_until:
            if state == "home" and toast_text == "Refreshed map list":
                msg_surf = small.render(toast_text, True, (220, 220, 220))
                box_w = msg_surf.get_width() + 32
                box_h = 40
                box = pygame.Rect(0, 0, box_w, box_h)
                box.center = (width // 2, height - 40)
                pygame.draw.rect(screen, (30, 30, 40), box, border_radius=10)
                pygame.draw.rect(screen, (120, 120, 140), box, 1, border_radius=10)
                screen.blit(msg_surf, (box.x + (box_w - msg_surf.get_width()) // 2, box.y + 10))
            else:
                box = pygame.Rect(0, 0, width - 60, 36)
                box.center = (width // 2, height - 120)
                pygame.draw.rect(screen, (30, 30, 40), box, border_radius=10)
                pygame.draw.rect(screen, (120, 120, 140), box, 1, border_radius=10)
                screen.blit(small.render(toast_text, True, (220, 220, 220)), (box.x + 12, box.y + 9))

        pygame.display.flip()

    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(prog="gd_rhythm_trainer", description="Precision-focused 1D GD macro click trainer (.gdr)")
    ap.add_argument("--dir", type=str, default="", help="Directory to scan for .gdr maps (default: directory containing this .py)")
    ap.add_argument("--btn", type=int, default=1, help="Button id to use (default: 1)")
    ap.add_argument("--p2", action="store_true", help="Use 2p inputs instead of 1p")
    ap.add_argument("--fps", type=float, default=0.0, help="Override FPS (auto-infer if 0)")
    ap.add_argument("--window-ms", type=float, default=18.0, help="Hit window in milliseconds (±)")
    ap.add_argument("--lead-in", type=float, default=1.5, help="Countdown seconds before play starts")
    ap.add_argument("--scroll-s", type=float, default=2.5, help="Seconds visible ahead on timeline")
    ap.add_argument("--hold-min-frames", type=int, default=3, help="Minimum DOWN->UP duration (frames) to draw as a hold note")
    ap.add_argument("--lane-y", type=float, default=0.55, help="Vertical lane position as fraction of height")
    ap.add_argument("--target-frac", type=float, default=1.0 / 3.0, help="Target position as fraction of screen width")
    ap.add_argument("--width", type=int, default=980)
    ap.add_argument("--height", type=int, default=520)
    args = ap.parse_args()

    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    scan_dir = Path(args.dir).expanduser().resolve() if args.dir.strip() else (base_dir / "maps")

    run_app(
        scan_dir=scan_dir,
        btn=args.btn,
        second_player=bool(args.p2),
        fps_override=args.fps,
        hit_window_ms=args.window_ms,
        lead_in_s=args.lead_in,
        width=args.width,
        height=args.height,
        scroll_s=args.scroll_s,
        lane_y_frac=args.lane_y,
        target_frac=args.target_frac,
        hold_min_frames=args.hold_min_frames,
    )


if __name__ == "__main__":
    main()

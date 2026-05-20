# ================================================================
#   CodeAlpha AI Internship — Object Detection & Tracking
#   Author  : [Your Name]
#   Model   : YOLOv8s — Open Images V7 (600+ Classes)
#   Tech    : Python | OpenCV | YOLOv8 (Ultralytics)
# ================================================================

import cv2
import time
import datetime
import collections
import numpy as np
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these values if needed)
# ────────────────────────────────────────────────────────────────
MODEL_PATH        = "yolov8s-oiv7.pt"   # 600+ classes (auto-downloads ~22 MB)
CONFIDENCE_THRESH = 0.20                # lower = detects more objects
WEBCAM_INDEX      = 0                   # change to 1 if webcam not found
INPUT_WIDTH       = 1280                # higher res = detects small objects
INPUT_HEIGHT      = 720
TRAIL_LENGTH      = 20                  # how many trail points per object
MIN_BOX_SIZE      = 20                  # ignore tiny noise boxes (pixels)


# ────────────────────────────────────────────────────────────────
#  HELPER — unique BGR colour for every class (based on class ID)
# ────────────────────────────────────────────────────────────────
def get_class_color(class_id):
    np.random.seed(class_id * 37 + 13)
    hue = int(np.random.randint(0, 180))
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ────────────────────────────────────────────────────────────────
#  HELPER — draw filled label above a bounding box
# ────────────────────────────────────────────────────────────────
def draw_label(frame, text, x, y, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    fscale     = 0.48 if len(text) > 18 else 0.55
    thickness  = 1
    pad        = 4
    (tw, th), bl = cv2.getTextSize(text, font, fscale, thickness)
    rx1, ry1   = x, y - th - bl - pad * 2
    rx2, ry2   = x + tw + pad * 2, y
    # clamp to frame edges
    if ry1 < 0:
        ry1, ry2 = 0, th + bl + pad * 2
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)
    cv2.putText(frame, text, (rx1 + pad, ry2 - bl - 1),
                font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────
#  HELPER — draw fading movement trail
# ────────────────────────────────────────────────────────────────
def draw_trail(frame, trail, color):
    for i in range(1, len(trail)):
        alpha     = i / len(trail)
        faded     = tuple(int(c * alpha) for c in color)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, trail[i - 1], trail[i], faded, thickness)


# ────────────────────────────────────────────────────────────────
#  HELPER — object counter panel (top-right corner)
# ────────────────────────────────────────────────────────────────
def draw_counter_panel(frame, counts):
    if not counts:
        return
    font     = cv2.FONT_HERSHEY_SIMPLEX
    line_h   = 22
    pad      = 8
    panel_w  = 215
    sorted_c = sorted(counts.items(), key=lambda x: -x[1])
    panel_h  = pad * 2 + line_h + len(sorted_c) * line_h
    h, w     = frame.shape[:2]
    px, py   = w - panel_w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Detected Objects",
                (px + pad, py + pad + 13),
                font, 0.52, (0, 215, 255), 1, cv2.LINE_AA)

    for i, (cls, cnt) in enumerate(sorted_c):
        yp = py + pad + line_h + i * line_h + 13
        cv2.putText(frame, f"  {cls[:22]:<22}: {cnt}",
                    (px + pad, yp), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────
#  HELPER — bottom stats bar
# ────────────────────────────────────────────────────────────────
def draw_stats_bar(frame, total, conf):
    h, w  = frame.shape[:2]
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Objects: {total}",
                (10, h - 8), font, 0.48, (0, 255, 120), 1, cv2.LINE_AA)

    title = "CodeAlpha | Object Detection & Tracking"
    (tw, _), _ = cv2.getTextSize(title, font, 0.46, 1)
    cv2.putText(frame, title, (w // 2 - tw // 2, h - 8),
                font, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    right = f"OIV7-600+  Conf:{conf:.0%}"
    (rw, _), _ = cv2.getTextSize(right, font, 0.46, 1)
    cv2.putText(frame, right, (w - rw - 10, h - 8),
                font, 0.46, (100, 200, 255), 1, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────
#  HELPER — FPS text + mini bar graph
# ────────────────────────────────────────────────────────────────
def draw_fps(frame, fps, fps_history):
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 215, 255), 2, cv2.LINE_AA)

    if len(fps_history) < 2:
        return
    gx, gy, gw, gh = 10, 40, 150, 35
    overlay = frame.copy()
    cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    max_fps = max(fps_history) or 1
    bar_w   = gw / len(fps_history)
    for i, f in enumerate(fps_history):
        bh = int((f / max_fps) * (gh - 4))
        bx = int(gx + i * bar_w)
        cv2.rectangle(frame,
                      (bx, gy + gh - bh),
                      (bx + max(1, int(bar_w) - 1), gy + gh),
                      (0, 200, 80), -1)


# ────────────────────────────────────────────────────────────────
#  HELPER — help overlay (press H)
# ────────────────────────────────────────────────────────────────
def draw_help_overlay(frame):
    h, w   = frame.shape[:2]
    keys   = [
        ("Q", "Quit application"),
        ("S", "Save screenshot"),
        ("P", "Pause / Resume"),
        ("+", "Increase confidence"),
        ("-", "Decrease confidence"),
        ("H", "Toggle this help"),
    ]
    pw, ph = 320, 200
    px, py = w // 2 - pw // 2, h // 2 - ph // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "KEYBOARD SHORTCUTS",
                (px + 60, py + 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (0, 215, 255), 1, cv2.LINE_AA)
    for i, (k, desc) in enumerate(keys):
        cv2.putText(frame, f"  [{k}]  {desc}",
                    (px + 15, py + 56 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────
#  HELPER — save screenshot with flash effect
# ────────────────────────────────────────────────────────────────
def save_screenshot(frame, flash):
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{ts}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Screenshot saved → {filename}")
    flash[0] = 6


# ────────────────────────────────────────────────────────────────
#  STARTUP BANNER
# ────────────────────────────────────────────────────────────────
def print_banner():
    print("\n" + "═" * 48)
    print("  CodeAlpha AI Internship")
    print("  Object Detection & Tracking")
    print("  Model  : YOLOv8s-OIV7  (600+ Classes)")
    print("  Hotkeys: Press H in the webcam window")
    print("═" * 48 + "\n")


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
def main():
    print_banner()

    # Load model
    print("[INFO] Loading model… (downloads ~22 MB on first run)")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded! Total classes: {len(model.names)}\n")

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Webcam not found!")
        print("  → Make sure webcam is plugged in.")
        print("  → Try changing WEBCAM_INDEX = 1 at the top of main.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  INPUT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)
    print("[INFO] Webcam ready!  Press H for help.  Press Q to quit.\n")

    # State
    conf_thresh = CONFIDENCE_THRESH
    paused      = False
    show_help   = False
    prev_time   = time.time()
    fps         = 0.0
    fps_history = collections.deque(maxlen=30)
    trails      = collections.defaultdict(lambda: collections.deque(maxlen=TRAIL_LENGTH))
    flash       = [0]
    last_frame  = None

    while True:

        # ── Key events ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit pressed.")
            break
        elif key == ord("p"):
            paused = not paused
            print("[INFO]", "PAUSED" if paused else "RESUMED")
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("s") and last_frame is not None:
            save_screenshot(last_frame, flash)
        elif key in (ord("+"), ord("=")):
            conf_thresh = min(0.90, round(conf_thresh + 0.05, 2))
            print(f"[INFO] Confidence → {conf_thresh:.0%}")
        elif key == ord("-"):
            conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
            print(f"[INFO] Confidence → {conf_thresh:.0%}")

        # ── Paused screen ─────────────────────────────────────
        if paused and last_frame is not None:
            display = last_frame.copy()
            h, w    = display.shape[:2]
            cv2.putText(display, "PAUSED",
                        (w // 2 - 90, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 0, 220), 5, cv2.LINE_AA)
            if show_help:
                draw_help_overlay(display)
            cv2.imshow("CodeAlpha - Object Detection & Tracking", display)
            continue

        # ── Read frame ────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Could not read frame, retrying…")
            continue

        orig_h, orig_w = frame.shape[:2]

        # ── Resize for inference ──────────────────────────────
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        scale_x = orig_w / INPUT_WIDTH
        scale_y = orig_h / INPUT_HEIGHT

        # ── Run detection + tracking ──────────────────────────
        results = model.track(
            resized,
            persist = True,
            conf    = conf_thresh,
            verbose = False
        )

        # ── Draw detections ───────────────────────────────────
        class_counts = {}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # scale coords back to original frame size
                rx1, ry1, rx2, ry2 = box.xyxy[0]
                x1 = int(rx1 * scale_x); y1 = int(ry1 * scale_y)
                x2 = int(rx2 * scale_x); y2 = int(ry2 * scale_y)

                # skip tiny/noise boxes
                if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                    continue

                conf_val   = float(box.conf[0])
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                track_id   = int(box.id[0]) if box.id is not None else -1
                color      = get_class_color(class_id)

                # count
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # trail
                if track_id != -1:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trails[track_id].append((cx, cy))
                    draw_trail(frame, list(trails[track_id]), color)

                # box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # label  →  "Person  87%  #3"
                id_str = f"  #{track_id}" if track_id != -1 else ""
                label  = f"{class_name}  {conf_val:.0%}{id_str}"
                draw_label(frame, label, x1, y1, color)

        # ── FPS ───────────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        fps_history.append(fps)

        # ── UI panels ─────────────────────────────────────────
        draw_fps(frame, fps, list(fps_history))
        draw_counter_panel(frame, class_counts)
        draw_stats_bar(frame, sum(class_counts.values()), conf_thresh)

        if show_help:
            draw_help_overlay(frame)

        # ── Screenshot flash ──────────────────────────────────
        if flash[0] > 0:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (orig_w, orig_h), (255, 255, 255), -1)
            cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
            flash[0] -= 1

        last_frame = frame.copy()
        cv2.imshow("CodeAlpha - Object Detection & Tracking", frame)

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Goodbye!")


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
from datetime import datetime


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left (smallest x+y)
    rect[2] = pts[np.argmax(s)]  # bottom-right (largest x+y)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right (smallest x-y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest x-y)

    return rect


def find_surface(frame, min_area_ratio=0.005):
    """
    Try to find the brightest, reasonably large planar region in the frame.
    This is called every frame in dynamic mode, or once in frozen mode.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If the bright area is the background instead of the paper,
    # invert so the paper-like region is white on black.
    if np.mean(thresh) > 127:
        thresh = 255 - thresh

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    h, w = gray.shape[:2]
    min_area = min_area_ratio * w * h

    # Pick largest bright blob
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    # Try to approximate as a 4-corner polygon
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        pts = approx.reshape(4, 2).astype("float32")
        return order_points(pts)
    else:
        # Fallback: axis-aligned bounding box of that bright region
        x, y, bw, bh = cv2.boundingRect(cnt)
        pts = np.array([
            [x,         y        ],
            [x + bw,    y        ],
            [x + bw,    y + bh   ],
            [x,         y + bh   ]
        ], dtype="float32")
        return order_points(pts)


def classify_crayola_color(h, s, v):
    """
    Map HSV values to an approximate Crayola-like color.
    Uses mostly hue; assumes input pixels are already filtered
    to be reasonably saturated & bright.
    """

    # Very low saturation but bright-ish -> gray
    if s < 40 and v > 80:
        return "gray", (128, 128, 128)

    # Pretty dark overall -> treat as brown-ish
    if v < 60:
        return "brown", (42, 42, 165)

    # Red region (wrap-around)
    if h < 10 or h >= 170:
        return "red", (0, 0, 255)

    # Orange / brown region
    if 10 <= h < 22:
        return "orange", (0, 165, 255)

    # Yellow
    if 22 <= h < 35:
        return "yellow", (0, 255, 255)

    # Yellow-green / lime
    if 35 <= h < 50:
        return "yellow-green", (50, 255, 50)

    # Green
    if 50 <= h < 85:
        return "green", (0, 200, 0)

    # Blue-green / teal
    if 85 <= h < 100:
        return "blue-green", (200, 200, 0)

    # Blue
    if 100 <= h < 130:
        return "blue", (255, 0, 0)

    # Purple / violet
    if 130 <= h < 155:
        return "purple", (255, 0, 255)

    # Magenta-ish
    if 155 <= h < 170:
        return "magenta", (255, 0, 200)

    # Fallback
    return "red", (0, 0, 255)


def find_marker_center(frame, board_mask=None):
    """
    Detect marker tip INSIDE the board region in the original camera frame.
    We ONLY look for colorful (high-sat, not-dark) pixels (no black detection).
    Returns: center, color_bgr, color_name
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    if board_mask is None:
        board_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Estimate background brightness from low-saturation pixels inside board
    paper_mask = (board_mask > 0) & (s_channel < 30)
    paper_vals = v_channel[paper_mask]
    if paper_vals.size > 0:
        paper_v = float(np.mean(paper_vals))
    else:
        paper_v = 180.0

    # Colorful marker mask: high-ish sat, not super dark
    v_color_thresh = int(max(60, 0.4 * paper_v))
    lower_color = np.array([0, 90, v_color_thresh])   # stricter saturation/value
    upper_color = np.array([179, 255, 255])
    mask_color = cv2.inRange(hsv, lower_color, upper_color)

    # Restrict to board
    mask_all = cv2.bitwise_and(mask_color, board_mask)

    # Clean up
    mask_all = cv2.GaussianBlur(mask_all, (7, 7), 0)
    kernel = np.ones((7, 7), np.uint8)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None, None, None

    # Center from min enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))

    # Centroid for sampling color
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = center

    cx = max(0, min(w_img - 1, cx))
    cy = max(0, min(h_img - 1, cy))

    h_val, s_val, v_val = hsv[cy, cx]
    color_name, color_bgr = classify_crayola_color(
        int(h_val), int(s_val), int(v_val)
    )

    return center, color_bgr, color_name


def save_screenshots(camera_img, canvas_img):
    """
    Save BOTH the camera panel (with overlay) and the canvas panel
    into ./freeform_screenshots with timestamped filenames.
    """
    os.makedirs("freeform_screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cam_filename = f"freeform_screenshots/camera_{timestamp}.png"
    canvas_filename = f"freeform_screenshots/canvas_{timestamp}.png"

    cv2.imwrite(cam_filename, camera_img)
    cv2.imwrite(canvas_filename, canvas_img)

    print(f"[screenshot] Saved camera view to {cam_filename}")
    print(f"[screenshot] Saved canvas view to {canvas_filename}")


def main():
    CANVAS_W = 800
    CANVAS_H = 600

    # top-down drawing canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    prev_canvas_pt = None
    smoothed_canvas_pt = None  # for exponential smoothing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # pen pause toggle
    pen_paused = False
    # surface mode toggle: True = dynamic (update every frame), False = frozen
    dynamic_surface = True

    surface_pts = None
    using_fallback = False
    last_color_name = "none"

    print("Controls:")
    print("  q / ESC : quit")
    print("  c       : clear canvas")
    print("  s       : save screenshots (camera + canvas) to ./iwb_screenshots")
    print("  SPACE   : toggle pen pause (no drawing when PAUSED)")
    print("  m       : toggle surface mode (DYNAMIC <-> FROZEN)")
    print("  r       : re-detect surface (in FROZEN mode sets surface to None)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Surface detection:
        # - In dynamic mode: update every frame
        # - In frozen mode: only detect if we don't have surface_pts yet
        if dynamic_surface or surface_pts is None:
            detected_pts = find_surface(frame, min_area_ratio=0.02)

            if detected_pts is None:
                surface_pts = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]
                ], dtype="float32")
                using_fallback = True
            else:
                surface_pts = detected_pts
                using_fallback = False

        # Homographies
        dst_pts = np.array([
            [0, 0],
            [CANVAS_W - 1, 0],
            [CANVAS_W - 1, CANVAS_H - 1],
            [0, CANVAS_H - 1]
        ], dtype="float32")

        H, _ = cv2.findHomography(surface_pts, dst_pts)
        H_inv, _ = cv2.findHomography(dst_pts, surface_pts)

        # Board mask for marker detection
        board_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(board_mask, [surface_pts.astype(np.int32)], 255)

        # Marker detection (with Crayola colors!)
        marker_center, marker_color, color_name = find_marker_center(
            frame, board_mask=board_mask
        )

        if marker_center is None:
            prev_canvas_pt = None
            smoothed_canvas_pt = None
            color_name_display = last_color_name
        else:
            last_color_name = color_name
            color_name_display = color_name

            # Draw on canvas ONLY if not paused
            if H is not None and not pen_paused:
                pts = np.array([[marker_center]], dtype="float32")
                projected = cv2.perspectiveTransform(pts, H)[0][0]
                u_raw, v_raw = projected[0], projected[1]

                if 0 <= u_raw < CANVAS_W and 0 <= v_raw < CANVAS_H:
                    alpha = 0.3  # smoothing factor

                    if smoothed_canvas_pt is None:
                        sm_u, sm_v = u_raw, v_raw
                    else:
                        prev_u, prev_v = smoothed_canvas_pt
                        sm_u = (1 - alpha) * prev_u + alpha * u_raw
                        sm_v = (1 - alpha) * prev_v + alpha * v_raw

                    u, v = int(sm_u), int(sm_v)
                    smoothed_canvas_pt = (sm_u, sm_v)

                    if prev_canvas_pt is not None:
                        cv2.line(canvas, prev_canvas_pt, (u, v),
                                 marker_color, thickness=4)
                    prev_canvas_pt = (u, v)
                else:
                    prev_canvas_pt = None
                    smoothed_canvas_pt = None
            else:
                prev_canvas_pt = None
                smoothed_canvas_pt = None

        # Warp the canvas back onto the camera view
        if H_inv is not None:
            overlay = cv2.warpPerspective(canvas, H_inv, (w, h))

            overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, mask_overlay = cv2.threshold(overlay_gray, 10, 255,
                                            cv2.THRESH_BINARY)
            mask_overlay_inv = cv2.bitwise_not(mask_overlay)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_overlay_inv)
            drawing_fg = cv2.bitwise_and(overlay, overlay, mask=mask_overlay)

            frame_with_drawing = cv2.add(frame_bg, drawing_fg)
        else:
            frame_with_drawing = frame.copy()

        # Outline the surface
        outline_color = (0, 255, 0) if not using_fallback else (0, 165, 255)
        cv2.polylines(
            frame_with_drawing,
            [surface_pts.astype(np.int32)],
            isClosed=True,
            color=outline_color,
            thickness=2
        )

        # Show marker dot
        if marker_center is not None and marker_color is not None:
            cv2.circle(frame_with_drawing, marker_center, 7, marker_color, -1)

        # Status text: mode + color + surface mode
        mode_text = "PAUSED" if pen_paused else "DRAWING"
        surface_mode_text = "DYNAMIC" if dynamic_surface else "FROZEN"

        cv2.putText(
            frame_with_drawing,
            f"Mode: {mode_text} | Color: {color_name_display} | Surface: {surface_mode_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Combined display
        disp_h, disp_w = 400, 600
        cam_disp = cv2.resize(frame_with_drawing, (disp_w, disp_h))
        canvas_disp = cv2.resize(canvas, (disp_w, disp_h))

        combined = np.hstack((cam_disp, canvas_disp))
        cv2.imshow("Interactive Whiteboard (Camera | Canvas)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            canvas[:] = 0
            prev_canvas_pt = None
            smoothed_canvas_pt = None
        elif key == ord('s'):
            # Save ORIGINAL camera view with overlay and the raw canvas
            save_screenshots(frame_with_drawing, canvas)
        elif key == 32:  # SPACE
            pen_paused = not pen_paused
            prev_canvas_pt = None
            smoothed_canvas_pt = None
            state = "PAUSED" if pen_paused else "DRAWING"
            print(f"[mode] Pen mode switched to: {state}")
        elif key == ord('m'):
            dynamic_surface = not dynamic_surface
            mode = "DYNAMIC" if dynamic_surface else "FROZEN"
            print(f"[surface] Surface mode switched to: {mode}")
        elif key == ord('r'):
            # Force re-detect surface next frame (only really meaningful in FROZEN mode)
            surface_pts = None
            print("[surface] Surface will be re-detected.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If the bright area is the background instead of the paper,
    # invert so the paper is white on black.
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


def classify_crayola_color(h, s, v, paper_v=None):
    """
    Map HSV values to an approximate Crayola-like color.
    NOTE: we are NOT trying to classify black here; black is excluded by the mask.
    """
    if paper_v is None or paper_v <= 0:
        paper_v = 220.0

    dark_thresh  = 0.60 * paper_v

    # Low saturation but reasonably bright -> gray-ish
    if s < 35 and v > dark_thresh:
        return "gray", (128, 128, 128)

    # Red region (wrap-around)
    if h < 8 or h >= 172:
        return "red", (0, 0, 255)

    # Orange / brown region
    if 8 <= h < 22:
        if v < dark_thresh:
            return "brown", (42, 42, 165)
        else:
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
    if 155 <= h < 172:
        return "magenta", (255, 0, 200)

    # Fallback
    return "red", (0, 0, 255)


def find_marker_center(frame, board_mask=None):
    """
    Detect marker tip INSIDE the board/paper region in the original camera frame.
    We ONLY look for colorful (high-sat, not-dark) pixels (no black detection).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    if board_mask is None:
        board_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Estimate paper brightness from low-saturation pixels inside board
    paper_mask = (board_mask > 0) & (s_channel < 40)
    paper_vals = v_channel[paper_mask]
    if paper_vals.size > 0:
        paper_v = float(np.mean(paper_vals))
    else:
        paper_v = 220.0

    # ---- Colorful marker mask: high-ish sat, not super dark ----
    v_color_thresh = int(max(40, 0.35 * paper_v))
    lower_color = np.array([0, 80, v_color_thresh])
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
        int(h_val), int(s_val), int(v_val), paper_v=paper_v
    )

    return center, color_bgr, color_name


def main():
    CANVAS_W = 800
    CANVAS_H = 600

    # Black background canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    prev_canvas_pt = None
    smoothed_canvas_pt = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    surface_pts = None
    last_color_name = "none"

    print("Controls:")
    print("  q / ESC : quit")
    print("  c       : clear canvas")
    print("  r       : re-detect surface\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Detect surface when we don't have it yet
        if surface_pts is None:
            detected_pts = find_surface(frame, min_area_ratio=0.02)

            if detected_pts is not None:
                surface_pts = detected_pts
            else:
                # Fallback = whole frame (not ideal but prevents crash)
                surface_pts = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]
                ], dtype="float32")

        # Compute homographies
        dst_pts = np.array([
            [0, 0],
            [CANVAS_W - 1, 0],
            [CANVAS_W - 1, CANVAS_H - 1],
            [0, CANVAS_H - 1]
        ], dtype="float32")

        H, _ = cv2.findHomography(surface_pts, dst_pts)
        H_inv, _ = cv2.findHomography(dst_pts, surface_pts)

        # Board mask for detection
        board_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(board_mask, [surface_pts.astype(np.int32)], 255)

        # Detect marker in camera space (only colorful, no black)
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

            if H is not None:
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

        # Warp the canvas back onto the camera paper
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

        # Draw surface outline
        cv2.polylines(
            frame_with_drawing,
            [surface_pts.astype(np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        # Show marker on camera panel
        if marker_center is not None and marker_color is not None:
            cv2.circle(frame_with_drawing, marker_center, 7, marker_color, -1)

        # Show current color
        cv2.putText(
            frame_with_drawing,
            f"Color: {color_name_display}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Left: camera+drawing, Right: canvas
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
        elif key == ord('r'):
            surface_pts = None
            prev_canvas_pt = None
            smoothed_canvas_pt = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

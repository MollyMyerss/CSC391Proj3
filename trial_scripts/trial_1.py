import cv2
import numpy as np


# ----------------- Utility: order corner points ----------------- #

def order_points(pts):
    """
    Given a set of 4 points on a roughly quadrilateral region,
    return them ordered as:
        [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]      # bottom-right (largest x+y)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right (smallest x-y)
    rect[3] = pts[np.argmax(diff)]   # bottom-left (largest x-y)

    return rect


# ----------------- Surface detection (paper / board) ----------------- #

def find_surface(frame, min_area_ratio=0.005):
    """
    Find a planar surface (like a whiteboard / sheet of paper) using
    thresholding + contours.

    Returns:
        4x2 float32 array of ordered corner points, or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
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


# ----------------- Marker detection (with board constraint) ----------------- #

def find_marker_center(frame, surface_poly=None):
    """
    Detect a colored marker (red, green, or blue) and return:
        (center_xy, color_bgr)

    Improvements:
      * Broader HSV ranges so colors are easier to pick up.
      * Area filtering to ignore very tiny noise & very large blobs (hand).
      * If surface_poly is given, only accept markers whose centroid
        lies INSIDE the detected board / paper.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Broader, less strict ranges for typical markers in indoor lighting
    red_lower1 = np.array([0, 80, 60])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 80, 60])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([35, 60, 60])
    green_upper = np.array([90, 255, 255])

    blue_lower = np.array([90, 60, 60])
    blue_upper = np.array([140, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # all colors combined for contour detection
    mask_all = cv2.bitwise_or(mask_red, mask_green)
    mask_all = cv2.bitwise_or(mask_all, mask_blue)

    # gentle cleanup so we don't erase the tip
    mask_all = cv2.GaussianBlur(mask_all, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    # Optional: if we have a surface polygon, we'll enforce "inside board"
    use_polygon = surface_poly is not None

    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignore really tiny specks and huge blobs
        if area < 80 or area > 8000:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if use_polygon:
            # Check if centroid lies inside the board (positive distance)
            dist = cv2.pointPolygonTest(surface_poly.astype(np.float32),
                                        (cx, cy), False)
            if dist < 0:
                # outside board, ignore
                continue

        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None, None

    # Final centroid for best contour
    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)

    # Clamp index just in case
    h_img, w_img = hsv.shape[:2]
    cx = max(0, min(w_img - 1, cx))
    cy = max(0, min(h_img - 1, cy))

    # sample hue at centroid
    h_val = hsv[cy, cx, 0]

    # default color (white / no ink)
    color_bgr = (255, 255, 255)

    if (0 <= h_val <= 15) or (h_val >= 165):
        color_bgr = (0, 0, 255)       # red
    elif 35 <= h_val <= 90:
        color_bgr = (0, 255, 0)       # green
    elif 90 <= h_val <= 140:
        color_bgr = (255, 0, 0)       # blue

    return center, color_bgr


# ----------------- Main loop ----------------- #

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Grab one frame to set canvas size dynamically
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame from camera.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)
    h0, w0 = frame.shape[:2]

    # Canvas matches camera resolution
    CANVAS_W = w0
    CANVAS_H = h0
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    prev_canvas_pt = None
    smoothed_canvas_pt = None  # for exponential smoothing

    # Surface lock state
    surface_pts = None          # most recent detected surface
    surface_pts_locked = None   # frozen corners (when locked)
    lock_surface = False        # True => ignore new detections

    print("Controls:")
    print("  q or ESC : quit")
    print("  c        : clear drawing")
    print("  f        : freeze / unfreeze detected board")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- Surface detection / locking ------------------------------------
        using_fallback = False

        if not lock_surface:
            # Auto-detect surface each frame (contour-based)
            surface_pts = find_surface(frame, min_area_ratio=0.005)

        # Decide what corners to actually use this frame
        if lock_surface and surface_pts_locked is not None:
            # Use frozen corners
            surface_pts_draw = surface_pts_locked
        elif surface_pts is not None:
            # Use the most recent detection
            surface_pts_draw = surface_pts
        else:
            # Fallback to full frame if we can't detect anything
            surface_pts_draw = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype="float32")
            using_fallback = True

        # --- Marker detection (now constrained to the board) -----------------
        marker_center, marker_color = find_marker_center(
            frame, surface_poly=surface_pts_draw
        )
        if marker_center is None:
            prev_canvas_pt = None
            smoothed_canvas_pt = None

        # Destination coordinates for canonical top-down canvas
        dst_pts = np.array([
            [0, 0],
            [CANVAS_W - 1, 0],
            [CANVAS_W - 1, CANVAS_H - 1],
            [0, CANVAS_H - 1]
        ], dtype="float32")

        H, _ = cv2.findHomography(surface_pts_draw, dst_pts)
        H_inv, _ = cv2.findHomography(dst_pts, surface_pts_draw)

        # --- Draw on canvas (in rectified coordinates) ----------------------
        if marker_center is not None and H is not None:
            pts = np.array([[marker_center]], dtype="float32")
            projected = cv2.perspectiveTransform(pts, H)[0][0]
            u_raw, v_raw = projected[0], projected[1]

            if 0 <= u_raw < CANVAS_W and 0 <= v_raw < CANVAS_H:
                alpha = 0.5  # smoothing factor (0 = none, 1 = all new)

                if smoothed_canvas_pt is None:
                    sm_u, sm_v = u_raw, v_raw
                else:
                    prev_u, prev_v = smoothed_canvas_pt
                    sm_u = (1 - alpha) * prev_u + alpha * u_raw
                    sm_v = (1 - alpha) * prev_v + alpha * v_raw

                u, v = int(sm_u), int(sm_v)
                smoothed_canvas_pt = (sm_u, sm_v)

                # Break the stroke if the marker jumps too far in one frame
                max_step = 50  # pixels
                if prev_canvas_pt is not None:
                    dx = u - prev_canvas_pt[0]
                    dy = v - prev_canvas_pt[1]
                    if dx * dx + dy * dy > max_step * max_step:
                        # Treat as new stroke: don't draw a huge connecting line
                        prev_canvas_pt = (u, v)
                    else:
                        cv2.line(canvas, prev_canvas_pt, (u, v),
                                 marker_color, thickness=5)
                        prev_canvas_pt = (u, v)
                else:
                    prev_canvas_pt = (u, v)
            else:
                prev_canvas_pt = None
                smoothed_canvas_pt = None

        # --- Warp canvas back onto camera frame -----------------------------
        if H_inv is not None:
            overlay = cv2.warpPerspective(canvas, H_inv, (w, h))

            overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, mask_overlay = cv2.threshold(
                overlay_gray, 10, 255, cv2.THRESH_BINARY
            )
            mask_overlay_inv = cv2.bitwise_not(mask_overlay)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_overlay_inv)
            drawing_fg = cv2.bitwise_and(overlay, overlay, mask=mask_overlay)

            frame_with_drawing = cv2.add(frame_bg, drawing_fg)
        else:
            frame_with_drawing = frame.copy()

        # Outline the detected / locked surface
        outline_color = (0, 255, 0) if not using_fallback else (0, 165, 255)
        cv2.polylines(
            frame_with_drawing,
            [surface_pts_draw.astype(np.int32)],
            isClosed=True,
            color=outline_color,
            thickness=2
        )

        # Show marker position
        if marker_center is not None:
            cv2.circle(frame_with_drawing, marker_center, 7, marker_color, -1)

        # Combined display: camera | canvas
        disp_h, disp_w = 400, 600
        cam_disp = cv2.resize(frame_with_drawing, (disp_w, disp_h))
        canvas_disp = cv2.resize(canvas, (disp_w, disp_h))
        combined = np.hstack((cam_disp, canvas_disp))

        cv2.imshow("Interactive Whiteboard (Camera | Canvas)", combined)

        # --- Keyboard controls ----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            canvas[:] = 0
            prev_canvas_pt = None
            smoothed_canvas_pt = None
        elif key == ord('f'):
            # Toggle surface lock
            if not lock_surface:
                if surface_pts is not None:
                    surface_pts_locked = surface_pts.copy()
                    lock_surface = True
                    print("Surface LOCKED.")
                else:
                    print("Cannot lock: no surface currently detected.")
            else:
                lock_surface = False
                surface_pts_locked = None
                print("Surface UNLOCKED (auto-detect on).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

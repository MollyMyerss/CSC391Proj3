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


def find_marker_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red
    red_lower1 = np.array([0, 180, 180])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 180, 180])
    red_upper2 = np.array([180, 255, 255])

    # green
    green_lower = np.array([40, 150, 150])
    green_upper = np.array([80, 255, 255])

    # blue
    blue_lower = np.array([90, 150, 150])
    blue_upper = np.array([130, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # all colors combined for contour detection
    mask_all = cv2.bitwise_or(mask_red, mask_green)
    mask_all = cv2.bitwise_or(mask_all, mask_blue)

    # clean up
    mask_all = cv2.GaussianBlur(mask_all, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 200:
        return None, None

    # center from min enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))

    # compute centroid (for sampling color)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = center

    # clamp to image bounds
    h, w = hsv.shape[:2]
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    # sample hue at centroid
    h_val = hsv[cy, cx, 0]

    # default to white (no color)
    color_bgr = (255, 255, 255)

    # determine color based on hue
    if (0 <= h_val <= 10) or (h_val >= 170):
        # red
        color_bgr = (0, 0, 255)
    elif 40 <= h_val <= 80:
        # green
        color_bgr = (0, 255, 0)
    elif 90 <= h_val <= 130:
        # blue
        color_bgr = (255, 0, 0)

    return center, color_bgr


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

    print("Press 'c' to clear, 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # surface detection
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

        dst_pts = np.array([
            [0, 0],
            [CANVAS_W - 1, 0],
            [CANVAS_W - 1, CANVAS_H - 1],
            [0, CANVAS_H - 1]
        ], dtype="float32")

        H, _ = cv2.findHomography(surface_pts, dst_pts)
        H_inv, _ = cv2.findHomography(dst_pts, surface_pts)

        # marker detection
        marker_center, marker_color = find_marker_center(frame)
        if marker_center is None:
            prev_canvas_pt = None
            smoothed_canvas_pt = None 


        # draw on canvas
        if marker_center is not None and H is not None:
            pts = np.array([[marker_center]], dtype="float32")
            projected = cv2.perspectiveTransform(pts, H)[0][0]
            u_raw, v_raw = projected[0], projected[1]

            if 0 <= u_raw < CANVAS_W and 0 <= v_raw < CANVAS_H:
                alpha = 0.3 

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

        outline_color = (0, 255, 0) if not using_fallback else (0, 165, 255)
        cv2.polylines(
            frame_with_drawing,
            [surface_pts.astype(np.int32)],
            isClosed=True,
            color=outline_color,
            thickness=2
        )

        if marker_center is not None:
            cv2.circle(frame_with_drawing, marker_center, 7, marker_color, -1)

        #combined display
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

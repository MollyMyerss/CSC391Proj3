import cv2
import numpy as np


# -----------------------------
#  Utility: order quadrilateral points
# -----------------------------
def order_points(pts):
    """
    Order 4 corner points as:
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


# -----------------------------
#  Surface localization: find white paper on dark background
# -----------------------------
def find_paper_corners(frame, min_area_ratio=0.05):
    """
    Given a BGR frame, find the largest 4-sided contour that looks like
    your white sheet of paper on a darker background.

    Returns:
        corners (4x2 float32 array) or None if not found.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Slight blur → helps edge detection / threshold robustness
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold → separate bright paper from dark wall
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # We expect paper to be the *bright* region.
    # If Otsu inverted things, we can flip based on mean intensity.
    # (Not strictly necessary with black wall + white paper, but safe.)
    if np.mean(gray[thresh == 255]) < np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # Find external contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    frame_area = h * w
    best_quad = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * frame_area:
            continue

        # Approximate to polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # We want 4-sided polygons (the sheet)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        return None

    corners = best_quad.reshape(4, 2).astype("float32")
    corners = order_points(corners)
    return corners


# -----------------------------
#  Marker detection in HSV space
# -----------------------------
def detect_marker_point(board_view, color="red"):
    """
    Given the rectified board view (warped image), detect a colored marker tip.

    Returns:
        (cx, cy) in board coordinates, or None if not found.
    """

    hsv = cv2.cvtColor(board_view, cv2.COLOR_BGR2HSV)

    # You can adjust these ranges depending on which marker you use.
    # Default: RED (two ranges because red wraps around 0° in HSV).
    if color == "red":
        lower1 = np.array([0, 120, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 80])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "green":
        lower = np.array([40, 70, 70])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif color == "blue":
        lower = np.array([100, 120, 70])
        upper = np.array([135, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    else:
        # Default to red if unknown
        lower1 = np.array([0, 120, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 80])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    # --- Morphology for noise reduction ---
    # Use a relatively large kernel so thin drawn lines get eroded away,
    # but the chunky marker tip remains.
    kernel = np.ones((7, 7), np.uint8)

    # Opening = erosion followed by dilation → removes thin noise/lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Slight dilation to consolidate the marker blob
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours (blobs)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Use the largest blob as the marker
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Area threshold: tune this if needed.
    if area < 150:
        return None

    # Compute centroid with spatial moments
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Debug: draw marker on board_view
    cv2.circle(board_view, (cx, cy), 6, (0, 0, 255), -1)

    return (cx, cy)


# -----------------------------
#  Main real-time loop
# -----------------------------
def main():
    # Target size for the rectified virtual board (homography destination)
    BOARD_W, BOARD_H = 800, 600

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Virtual drawing canvas: same size as board plane
    canvas = np.zeros((BOARD_H, BOARD_W, 3), dtype=np.uint8)

    last_point = None
    H_img2board = None  # homography from camera frame → board space
    paper_corners = None

    # Current drawing color in canvas (BGR)
    draw_color = (0, 0, 255)  # red line by default
    marker_color_name = "red"

    print("Controls:")
    print("  ESC : quit")
    print("  c   : clear canvas")
    print("  r   : re-detect paper surface")
    print("  1/2/3 : switch marker color (1=red, 2=green, 3=blue)")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for a more natural feeling
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # If we don't have a homography yet, try to detect the paper
        if H_img2board is None:
            corners = find_paper_corners(frame)

            if corners is not None:
                paper_corners = corners

                # Draw detected paper contour on the original frame
                cv2.polylines(
                    display,
                    [paper_corners.astype(int)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

                # Compute homography from image → canonical board
                dst_pts = np.array(
                    [
                        [0, 0],
                        [BOARD_W - 1, 0],
                        [BOARD_W - 1, BOARD_H - 1],
                        [0, BOARD_H - 1],
                    ],
                    dtype="float32",
                )

                H_img2board = cv2.getPerspectiveTransform(
                    paper_corners, dst_pts
                )

        else:
            # If we already have H, draw the old paper corners back on top
            if paper_corners is not None:
                cv2.polylines(
                    display,
                    [paper_corners.astype(int)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

        # If we have a homography, warp the frame to get the board view
        if H_img2board is not None:
            board_view = cv2.warpPerspective(
                frame, H_img2board, (BOARD_W, BOARD_H)
            )

            # Detect marker in the rectified board space
            pt = detect_marker_point(board_view, color=marker_color_name)

            if pt is not None:
                if last_point is not None:
                    # Draw on the virtual canvas (in board coordinates)
                    cv2.line(canvas, last_point, pt, draw_color, thickness=4)
                last_point = pt
            else:
                # Marker not found → lift pen
                last_point = None

            # Combine board_view and the drawing canvas for visualization
            alpha = 0.4  # transparency of the original board
            beta = 1.0 - alpha
            board_with_drawing = cv2.addWeighted(
                board_view, alpha, canvas, beta, 0
            )

        else:
            # No homography yet → just show an empty board placeholder
            board_with_drawing = np.zeros(
                (BOARD_H, BOARD_W, 3), dtype=np.uint8
            )
            cv2.putText(
                board_with_drawing,
                "Detecting paper...",
                (20, BOARD_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

        # Stack original + board horizontally for a single popup
        # Resize original to match height of board view
        disp_h, disp_w = board_with_drawing.shape[:2]
        scale = disp_h / display.shape[0]
        new_w = int(display.shape[1] * scale)
        display_resized = cv2.resize(display, (new_w, disp_h))

        combined = np.hstack((display_resized, board_with_drawing))

        cv2.imshow("Interactive Whiteboard (Left: Camera, Right: Virtual Board)", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("c"):
            # Clear virtual canvas
            canvas[:] = 0
            last_point = None
        elif key == ord("r"):
            # Force re-detection of paper
            H_img2board = None
            paper_corners = None
            last_point = None
        elif key == ord("1"):
            # Red
            draw_color = (0, 0, 255)
            marker_color_name = "red"
            print("Switched to RED marker detection / drawing.")
        elif key == ord("2"):
            # Green
            draw_color = (0, 255, 0)
            marker_color_name = "green"
            print("Switched to GREEN marker detection / drawing.")
        elif key == ord("3"):
            # Blue
            draw_color = (255, 0, 0)
            marker_color_name = "blue"
            print("Switched to BLUE marker detection / drawing.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def is_rectangle_or_square(approx, angle_tolerance=15, aspect_ratio_tol=0.25):
    if len(approx) != 4:
        return False
    pts = approx.reshape(4, 2)
    angles = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return False
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (norm_v1 * norm_v2)))
        angles.append(angle)
    if not all(abs(a - 90) < angle_tolerance for a in angles):
        return False
    sides = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    min_side = min(sides)
    max_side = max(sides)
    aspect_ratio = max_side / min_side if min_side != 0 else 0
    return (1 - aspect_ratio_tol) <= aspect_ratio <= (1 + aspect_ratio_tol) or aspect_ratio > (1 + aspect_ratio_tol)

def get_fill_percentage(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    if cv2.countNonZero(mask) == 0:
        return 0.0
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    if np.count_nonzero(roi) == 0:
        return 0.0
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return 0.0
    x, y, w, h = cv2.boundingRect(contour)
    sample = hsv[y:y+max(5, h//10), x:x+max(5, w//10)]
    if sample.size == 0:
        return 0.0
    avg_h, avg_s, avg_v = np.mean(sample, axis=(0,1))
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_mask = cv2.bitwise_and(white_mask, mask)
    total = cv2.countNonZero(mask)
    white_pixels = cv2.countNonZero(white_mask)
    fill = total - white_pixels
    percent = (fill / total) * 100 if total > 0 else 0.0
    return round(percent, 2)

class ShapeAnalyzerTransformer(VideoTransformerBase):
    def __init__(self):
        self.min_area = 5000
        self.max_area = 150000

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        display = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        canny = cv2.Canny(thresh, 50, 150)
        thresh = cv2.dilate(canny, None, iterations=2)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if is_rectangle_or_square(approx):
                found = True
                percent = get_fill_percentage(img, approx)
                cv2.drawContours(display, [approx], -1, (0, 255, 0), 3)
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = approx[0][0]
                color = (
                    (0, 255, 0)
                    if percent > 75
                    else (0, 165, 255)
                    if percent > 25
                    else (0, 0, 255)
                )
                label = f"Fill: {percent:.1f}%"
                cv2.rectangle(display, (cx - 60, cy - 30), (cx + 60, cy + 10), (0, 0, 0), -1)
                cv2.putText(
                    display,
                    label,
                    (cx - 55, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
        if not found:
            cv2.putText(
                display,
                "No shape detected ",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        return display

st.title("Real-time Quadrilateral Fill Analyzer")
st.write("Show a quadrilateral to your webcam to see the fill percentage.")

webrtc_streamer(
    key="shape-analyzer",
    video_transformer_factory=ShapeAnalyzerTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
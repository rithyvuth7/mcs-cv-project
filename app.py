from flask import Flask, request, render_template_string, send_file
import cv2
import numpy as np
import joblib
import os
import uuid
from skimage.feature import hog

# ======================================
# Load trained model & labels
# ======================================
svm = joblib.load("svm_vehicle_model_3_class.pkl")
LABELS = joblib.load("labels_3_class.pkl")
ID_TO_LABEL = {v: k for k, v in LABELS.items()}

# ======================================
# Flask setup
# ======================================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ======================================
# Parameters (TUNED & STABLE)
# ======================================
TARGET_WIDTH = 800
STEP_SIZE = 30
WINDOW_SIZES = [(360, 360)]

CLASS_THRESHOLDS = {
    "car": 0.95,
    "motorbike": 0.92,
    "tuktuk": 0.93,
    "bike": 0.92
}
IOU_THRESH = 0.01  # cross-scale suppression
CENTER_DIST_THRESH = 60  # cross-scale suppression

# =====================================================
# Resize image (keep aspect ratio)
# =====================================================
def resize_keep_ratio(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale

# =====================================================
# HOG feature extraction
# =====================================================
def extract_hog(window):
    window = cv2.resize(window, (128, 128))
    return hog(
        window,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

# =====================================================
# Sliding window generator
# =====================================================
def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y+window_size[1], x:x+window_size[0]])

# =====================================================
# Geometry helpers
# =====================================================
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def center_distance(boxA, boxB):
    cx1, cy1 = box_center(boxA)
    cx2, cy2 = box_center(boxB)
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / (areaA + areaB - interArea + 1e-6)

# =====================================================
# CROSS-SCALE NMS (KEY PART)
# =====================================================
def nms_cross_scale(detections):
    """
    detections: (x1,y1,x2,y2,class,confidence)
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    kept = []

    for det in detections:
        x1, y1, x2, y2, cls, conf = det
        keep = True

        for k in kept:
            kx1, ky1, kx2, ky2, kcls, kconf = k
            if cls != kcls:
                continue

            if iou((x1,y1,x2,y2), (kx1,ky1,kx2,ky2)) > IOU_THRESH:
                keep = False
                break

            if center_distance(
                (x1,y1,x2,y2),
                (kx1,ky1,kx2,ky2)
            ) < CENTER_DIST_THRESH:
                keep = False
                break

        if keep:
            kept.append(det)

    return kept

# =====================================================
# Object detection pipeline
# =====================================================
def detect_objects(image_path):
    img = cv2.imread(image_path)
    img, _ = resize_keep_ratio(img, TARGET_WIDTH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = []

    for winW, winH in WINDOW_SIZES:
        for (x, y, window) in sliding_window(gray, STEP_SIZE, (winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            feat = extract_hog(window)
            probs = svm.predict_proba([feat])[0]

            class_id = np.argmax(probs)
            confidence = probs[class_id]
            cls_name = ID_TO_LABEL[class_id]

            if confidence < CLASS_THRESHOLDS.get(cls_name, 0.95):
                continue

            detections.append(
                (x, y, x + winW, y + winH, cls_name, confidence)
            )

    final_boxes = nms_cross_scale(detections)
    return final_boxes, img

# =====================================================
# Draw boxes
# =====================================================
CLASS_COLORS = {
    "car": (0, 255, 0),        # Green
    "motorbike": (255, 0, 0),  # Blue
    "tuktuk": (0, 255, 255),   # Yellow
    "bike": (0, 0, 255),      # Red (if you add later)
}

def draw_boxes(image, boxes):
    for (x1, y1, x2, y2, cls, conf) in boxes:

        # pick color for this class (default = white)
        color = CLASS_COLORS.get(cls, (255, 255, 255))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            f"{cls} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return image


# =====================================================
# Web routes
# =====================================================
HTML_PAGE = """
<!doctype html>
<title>Vehicle Detection</title>
<h2>Upload Image</h2>
<form method="post" enctype="multipart/form-data" action="/detect">
  <input type="file" name="image" required>
  <input type="submit" value="Detect">
</form>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    filename = f"{uuid.uuid4()}.jpg"

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)

    file.save(upload_path)

    boxes, img = detect_objects(upload_path)
    output = draw_boxes(img, boxes)

    cv2.imwrite(result_path, output)
    return send_file(result_path, mimetype="image/jpeg")

# =====================================================
# Run server
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
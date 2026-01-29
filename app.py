from flask import Flask, request, render_template_string
import cv2
import numpy as np
import joblib
import os
import uuid
from skimage.feature import hog

# ===== CNN imports =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# ======================================
# Load SVM model & labels
# ======================================
svm = joblib.load("svm_vehicle_model_3_class.pkl")
LABELS = joblib.load("labels_3_class.pkl")
ID_TO_LABEL = {v: k for k, v in LABELS.items()}

# ======================================
# Load CNN model & labels
# ======================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("cnn_labels.json") as f:
    CNN_CLASSES = json.load(f)

cnn_model = models.resnet18(pretrained=False)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, len(CNN_CLASSES))
cnn_model.load_state_dict(
    torch.load("cnn_vehicle_model.pth", map_location=DEVICE)
)
cnn_model.to(DEVICE)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================
# Flask setup
# ======================================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ======================================
# Parameters (UNCHANGED)
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

IOU_THRESH = 0.01
CENTER_DIST_THRESH = 60

# ======================================
# Resize image
# ======================================
def resize_keep_ratio(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)))

# ======================================
# HOG feature extraction (SVM)
# ======================================
def extract_hog(window):
    window = cv2.resize(window, (128, 128))
    return hog(
        window,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

# ======================================
# Sliding window
# ======================================
def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield x, y, image[y:y+window_size[1], x:x+window_size[0]]

# ======================================
# Geometry helpers
# ======================================
def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def center_distance(a,b):
    ax,ay = box_center(a)
    bx,by = box_center(b)
    return np.sqrt((ax-bx)**2 + (ay-by)**2)

def iou(a,b):
    xA,yA,xB,yB = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,xB-xA+1)*max(0,yB-yA+1)
    areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter/(areaA+areaB-inter+1e-6)

# ======================================
# Cross-scale NMS
# ======================================
def nms_cross_scale(dets):
    dets = sorted(dets, key=lambda x: x[5], reverse=True)
    keep=[]
    for d in dets:
        x1,y1,x2,y2,cls,conf=d
        ok=True
        for k in keep:
            if cls!=k[4]: continue
            if iou((x1,y1,x2,y2),(k[0],k[1],k[2],k[3]))>IOU_THRESH:
                ok=False; break
            if center_distance((x1,y1,x2,y2),(k[0],k[1],k[2],k[3]))<CENTER_DIST_THRESH:
                ok=False; break
        if ok: keep.append(d)
    return keep

# ======================================
# SVM prediction
# ======================================
def svm_predict(window):
    feat = extract_hog(window)
    probs = svm.predict_proba([feat])[0]
    cid = np.argmax(probs)
    return ID_TO_LABEL[cid], probs[cid]

# ======================================
# CNN prediction
# ======================================
def cnn_predict(window):
    window = cv2.cvtColor(window, cv2.COLOR_GRAY2RGB)
    window = cv2.resize(window, (224,224))
    img = Image.fromarray(window)
    img = cnn_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(cnn_model(img), dim=1)[0]
    cid = torch.argmax(probs).item()
    return CNN_CLASSES[cid], probs[cid].item()

# ======================================
# Detection pipelines
# ======================================
def detect(image_path, predictor):
    img = cv2.imread(image_path)
    img = resize_keep_ratio(img, TARGET_WIDTH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets=[]
    for w,h in WINDOW_SIZES:
        for x,y,win in sliding_window(gray, STEP_SIZE, (w,h)):
            cls,conf = predictor(win)
            if conf < CLASS_THRESHOLDS.get(cls,0.95): continue
            dets.append((x,y,x+w,y+h,cls,conf))

    return nms_cross_scale(dets), img

# ======================================
# Draw boxes
# ======================================
COLORS = {
    "car":(0,255,0),
    "motorbike":(255,0,0),
    "tuktuk":(0,255,255),
    "bike":(0,0,255)
}

def draw_boxes(img, boxes):
    for x1,y1,x2,y2,cls,conf in boxes:
        c = COLORS.get(cls,(255,255,255))
        cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
        cv2.putText(img,f"{cls} {conf:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)
    return img

# ======================================
# Web UI
# ======================================
HTML = """
<h2>Vehicle Detection Comparison</h2>
<form method="post" enctype="multipart/form-data" action="/detect">
  <input type="file" name="image" required>
  <input type="submit" value="Detect">
</form>
"""

from flask import Flask, request, render_template_string
import cv2
import numpy as np
import joblib
import os
import uuid
from skimage.feature import hog

# ===== CNN imports =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# ======================================
# Load SVM model & labels
# ======================================
svm = joblib.load("svm_vehicle_model_3_class.pkl")
LABELS = joblib.load("labels_3_class.pkl")
ID_TO_LABEL = {v: k for k, v in LABELS.items()}

# ======================================
# Load CNN model & labels
# ======================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("cnn_labels.json") as f:
    CNN_CLASSES = json.load(f)

cnn_model = models.resnet18(pretrained=False)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, len(CNN_CLASSES))
cnn_model.load_state_dict(
    torch.load("cnn_vehicle_model.pth", map_location=DEVICE)
)
cnn_model.to(DEVICE)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================
# Flask setup
# ======================================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ======================================
# Parameters (UNCHANGED)
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

IOU_THRESH = 0.01
CENTER_DIST_THRESH = 60

# ======================================
# Resize image
# ======================================
def resize_keep_ratio(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)))

# ======================================
# HOG feature extraction (SVM)
# ======================================
def extract_hog(window):
    window = cv2.resize(window, (128, 128))
    return hog(
        window,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

# ======================================
# Sliding window
# ======================================
def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield x, y, image[y:y+window_size[1], x:x+window_size[0]]

# ======================================
# Geometry helpers
# ======================================
def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def center_distance(a,b):
    ax,ay = box_center(a)
    bx,by = box_center(b)
    return np.sqrt((ax-bx)**2 + (ay-by)**2)

def iou(a,b):
    xA,yA,xB,yB = max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,xB-xA+1)*max(0,yB-yA+1)
    areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter/(areaA+areaB-inter+1e-6)

# ======================================
# Cross-scale NMS
# ======================================
def nms_cross_scale(dets):
    dets = sorted(dets, key=lambda x: x[5], reverse=True)
    keep=[]
    for d in dets:
        x1,y1,x2,y2,cls,conf=d
        ok=True
        for k in keep:
            if cls!=k[4]: continue
            if iou((x1,y1,x2,y2),(k[0],k[1],k[2],k[3]))>IOU_THRESH:
                ok=False; break
            if center_distance((x1,y1,x2,y2),(k[0],k[1],k[2],k[3]))<CENTER_DIST_THRESH:
                ok=False; break
        if ok: keep.append(d)
    return keep

# ======================================
# SVM prediction
# ======================================
def svm_predict(window):
    feat = extract_hog(window)
    probs = svm.predict_proba([feat])[0]
    cid = np.argmax(probs)
    return ID_TO_LABEL[cid], probs[cid]

# ======================================
# CNN prediction
# ======================================
def cnn_predict(window):
    window = cv2.cvtColor(window, cv2.COLOR_GRAY2RGB)
    window = cv2.resize(window, (224,224))
    img = Image.fromarray(window)
    img = cnn_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(cnn_model(img), dim=1)[0]
    cid = torch.argmax(probs).item()
    return CNN_CLASSES[cid], probs[cid].item()

# ======================================
# Detection pipelines
# ======================================
def detect(image_path, predictor):
    img = cv2.imread(image_path)
    img = resize_keep_ratio(img, TARGET_WIDTH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets=[]
    for w,h in WINDOW_SIZES:
        for x,y,win in sliding_window(gray, STEP_SIZE, (w,h)):
            cls,conf = predictor(win)
            if conf < CLASS_THRESHOLDS.get(cls,0.95): continue
            dets.append((x,y,x+w,y+h,cls,conf))

    return nms_cross_scale(dets), img

# ======================================
# Draw boxes
# ======================================
COLORS = {
    "car":(0,255,0),
    "motorbike":(255,0,0),
    "tuktuk":(0,255,255),
    "bike":(0,0,255)
}

def draw_boxes(img, boxes):
    for x1,y1,x2,y2,cls,conf in boxes:
        c = COLORS.get(cls,(255,255,255))
        cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
        cv2.putText(img,f"{cls} {conf:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)
    return img

# ======================================
# Web UI
# ======================================
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Vehicle Detection Comparison</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f6f8;
      text-align: center;
      padding: 40px;
    }
    .container {
      background: white;
      padding: 30px;
      max-width: 500px;
      margin: auto;
      border-radius: 10px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    h1 {
      margin-bottom: 10px;
    }
    p {
      color: #666;
    }
    input[type=file] {
      margin: 20px 0;
    }
    button {
      background: #007bff;
      color: white;
      border: none;
      padding: 12px 25px;
      border-radius: 5px;
      font-size: 16px;
      cursor: pointer;
    }
    button:hover {
      background: #0056b3;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Vehicle Detection</h1>
    <p>SVM (HOG) vs CNN (ResNet)</p>

    <form method="post" enctype="multipart/form-data" action="/detect">
      <input type="file" name="image" required><br>
      <button type="submit">Run Detection</button>
    </form>
  </div>
</body>
</html>
"""


RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Detection Results</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #f4f6f8;
      padding: 30px;
      text-align: center;
    }}
    h1 {{
      margin-bottom: 20px;
    }}
    .grid {{
      display: flex;
      justify-content: center;
      gap: 30px;
      flex-wrap: wrap;
    }}
    .card {{
      background: white;
      padding: 15px;
      border-radius: 10px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      width: 420px;
    }}
    .card h2 {{
      margin-bottom: 10px;
    }}
    img {{
      max-width: 100%;
      border-radius: 5px;
      border: 1px solid #ddd;
    }}
    a {{
      display: inline-block;
      margin-top: 30px;
      text-decoration: none;
      color: white;
      background: #28a745;
      padding: 10px 20px;
      border-radius: 5px;
    }}
    a:hover {{
      background: #1e7e34;
    }}
  </style>
</head>
<body>

<h1>Detection Comparison</h1>

<div class="grid">
  <div class="card">
    <h2>SVM (HOG)</h2>
    <img src="/static/{svm_img}">
  </div>

  <div class="card">
    <h2>CNN (ResNet)</h2>
    <img src="/static/{cnn_img}">
  </div>
</div>

<a href="/">Try Another Image</a>

</body>
</html>
"""


@app.route("/")
def index():
    return HTML

@app.route("/detect", methods=["POST"])
def detect_route():
    file = request.files["image"]
    name = f"{uuid.uuid4()}.jpg"
    up = os.path.join(UPLOAD_FOLDER, name)
    file.save(up)

    svm_boxes, svm_img = detect(up, svm_predict)
    cnn_boxes, cnn_img = detect(up, cnn_predict)

    svm_out = f"svm_{name}"
    cnn_out = f"cnn_{name}"

    cv2.imwrite(os.path.join("static", svm_out), draw_boxes(svm_img, svm_boxes))
    cv2.imwrite(os.path.join("static", cnn_out), draw_boxes(cnn_img, cnn_boxes))

    return RESULT_HTML.format(svm_img=svm_out, cnn_img=cnn_out)

# ======================================
# Run
# ======================================
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)

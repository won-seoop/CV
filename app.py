import io
import os
import sys
import base64
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- Model load ---

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] model.pt not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        class_names = checkpoint["class_names"]

        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, len(class_names)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[INFO] Model loaded. Classes: {class_names}")
        return model, class_names
    except Exception as e:
        print(f"[ERROR] Failed to load model.pt: {e}", file=sys.stderr)
        sys.exit(1)

model, CLASS_NAMES = load_model()

# --- Preprocessing (ImageNet standard) ---

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Inference ---

def classify(pil_img):
    tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    top_prob, top_idx = probs.max(0)
    return CLASS_NAMES[top_idx.item()], round(top_prob.item() * 100, 1)

# --- HTML ---

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Image Classifier</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111; color: #fff; font-family: sans-serif;
      display: flex; flex-direction: column; align-items: center;
      padding: 24px; gap: 16px;
    }
    h1 { font-size: 1.4rem; }
    #wrap { position: relative; display: inline-block; }
    video {
      display: block; width: 640px; border-radius: 8px;
      transform: scaleX(-1);
    }
    #overlay {
      position: absolute; top: 16px; left: 50%;
      transform: translateX(-50%);
      background: rgba(0,0,0,0.6);
      color: #00e676;
      font-size: 1.6rem; font-weight: bold;
      padding: 8px 20px; border-radius: 8px;
      pointer-events: none;
      white-space: nowrap;
    }
    #captureCanvas { display: none; }
    #status { font-size: 0.78rem; color: #888; }
  </style>
</head>
<body>
  <h1>Image Classifier — {{ classes }}</h1>
  <div id="wrap">
    <video id="video" autoplay playsinline></video>
    <div id="overlay">—</div>
  </div>
  <canvas id="captureCanvas"></canvas>
  <p id="status">Starting webcam...</p>

  <script>
    const video   = document.getElementById('video');
    const capCvs  = document.getElementById('captureCanvas');
    const capCtx  = capCvs.getContext('2d');
    const overlay = document.getElementById('overlay');
    const status  = document.getElementById('status');

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          capCvs.width  = video.videoWidth;
          capCvs.height = video.videoHeight;
          status.textContent = 'Running...';
          sendFrame();
        };
      })
      .catch(err => { status.textContent = 'Webcam error: ' + err.message; });

    async function sendFrame() {
      // 좌우반전 적용해서 서버에 전송 (화면과 동일 방향)
      capCtx.save();
      capCtx.scale(-1, 1);
      capCtx.translate(-capCvs.width, 0);
      capCtx.drawImage(video, 0, 0);
      capCtx.restore();

      const dataURL = capCvs.toDataURL('image/jpeg', 0.8);

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataURL }),
        });
        const data = await res.json();
        overlay.textContent = `${data.label}  ${data.score}%`;
        status.textContent  = `Last: ${data.label} (${data.score}%)`;
      } catch(e) {
        status.textContent = 'Error: ' + e.message;
      }

      setTimeout(sendFrame, 2000);
    }
  </script>
</body>
</html>
"""

# --- Routes ---

@app.route("/")
def index():
    return render_template_string(HTML, classes=" / ".join(CLASS_NAMES))

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.json.get("image", "")
    _, enc = data.split(",", 1)
    img    = Image.open(io.BytesIO(base64.b64decode(enc)))
    label, score = classify(img)
    return jsonify({"label": label, "score": score})

if __name__ == "__main__":
    print(f"Server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)

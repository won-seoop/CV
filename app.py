import io
import base64
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- Model (COCO object detection) ---

weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
model   = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()
COCO_LABELS = weights.meta["categories"]

CONFIDENCE_THRESHOLD = 0.5

# --- Inference ---

def detect(pil_img):
    tensor = TF.to_tensor(pil_img.convert("RGB"))   # [3,H,W] 0~1
    with torch.no_grad():
        output = model([tensor])[0]

    keep = output["scores"] > CONFIDENCE_THRESHOLD
    results = []
    for box, label, score in zip(
        output["boxes"][keep],
        output["labels"][keep],
        output["scores"][keep],
    ):
        x1, y1, x2, y2 = [round(v) for v in box.tolist()]
        results.append({
            "box":   [x1, y1, x2, y2],
            "label": COCO_LABELS[label.item()],
            "score": round(score.item() * 100, 1),
        })
    return results

# --- HTML ---

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Object Detection Live</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111; color: #fff; font-family: sans-serif;
      display: flex; flex-direction: column; align-items: center;
      padding: 24px; gap: 16px;
    }
    h1 { font-size: 1.4rem; }
    #wrap {
      position: relative;
      display: inline-block;
    }
    video {
      display: block;
      width: 640px;
      border-radius: 8px;
      transform: scaleX(-1); /* 비디오만 좌우반전 */
    }
    #boxCanvas {
      position: absolute;
      top: 0; left: 0;
      pointer-events: none;
    }
    #captureCanvas { display: none; }
    #status { font-size: 0.78rem; color: #888; }
  </style>
</head>
<body>
  <h1>SSDLite Object Detection</h1>
  <div id="wrap">
    <video id="video" autoplay playsinline></video>
    <canvas id="boxCanvas"></canvas>
  </div>
  <canvas id="captureCanvas"></canvas>
  <p id="status">Starting webcam...</p>

  <script>
    const video         = document.getElementById('video');
    const boxCanvas     = document.getElementById('boxCanvas');
    const captureCanvas = document.getElementById('captureCanvas');
    const boxCtx        = boxCanvas.getContext('2d');
    const capCtx        = captureCanvas.getContext('2d');
    const status        = document.getElementById('status');

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          const w = video.videoWidth;
          const h = video.videoHeight;
          boxCanvas.width     = w; boxCanvas.height     = h;
          captureCanvas.width = w; captureCanvas.height = h;
          status.textContent = 'Running...';
          sendFrame();          // 첫 탐지 즉시 실행
        };
      })
      .catch(err => { status.textContent = 'Webcam error: ' + err.message; });

    async function sendFrame() {
      // 캡처: 좌우반전 적용해서 서버에 전송 (화면과 동일 방향)
      capCtx.save();
      capCtx.scale(-1, 1);
      capCtx.translate(-captureCanvas.width, 0);
      capCtx.drawImage(video, 0, 0);
      capCtx.restore();

      const dataURL = captureCanvas.toDataURL('image/jpeg', 0.8);

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataURL }),
        });
        const data = await res.json();
        drawBoxes(data.results);
        status.textContent = `Detected: ${data.results.length} object(s)`;
      } catch(e) {
        status.textContent = 'Error: ' + e.message;
      }

      setTimeout(sendFrame, 2000); // 2초마다 탐지
    }

    function drawBoxes(results) {
      boxCtx.clearRect(0, 0, boxCanvas.width, boxCanvas.height);
      if (!results || !results.length) return;

      results.forEach(det => {
        const [x1, y1, x2, y2] = det.box;
        const w = x2 - x1;
        const h = y2 - y1;
        const label = `${det.label} ${det.score}%`;

        // 박스
        boxCtx.strokeStyle = '#00e676';
        boxCtx.lineWidth   = 2;
        boxCtx.strokeRect(x1, y1, w, h);

        // 라벨 배경
        boxCtx.font = 'bold 13px sans-serif';
        const textW = boxCtx.measureText(label).width;
        boxCtx.fillStyle = '#00e676';
        boxCtx.fillRect(x1, y1 - 22, textW + 8, 22);

        // 라벨 텍스트
        boxCtx.fillStyle = '#000';
        boxCtx.fillText(label, x1 + 4, y1 - 6);
      });
    }
  </script>
</body>
</html>
"""

# --- Routes ---

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data    = request.json.get("image", "")
    _, enc  = data.split(",", 1)
    img     = Image.open(io.BytesIO(base64.b64decode(enc)))
    results = detect(img)
    return jsonify({"results": results})

if __name__ == "__main__":
    print("Loading SSDLite model...")
    print("Server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)

# MobileNetV2 파인튜닝 — 연필 / 핸드폰 분류

### 실험 필요성

SSDLite의 COCO pretrained 모델은 연필과 핸드폰에 대한 객체 탐지 성능이 좋지 않아, MobileNetV2를 해당 클래스에 맞게 파인튜닝하여 성능을 확인하였다.

---

### 데모

| 연필 탐지 | 핸드폰 탐지 |
|-----------|------------|
| <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/e7b07906-a7b1-4902-ac59-499526ca0e06" />| <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/eb3fe887-bcdb-4a62-842e-3fffcbe5b5af" />|

---

### 파이프라인

```
Bing 이미지 크롤링
       ↓
train / val 분할 (80:20)
       ↓
MobileNetV2 파인튜닝 (Google Colab, GPU)
       ↓
model.pt 저장
       ↓
Flask 서버에서 웹캠 실시간 분류
```

---

### 데이터

| 클래스 | 수집 | train | val |
|--------|------|-------|-----|
| pencil | 67장 | 54장  | 13장 |
| phone  | 64장 | 52장  | 12장 |

- icrawler (Bing) 로 자동 수집
- 키워드: `pencil object white background`, `smartphone mobile phone object`

<!-- 크롤링 샘플 이미지 (클래스별 5장) -->
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/d688ec44-e582-4a6d-918d-7d6da7313916" />

---

### 모델 구조

```
MobileNetV2 (ImageNet pretrained)
  ├─ features     → Freeze (학습 안 함)
  └─ classifier   → 교체하여 학습
       Dropout(0.3)
       Linear(1280 → 2)   # 학습 파라미터 2,562개
```

---

### 학습 설정

| 항목 | 값 |
|------|---|
| Epochs | 20 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | StepLR (step=7, γ=0.1) |
| Loss | CrossEntropyLoss |
| 전처리 (train) | Resize(224) + RandomFlip + RandomRotation(15°) + ColorJitter + Normalize |
| 전처리 (val) | Resize(224) + Normalize |
| 환경 | Google Colab (Tesla T4 GPU) |

---

### 학습 결과

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1  | 0.7547 | 57.6% | 0.8460 | 48.0% |
| 3  | 0.4490 | 80.2% | 0.3009 | **100.0%** |
| 20 | 0.1727 | 94.3% | 0.1351 | **100.0%** |

**최고 검증 정확도: 100%** (Epoch 3부터 수렴)

<!-- 학습 곡선 그래프 (Loss / Accuracy) -->
<img width="1674" height="616" alt="image" src="https://github.com/user-attachments/assets/e03861ea-fead-4a40-9941-01f03698f3a3" />

---

### 실행 방법

#### 로컬 (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

#### Docker Compose

```bash
docker compose up --build
# → http://localhost:8888
```

> `model.pt` 파일이 프로젝트 루트에 있어야 합니다.

---

### 파일 구조

```
CV/
├── app.py              # Flask 서버 (웹캠 분류)
├── train.ipynb         # Colab 학습 노트북
├── model.pt            # 학습된 가중치
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

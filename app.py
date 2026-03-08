import cv2  # OpenCV: 웹캠 프레임 읽기/그리기/화면 출력에 사용
import torch  # PyTorch: 모델 추론과 텐서 연산에 사용
import torchvision.transforms as transforms  # 이미지 전처리 파이프라인 구성
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights  # MobileNetV2 모델과 사전학습 가중치 enum
from PIL import Image  # OpenCV 이미지를 PIL 이미지로 변환할 때 사용

# ImageNet class labels (top-level categories abbreviated)
IMAGENET_CLASSES_URL = None  # 현재 코드에서는 미사용(라벨은 weights.meta에서 직접 읽음)


def load_model():  # 모델과 라벨 목록을 로드해서 반환
    weights = MobileNet_V2_Weights.IMAGENET1K_V1  # ImageNet-1K로 사전학습된 v1 가중치 선택
    model = mobilenet_v2(weights=weights)  # 선택한 가중치를 적용해 MobileNetV2 생성
    model.eval()  # 추론 모드로 전환(dropout/batchnorm 학습 동작 비활성화)
    labels = weights.meta["categories"]  # 클래스 인덱스(0~999)에 대응하는 라벨 문자열 목록
    return model, labels  # 호출부에서 바로 쓰도록 모델과 라벨을 함께 반환


def build_transform():  # 웹캠 프레임을 모델 입력 형태로 바꾸는 전처리 생성
    return transforms.Compose([  # 여러 전처리를 순서대로 적용하는 파이프라인
        transforms.Resize(256),  # 짧은 변을 256으로 리사이즈
        transforms.CenterCrop(224),  # 중앙 224x224 영역만 잘라 모델 입력 크기에 맞춤
        transforms.ToTensor(),  # PIL 이미지를 [0,1] 범위 텐서(C,H,W)로 변환
        transforms.Normalize(  # ImageNet 기준 평균/표준편차로 정규화
            mean=[0.485, 0.456, 0.406],  # RGB 채널 평균
            std=[0.229, 0.224, 0.225],  # RGB 채널 표준편차
        ),
    ])


def classify_frame(model, labels, transform, frame):  # 단일 프레임에서 top-5 분류 결과 계산
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV BGR 프레임을 RGB로 변환
    pil_img = Image.fromarray(rgb)  # torchvision transform 적용을 위해 PIL 이미지로 변환
    tensor = transform(pil_img).unsqueeze(0)  # 배치 차원 추가: (1, 3, 224, 224)

    with torch.no_grad():  # 추론 시 gradient 계산 비활성화(메모리/속도 이점)
        outputs = model(tensor)  # 모델 출력 로짓 계산(클래스별 점수)
        probs = torch.softmax(outputs[0], dim=0)  # 로짓을 확률 분포로 변환

    top5_probs, top5_idx = torch.topk(probs, 5)  # 확률 상위 5개 값/인덱스 추출
    results = [  # (라벨, 확률) 형태의 결과 리스트 구성
        (labels[idx.item()], prob.item())  # 텐서를 파이썬 스칼라로 꺼내 사람이 읽기 쉽게 변환
        for idx, prob in zip(top5_idx, top5_probs)  # 인덱스와 확률을 짝지어 순회
    ]
    return results  # top-5 결과 반환


def draw_results(frame, results):  # 프레임 위에 top-5 결과를 시각적으로 오버레이
    overlay = frame.copy()  # 반투명 배경 박스용 복사본
    box_h = 30 + len(results) * 28  # 결과 개수에 맞춰 정보 박스 높이 계산
    cv2.rectangle(overlay, (8, 8), (420, box_h), (0, 0, 0), -1)  # 검은 배경 직사각형 채우기
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # 원본과 섞어 반투명 효과 적용

    cv2.putText(frame, "Top-5 Predictions", (14, 28),  # 제목 텍스트 위치/내용 지정
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)  # 글꼴/크기/색/두께/안티앨리어싱

    for i, (label, prob) in enumerate(results):  # 각 예측 결과를 한 줄씩 그림
        y = 28 + (i + 1) * 28  # i번째 항목의 y 좌표 계산
        bar_w = int(prob * 200)  # 확률(0~1)을 막대 너비(px)로 변환
        color = (0, 200, 80) if i == 0 else (80, 160, 200)  # 1등은 초록, 나머지는 파랑 계열
        cv2.rectangle(frame, (14, y - 16), (14 + bar_w, y), color, -1)  # 확률 막대 그리기
        text = f"{label[:28]}: {prob * 100:.1f}%"  # 라벨(최대 28자) + 백분율 문자열 생성
        cv2.putText(frame, text, (14, y - 2),  # 각 항목 텍스트 위치/내용 지정
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)  # 텍스트 렌더링 옵션

    return frame  # 주석이 추가된 프레임 반환


def main():  # 프로그램 진입점: 모델 로드, 웹캠 루프, 화면 표시
    print("Loading MobileNetV2 model...")  # 모델 로드 시작 로그
    model, labels = load_model()  # 모델과 라벨 목록 초기화
    transform = build_transform()  # 전처리 파이프라인 준비
    print("Model loaded. Opening webcam (/dev/video0)...")  # 웹캠 열기 시작 로그

    cap = cv2.VideoCapture(0)  # 기본 웹캠 장치(인덱스 0) 열기
    if not cap.isOpened():  # 웹캠 장치 오픈 실패 여부 확인
        raise RuntimeError("Cannot open webcam. Check /dev/video0 device.")  # 실패 시 즉시 예외 발생

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처 프레임 너비 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처 프레임 높이 설정

    frame_count = 0  # 프레임 카운터(주기적 추론 제어용)
    results = []  # 최근 추론 결과 저장 변수

    print("Press 'q' to quit.")  # 종료 키 안내
    while True:  # 사용자 종료 전까지 반복
        ret, frame = cap.read()  # 웹캠에서 프레임 1장 읽기
        if not ret:  # 프레임 읽기 실패 시
            print("Failed to read frame.")  # 에러 로그 출력
            break  # 루프 종료

        # Run inference every 5 frames to keep display responsive
        if frame_count % 5 == 0:  # 5프레임마다 한 번만 추론해 FPS 저하 완화
            results = classify_frame(model, labels, transform, frame)  # 현재 프레임 분류 수행

        if results:  # 표시할 결과가 있으면
            frame = draw_results(frame, results)  # 결과 오버레이 렌더링

        cv2.imshow("MobileNetV2 Object Classification", frame)  # 화면 창에 현재 프레임 표시
        frame_count += 1  # 프레임 카운터 증가

        if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q' 입력 시 종료
            break  # 루프 탈출

    cap.release()  # 웹캠 장치 해제
    cv2.destroyAllWindows()  # OpenCV 창 모두 닫기


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 main() 호출
    main()  # 프로그램 시작

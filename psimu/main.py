import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os

# 1. FastAPI 앱 설정
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MediaPipe Face Landmarker 설정
# 모델 파일 경로 (main.py와 같은 위치에 있다고 가정)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_landmarker.task")

print("MODEL_PATH:", model_path, "exists?", os.path.exists(model_path))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"'{model_path}' 파일을 찾을 수 없습니다. main.py와 같은 폴더에 위치시켜주세요.")

# BaseOptions: 모델 파일 지정
base_options = python.BaseOptions(model_asset_path=model_path)

# FaceLandmarkerOptions: 실행 모드 및 출력 설정
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True, # 표정 분석(블렌드쉐이프) 필요 시 True
    output_facial_transformation_matrixes=True,
    num_faces=1,
)

# Landmarker 인스턴스 생성
detector = vision.FaceLandmarker.create_from_options(options)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> Client Connected")

    try:
        while True:
            # 1. 데이터 수신
            data = await websocket.receive_text()

            try:
                # 2. Base64 디코딩
                if ',' in data:
                    data = data.split(',')[1]
                
                image_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                
                # BGR 이미지 (OpenCV 포맷)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # 3. MediaPipe Image 객체로 변환
                # OpenCV는 BGR을 쓰지만, MediaPipe는 RGB를 기대합니다.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # 4. 추론 실행 (detect: 이미지 모드)
                # 비디오 스트림이지만 WebSocket 특성상 단일 이미지 처리(detect)가 구현이 간편하고 안정적입니다.
                detection_result = detector.detect(mp_image)

                # 5. 결과 출력
                if detection_result.face_landmarks:
                    # 첫 번째 얼굴의 랜드마크 가져오기
                    face_landmarks = detection_result.face_landmarks[0]
                    
                    # 예: 코 끝 (인덱스 1) 좌표 출력
                    nose_tip = face_landmarks[1]
                    
                    # Blendshapes(표정 점수)가 있다면 출력 (예: 눈 깜빡임, 입 벌림 등)
                    # face_blendshapes = detection_result.face_blendshapes[0]
                    
                    print(f"[Face Detected] Nose: x={nose_tip.x:.2f}, y={nose_tip.y:.2f} | Landmarks Count: {len(face_landmarks)}")
                else:
                    print("[No Face] Waiting for user...")

            except Exception as e:
                print(f"Error processing frame: {e}")

    except WebSocketDisconnect:
        print(">>> Client Disconnected")
    except Exception as e:
        print(f">>> Connection Error: {e}")
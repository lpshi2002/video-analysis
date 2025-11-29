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
# 모델 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"'{model_path}' 파일을 찾을 수 없습니다. main.py와 같은 폴더에 위치시켜주세요.")


def get_blendshape_map(blendshapes):
    return {b.category_name: b.score for b in blendshapes}

def matrix_to_euler(matrix):
    R = matrix[:3, :3]
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw = np.arctan2(-R[2,0], sy)
        roll = 0
        
    return pitch, yaw, roll

# smile detection
def smile(blendshapes):
    m = get_blendshape_map(blendshapes)
    left=m.get("mouthSmileLeft", 0.0)
    right=m.get("mouthSmileRight", 0.0)
    # 눈가 미소
    squint_l = m.get("eyeSquintLeft", 0.0)
    squint_r = m.get("eyeSquintRight", 0.0)
    
    base_smile = (left + right) / 2.0
    duchenne_bonus = (squint_l + squint_r) / 2.0 * 0.5
    score = np.clip(base_smile + duchenne_bonus, 0.0, 1.0)
    
    return float(score)
    
# head pose detection
def head_pose_matrix(result):
    if not result.facial_transformation_matrixes:
        return None
    
    m = result.facial_transformation_matrixes[0]
    matrix4 = np.array(m, dtype = np.float32).reshape(4,4).T
    
    pitch,yaw,roll = matrix_to_euler(matrix4)
    return pitch, yaw, roll

# eye contact detection
def eye_contact(blendshapes):
    m = get_blendshape_map(blendshapes)

    # 각 눈에 대해 "정면에서 벗어난 정도" 계산
    # in/out/up/down이 크면 클수록 정면에서 벗어난 것
    left_dev = (
        m.get("eyeLookInLeft", 0.0)
        + m.get("eyeLookOutLeft", 0.0)
        + m.get("eyeLookUpLeft", 0.0)
        + m.get("eyeLookDownLeft", 0.0)
    ) / 4.0

    right_dev = (
        m.get("eyeLookInRight", 0.0)
        + m.get("eyeLookOutRight", 0.0)
        + m.get("eyeLookUpRight", 0.0)
        + m.get("eyeLookDownRight", 0.0)
    ) / 4.0

    dev = (left_dev + right_dev) / 2.0  # 0(정면) ~ 1(완전 옆/위/아래)

    # "벗어난 정도"의 반대로 eye-contact 점수 만들기
    score = 1.0 - np.clip(dev, 0.0, 1.0)
    return float(score)

# web
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> Client Connected")

    # BaseOptions: 모델 파일 지정
    base_options = python.BaseOptions(model_asset_path=model_path)

    # FaceLandmarkerOptions: 실행 모드 및 출력 설정
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, # 비디오 스트림 모드
        min_face_detection_confidence=0.5, # 얼굴 감지 최소 신뢰도
        min_face_presence_confidence=0.5, # 얼굴 존재 최소 신뢰도
        min_tracking_confidence=0.5, # 추적 최소 신뢰도
        output_face_blendshapes=True, # 표정 분석(블렌드쉐이프) 필요 시 True
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    # Landmarker 인스턴스 생성
    detector = vision.FaceLandmarker.create_from_options(options)
    
    frame_id = 0

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

                # 프레임 타임스탬프 계산 (ms 단위)
                frame_id += 1
    
                # 4. 추론 실행
                detection_result = detector.detect_for_video(mp_image,frame_id)

                # 5. 결과 출력
                if detection_result.face_landmarks:
                    # 첫 번째 얼굴의 랜드마크 가져오기
                    face_landmarks = detection_result.face_landmarks[0]
                    # 얼굴 표정(블렌드쉐이프) 가져오기
                    blend = detection_result.face_blendshapes[0]
                    
                    smile_score = smile(blend)# 미소 점수
                    pitch, yaw, roll = head_pose_matrix(detection_result)  # 머리 자세
                    eye_contact_score = eye_contact(blend)  # 아이 컨택트 점수
                    
                    # 결과 출력
                    print(f"[Face Detected] Smile: {smile_score:.2f}, Head Pose (Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}), Eye Contact: {eye_contact_score:.2f}")
                    
                else:
                    print("[No Face] Waiting for user...")

            except Exception as e:
                print(f"Error processing frame: {e}")

    except WebSocketDisconnect:
        print(">>> Client Disconnected")
    except Exception as e:
        print(f">>> Connection Error: {e}")
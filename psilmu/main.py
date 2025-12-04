import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os

# face.py 에 있는 분석 함수들 import
import face  # 같은 디렉토리에 face.py가 있어야 함

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> Client Connected")

    # BaseOptions: 모델 파일 지정
    base_options = python.BaseOptions(model_asset_path=model_path)

    # FaceLandmarkerOptions: 실행 모드 및 출력 설정
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # 비디오 스트림 모드
        min_face_detection_confidence=0.5,      # 얼굴 감지 최소 신뢰도
        min_face_presence_confidence=0.5,       # 얼굴 존재 최소 신뢰도
        min_tracking_confidence=0.5,            # 추적 최소 신뢰도
        output_face_blendshapes=True,           # 표정 분석(블렌드쉐이프)
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    # Landmarker 인스턴스 생성
    detector = vision.FaceLandmarker.create_from_options(options)

    frame_id = 0

    # --- 인터뷰 통계용 누적 변수 (face.py VIDEO SUMMARY 형식 그대로) ---
    face_frame_count = 0
    smile_sum = 0.0
    eye_sum = 0.0
    eye_on_frames = 0
    pitch_history = []
    prev_nose_z = None
    nod_count = 0
    lean_forward_count = 0

    # 임계값 (face.py에서 쓰는 EYE_TH와 맞춰 사용)
    SMILE_TH = 0.15
    EYE_TH = 0.4

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

                # 3. MediaPipe Image 객체로 변환 (BGR -> RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                frame_id += 1

                # 프레임 수 줄이기 (2장마다 1번만 추론)
                if frame_id % 2 != 0:
                    continue

                # 4. 추론 실행
                detection_result = detector.detect_for_video(mp_image, frame_id)

                # 5. 결과 처리
                if detection_result.face_landmarks:
                    face_landmarks = detection_result.face_landmarks[0]
                    blend = detection_result.face_blendshapes[0]

                    # --- face.py 함수들 사용 ---
                    smile_score = face.smile(blend)  # 미소 점수
                    pitch, yaw, roll = face.head_pose_matrix(detection_result)
                    eye_contact_score = face.combined_eye_contact(
                        blendshapes=blend,
                        landmarks=face_landmarks,
                        pitch=pitch,
                        yaw=yaw,
                        max_dev=0.7,
                    )

                    # --- 통계 누적 ---
                    face_frame_count += 1
                    smile_sum += smile_score
                    eye_sum += eye_contact_score

                    if eye_contact_score >= EYE_TH:
                        eye_on_frames += 1

                    # 끄덕임 감지 (face.detect_nod 사용)
                    pitch_history.append(pitch)
                    if face.detect_nod(pitch_history):
                        nod_count += 1
                        pitch_history = []  # 한 번 감지 후 초기화

                    # 몸 앞으로 숙이기 감지 (face.detect_lean_forward 사용)
                    nose = face_landmarks[1]  # 코 근처 인덱스
                    cur_nose_z = nose.z
                    if face.detect_lean_forward(prev_nose_z, cur_nose_z):
                        lean_forward_count += 1
                    prev_nose_z = cur_nose_z

                    # 디버깅 로그
                    if frame_id % 10 == 0:
                        print(
                            f"[Frame {frame_id}] "
                            f"Smile: {smile_score:.2f}, "
                            f"Eye: {eye_contact_score:.2f}, "
                            f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}"
                        )

                else:
                    print("[No Face] Waiting for user...")

            except Exception as e:
                print(f"Error processing frame: {e}")

    except WebSocketDisconnect:
        if face_frame_count > 0:
            avg_smile_0_1 = smile_sum / face_frame_count
            avg_eye_0_1 = eye_sum / face_frame_count

            avg_smile_0_100 = avg_smile_0_1 * 100.0
            avg_eye_0_100 = avg_eye_0_1 * 100.0

            eye_ratio = eye_on_frames / face_frame_count

            print("\n=== VIDEO SUMMARY ===")
            print(f"Mean SmileIntensity: {avg_smile_0_100:.2f} / 100")
            print(f"Mean EyeContact: {avg_eye_0_100:.2f} / 100")
            print(f"EyeContact ratio (> {EYE_TH}): {eye_ratio * 100:.1f}%")
            print(f"Nod Count: {nod_count}")
            print(f"LeanForward Count: {lean_forward_count}")
            print(f"z-normalized Smile: {(avg_smile_0_100 - 31.09) / 16.50:.2f}")
            print(f"z-normalized EyeContact: {(avg_eye_0_100 - 50.21) / 6.17:.2f}")
            print(f"z-normalized Eye ratio: {(eye_ratio - .8346) / .1990:.2f}")
        else:
            print("No face detected in this session.")

        print(">>> Client Disconnected")

    except Exception as e:
        print(f">>> Connection Error: {e}")
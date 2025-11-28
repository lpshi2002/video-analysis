import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

#기본 옵션과 얼굴 랜드마커 옵션 설정
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

#모델을 다운받아서 사용하므로 다운받은 경로 설정
model_path = r"C:/coding/myenv/psimu/face_landmarker.task"

# face landmarker 옵션 설정
option = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1,
    min_face_detection_confidence = 0.5,
    min_face_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    output_face_blendshapes = True,
    output_facial_transformation_matrixes = True,
)

#open CV로 웹캠 열기
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(option) as landmarker:
    prev_time = time.time()
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # OpenCV는 BGR, MediaPipe는 RGB 필요
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image로 감싸기
        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = frame_rgb
        )
        
         # timestamp (ms 단위) – VIDEO 모드에서는 꼭 넣어야 함  
        timestamp_ms = int(time.time() * 1000)
        
        # 얼굴 랜드마킹
        result = landmarker.detect_for_video(mp_image, timestamp_ms)# 비디오 모드 전용 함수
        
        # 결과 그리기
        if result.face_landmarks:
            h, w, _ = frame_bgr.shape
            for lm in result.face_landmarks[0]:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)
                    
        cv2.imshow("Face Landmarker", frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27: # ESC 키
            break
        
cap.release()
cv2.destroyAllWindows()
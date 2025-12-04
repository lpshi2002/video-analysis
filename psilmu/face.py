import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

# ============================================================
# 1. 모델 파일 경로
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"'{model_path}' 파일을 찾을 수 없습니다. main.py와 같은 폴더에 위치시켜주세요.")


# ============================================================
# 2. 공통 유틸 (블렌드셰이프, 행렬 → 오일러각)
# ============================================================

def get_blendshape_map(blendshapes):
    """MediaPipe blendshapes 리스트를 {name: score} 딕셔너리로 변환."""
    return {b.category_name: b.score for b in blendshapes}


def matrix_to_euler(matrix4: np.ndarray):
    """
    4x4 변환 행렬에서 pitch, yaw, roll(rad) 추출.
    MediaPipe FaceLandmarker 의 facial_transformation_matrixes 사용.
    """
    R = matrix4[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = 0.0

    return pitch, yaw, roll


def head_pose_matrix(result):
    """
    FaceLandmarkerResult → (pitch, yaw, roll) in rad
    facial_transformation_matrixes 가 없으면 (0,0,0) 리턴.
    """
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if not matrices or len(matrices) == 0:
        return 0.0, 0.0, 0.0

    # Python API에선 이미 np.ndarray 이므로 바로 변환
    m = np.array(matrices[0], dtype=np.float32).reshape(4, 4).T  # column-major 보정
    pitch, yaw, roll = matrix_to_euler(m)
    return pitch, yaw, roll


# ============================================================
# 3. 표정 관련 (미소, 끄덕임, 앞으로 숙이기)
# ============================================================

def smile(blendshapes):
    m = get_blendshape_map(blendshapes)
    left  = m.get("mouthSmileLeft", 0.0)
    right = m.get("mouthSmileRight", 0.0)

    squint_l = m.get("eyeSquintLeft", 0.0)
    squint_r = m.get("eyeSquintRight", 0.0)

    base_smile = (left + right) / 2.0
    duchenne_bonus = (squint_l + squint_r) / 2.0 * 0.5
    score = np.clip(base_smile + duchenne_bonus, 0.0, 1.0)

    return float(score)


DOWN_TH = -0.25
UP_TH   = -0.15
MAX_NOD_FRAMES = 240  # 대략 1초 정도

def detect_nod(pitch_list):
    """
    pitch_list: 최근 pitch 값들.
    '아래로 숙였다가 다시 올린' 패턴이 있으면 True.
    """
    if len(pitch_list) < 2:
        return False

    # 마지막 값이 충분히 아래로 숙인 상태인지
    if pitch_list[-1] > DOWN_TH:
        return False

    # 최근 MAX_NOD_FRAMES 안에서 UP_TH 이상으로 올라간 적이 있었는지
    recent = pitch_list[-MAX_NOD_FRAMES:]
    went_up = any(p > UP_TH for p in recent)

    return went_up


def detect_lean_forward(prev_nose_z, cur_nose_z, move_th=0.02):
    """
    코 z 좌표가 카메라 쪽으로 가까워지는지(몸을 앞으로 숙이는지) 감지.
    """
    if prev_nose_z is None:
        return False

    dz = prev_nose_z - cur_nose_z  # z 감소 → 카메라와 거리 감소
    return dz > move_th


# ============================================================
# 4. 아이컨택 관련: 홍채 + 머리 각도 기반
#   - iris_center_xy / compute_eye_offsets
#   - eye_contact (blendshape dev)
#   - gaze_compensation_score (머리-눈 보상)
#   - combined_eye_contact (최종)
# ============================================================

def iris_center_xy(landmarks, idxs):
    """지정된 iris 인덱스들의 (x, y) 평균 좌표."""
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)


def _eye_offsets_single(landmarks,
                        inner_idx, outer_idx,
                        up_idx, down_idx,
                        iris_xy: np.ndarray):
    """
    한쪽 눈에 대해:
      - inner(코쪽), outer(귀쪽), up(윗), down(아랫) + iris 위치
      - 수평/수직 상대 위치를 [-1, 1] 범위로 반환.
    """
    inner = np.array([landmarks[inner_idx].x, landmarks[inner_idx].y], dtype=np.float32)
    outer = np.array([landmarks[outer_idx].x, landmarks[outer_idx].y], dtype=np.float32)
    up    = np.array([landmarks[up_idx].x,    landmarks[up_idx].y],    dtype=np.float32)
    down  = np.array([landmarks[down_idx].x,  landmarks[down_idx].y],  dtype=np.float32)

    # --- 수평: inner(0) → outer(1) ---
    eye_vec = outer - inner
    eye_len = np.linalg.norm(eye_vec)
    if eye_len < 1e-6:
        t_h = 0.5
    else:
        t_h = float(np.dot(iris_xy - inner, eye_vec) / (eye_len ** 2))
    t_h = np.clip(t_h, 0.0, 1.0)
    offset_h = (t_h - 0.5) / 0.5  # 0~1 → -1~1

    # --- 수직: up(0) → down(1) ---
    v_vec = down - up
    v_len = np.linalg.norm(v_vec)
    if v_len < 1e-6:
        t_v = 0.5
    else:
        t_v = float(np.dot(iris_xy - up, v_vec) / (v_len ** 2))
    t_v = np.clip(t_v, 0.0, 1.0)
    offset_v = (t_v - 0.5) / 0.5  # 0~1 → -1~1

    return float(offset_h), float(offset_v)


def compute_eye_offsets(landmarks):
    """
    양 눈 홍채 위치로부터:
      - offset_h: 왼(-), 오른(+)  [-1,1]
      - offset_v: 위(-), 아래(+)  [-1,1]
    를 평균값으로 계산.
    """
    if landmarks is None or len(landmarks) < 478:
        return 0.0, 0.0

    # MediaPipe FaceMesh 인덱스 기준
    # 왼쪽 눈
    L_IN, L_OUT, L_UP, L_DOWN = 133, 33, 159, 145
    # 오른쪽 눈
    R_IN, R_OUT, R_UP, R_DOWN = 362, 263, 386, 374

    iris_left_idxs  = [473, 474, 475, 476, 477]
    iris_right_idxs = [468, 469, 470, 471, 472]

    iris_left_xy  = iris_center_xy(landmarks, iris_left_idxs)
    iris_right_xy = iris_center_xy(landmarks, iris_right_idxs)

    lh, lv = _eye_offsets_single(
        landmarks,
        inner_idx=L_IN, outer_idx=L_OUT,
        up_idx=L_UP, down_idx=L_DOWN,
        iris_xy=iris_left_xy,
    )
    rh, rv = _eye_offsets_single(
        landmarks,
        inner_idx=R_IN, outer_idx=R_OUT,
        up_idx=R_UP, down_idx=R_DOWN,
        iris_xy=iris_right_xy,
    )

    offset_h = (lh + rh) * 0.5
    offset_v = (lv + rv) * 0.5

    offset_h = float(np.clip(offset_h, -1.0, 1.0))
    offset_v = float(np.clip(offset_v, -1.0, 1.0))

    print(f"[EYE-IRIS] left=({lh:.2f},{lv:.2f}), right=({rh:.2f},{rv:.2f}), avg=({offset_h:.2f},{offset_v:.2f})")

    return offset_h, offset_v


def eye_contact(blendshapes,
                dev_center=0.25,
                max_dev=0.7,
                gamma=1.5):
    """
    블렌드셰이프 기반 눈 편차 점수.
    dev_center 부근이면 1에 가깝고, 멀어질수록 0에 가까워짐.
    (너무 빡세지 않게 gamma 살짝 줄임)
    """
    m = get_blendshape_map(blendshapes)

    lh = m.get("eyeLookOutLeft", 0.0) - m.get("eyeLookInLeft", 0.0)
    lv = m.get("eyeLookUpLeft",  0.0) - m.get("eyeLookDownLeft", 0.0)
    left_mag = np.sqrt(lh * lh + lv * lv)

    rh = m.get("eyeLookOutRight", 0.0) - m.get("eyeLookInRight", 0.0)
    rv = m.get("eyeLookUpRight",  0.0) - m.get("eyeLookDownRight", 0.0)
    right_mag = np.sqrt(rh * rh + rv * rv)

    dev_raw = (left_mag + right_mag) / 2.0
    dev = abs(dev_raw - dev_center)

    denom = max(max_dev, 1e-6)
    dev_norm = np.clip(dev / denom, 0.0, 1.0)

    dev_shaped = dev_norm ** gamma
    score = 1.0 - dev_shaped
    score = float(np.clip(score, 0.0, 1.0))

    print(f"[EYE-BLEND] dev_raw={dev_raw:.3f}, dev={dev:.3f}, score={score:.3f}")
    return score


def gaze_compensation_score(pitch, yaw,
                            offset_h, offset_v,
                            yaw_max_deg=35.0,
                            pitch_max_deg=25.0,
                            w_h=0.7, w_v=0.3,
                            err_cap=0.7):
    """
    머리 각도(pitch,yaw)와 홍채 offset의 '보상 관계'를 점수화.
    - 이상적인 관계: yaw_norm + offset_h ≈ 0, pitch_norm + offset_v ≈ 0
    - err_cap: 이 값 이상으로 틀어져도 더 이상 패널티를 키우지 않음 (완화용)
    """
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)

    yaw_n   = np.clip(yaw_deg   / yaw_max_deg,   -1.0, 1.0)
    pitch_n = np.clip(pitch_deg / pitch_max_deg, -1.0, 1.0)

    # "머리 + 눈 = 0" 이면 err = 0
    err_h = abs(yaw_n   + offset_h)
    err_v = abs(pitch_n + offset_v)

    # 너무 멀어지면 그냥 최대로 본다 (err_cap 이상은 전부 같게)
    err_h = np.clip(err_h, 0.0, err_cap) / max(err_cap, 1e-6)
    err_v = np.clip(err_v, 0.0, err_cap) / max(err_cap, 1e-6)

    score_h = 1.0 - err_h
    score_v = 1.0 - err_v

    score = w_h * score_h + w_v * score_v
    score = float(np.clip(score, 0.0, 1.0))

    print(
        f"[GAZE] yaw_deg={yaw_deg:.1f}, pitch_deg={pitch_deg:.1f}, "
        f"offset_h={offset_h:.2f}, offset_v={offset_v:.2f}, "
        f"score_h={score_h:.3f}, score_v={score_v:.3f}, score={score:.3f}"
    )
    return score



def combined_eye_contact(blendshapes, landmarks, pitch, yaw,
                         max_dev=0.7):
    """
    최종 아이컨택 점수:
      - base_eye: 블렌드셰이프 기반 눈 dev (부가 정보)
      - gaze_score: 머리-눈 보상 관계 (메인)
    """
    base_eye_score = eye_contact(blendshapes, max_dev=max_dev)

    if landmarks is None or len(landmarks) < 478:
        return float(base_eye_score)

    offset_h, offset_v = compute_eye_offsets(landmarks)
    gaze_score = gaze_compensation_score(
        pitch, yaw,
        offset_h, offset_v,
        yaw_max_deg=35.0,
        pitch_max_deg=25.0,
        w_h=0.7, w_v=0.3,
        err_cap=0.7,   # 위에서 추가한 파라미터
    )

    # 기본적으로는 gaze_score 위주로 가되,
    # base_eye_score는 살짝 보정 정도만 참여
    raw = 0.2 * base_eye_score + 0.8 * gaze_score

    # sqrt로 위로 당겨주기: 0.25→0.5, 0.36→0.6, 0.49→0.7 ...
    final = np.sqrt(np.clip(raw, 0.0, 1.0))
    final = float(final)

    print(f"[FINAL] base={base_eye_score:.3f}, gaze={gaze_score:.3f}, raw={raw:.3f}, final={final:.3f}")
    return final



# ============================================================
# 5. 기타 유틸
# ============================================================

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


# ============================================================
# 6. 비디오 처리 루틴
# ============================================================

def process_video(video_path: str):
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    detector = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_interval = int(fps / 15)  # 초당 15프레임
    if frame_interval <= 0:
        frame_interval = 1

    frame_id = 0

    # 통계 누적용 변수
    face_frame_count = 0
    smile_sum = 0.0
    eye_sum = 0.0
    eye_on_frames = 0

    pitch_history = []
    prev_nose_z = None
    nod_count = 0
    lean_forward_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % frame_interval != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect_for_video(mp_image, frame_id)

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            blend = detection_result.face_blendshapes[0]

            smile_score = smile(blend)
            pitch, yaw, roll = head_pose_matrix(detection_result)

            # ---- 수정된 아이컨택: combined_eye_contact만 사용 ----
            eye_contact_score = combined_eye_contact(
                blendshapes=blend,
                landmarks=face_landmarks,
                pitch=pitch,
                yaw=yaw,
                max_dev=0.7,
            )

            face_frame_count += 1
            smile_sum += smile_score
            eye_sum += eye_contact_score

            EYE_TH = 0.4  # 필요하면 여기서 0.4~0.6 사이로 튜닝
            if eye_contact_score >= EYE_TH:
                eye_on_frames += 1

            pitch_history.append(pitch)
            if detect_nod(pitch_history):
                nod_count += 1
                pitch_history = []

            # 코 z좌표로 몸 앞으로 숙이기 감지
            nose = face_landmarks[1]  # 코 근처 인덱스
            cur_nose_z = nose.z
            if detect_lean_forward(prev_nose_z, cur_nose_z):
                lean_forward_count += 1
            prev_nose_z = cur_nose_z

            print(
                f"[Frame {frame_id}] Smile: {smile_score:.3f}, "
                f"Eye: {eye_contact_score:.3f}, "
                f"Pitch: {pitch:.3f}, Yaw: {yaw:.3f}"
            )
        else:
            # 얼굴 미검출 프레임
            pass

    cap.release()

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
        print(f"\nZ-Normalized Scores:")
        print(f"  Smile:      {(avg_smile_0_100 - 31.09) / 16.50:.2f}")
        print(f"  EyeContact: {(avg_eye_0_100 - 50.21) / 6.17:.2f}")
        print(f"  Eye ratio:  {(eye_ratio - 0.8346) / 0.1990:.2f}")
        return avg_smile_0_100, avg_eye_0_100, eye_ratio
    else:
        print("No face detected in this video.")


# ============================================================
# 7. 테스트 실행
# ============================================================

def iter_video_files():
    root = Path(BASE_DIR) / "files"
    for p in sorted(root.glob("[1-5]/*.mp4")):
        yield str(p)

if __name__ == "__main__":
    results = []
    for video_file in iter_video_files():
        print(f"\n=== Processing {video_file} ===")
        try:
            r = process_video(video_file)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"[ERROR] {video_file}: {e}")

    if results:
        arr = np.array(results)  # shape (N, 3): [avg_smile, avg_eye, eye_ratio]
        mean_smile, mean_eye, mean_ratio = arr.mean(axis=0)
        std_smile, std_eye, std_ratio = arr.std(axis=0, ddof=0)
        print("\n=== ALL VIDEOS STATS ===")
        print(f"Mean Smile: {mean_smile:.2f} / 100 (std {std_smile:.2f})")
        print(f"Mean Eye:   {mean_eye:.2f} / 100 (std {std_eye:.2f})")
        print(f"Eye ratio:  {mean_ratio*100:.2f}% (std {std_ratio*100:.2f}%)")
    else:
        print("No valid video results.")

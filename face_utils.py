import cv2
import mediapipe as mp
import math
import json
import os

mp_face = mp.solutions.face_detection

# --- 1) Keypoint çıkarımı ---
def get_face_keypoints(image_bgr, model_selection=0, min_conf=0.5):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face.FaceDetection(model_selection=model_selection,
                               min_detection_confidence=min_conf) as fd:
        res = fd.process(rgb)
    if not res.detections:
        return None

    det = res.detections[0]
    kps = det.location_data.relative_keypoints  # [right_eye, left_eye, nose, mouth, right_ear, left_ear]

    rx, ry = kps[0].x, kps[0].y
    lx, ly = kps[1].x, kps[1].y
    nx, ny = kps[2].x, kps[2].y
    mx, my = kps[3].x, kps[3].y

    cx, cy = (lx + rx)/2.0, (ly + ry)/2.0
    D = math.hypot(lx - rx, ly - ry) + 1e-6

    return {'L': (lx, ly), 'R': (rx, ry), 'N': (nx, ny),
            'M': (mx, my), 'C': (cx, cy), 'D': D}


# --- 2) Metrikler ---
def metric_S(kp):
    """Burun - merkez ofseti"""
    if kp is None or kp['D'] <= 1e-6:
        return None
    xn, xc = kp['N'][0], kp['C'][0]
    return (xn - xc) / kp['D']

def metric_S_mouth(kp):
    """Dudak - merkez ofseti (isteğe bağlı)"""
    if kp is None or kp['D'] <= 1e-6:
        return None
    xm, xc = kp['M'][0], kp['C'][0]
    return (xm - xc) / kp['D']


# --- 3) Karar fonksiyonu ---
def face_direction_from_metrics(kp, T=0.12, mirror=False):
    S = metric_S(kp)
    if S is None:
        return "UNKNOWN", None

    if mirror:
        S = -S

    if S > T:
        return "RIGHT", S
    elif S < -T:
        return "LEFT", S
    else:
        return "CENTER", S


# --- 4) Ana akış ---
def main():
    print("Face Direction Estimator")
    print("-"*35)
    path = input("Görüntü dosya yolunu girin: ").strip('"').strip("'")

    if not os.path.exists(path):
        print(f"[Hata] Dosya bulunamadı: {path}")
        return

    img = cv2.imread(path)
    if img is None:
        print("[Hata] Görüntü okunamadı!")
        return

    kp = get_face_keypoints(img)
    if kp is None:
        print("[Sonuç] Yüz tespit edilemedi.")
        return

    label, S = face_direction_from_metrics(kp, T=0.12, mirror=False)

    print(f"[Sonuç] Yön: {label}")
    print(f"S (nose-center offset) = {S:.3f}")
    print(f"Göz mesafesi D = {kp['D']:.4f}")
    print(f"Merkez (C): {kp['C']}")
    print(f"Burun (N):  {kp['N']}")
    print(f"Dudak (M):  {kp['M']}")

    # Sonuçları JSON olarak da kaydedelim
    result = {
        "image": path,
        "label": label,
        "S": S,
        "C": kp['C'],
        "N": kp['N'],
        "M": kp['M'],
        "D": kp['D']
    }
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nSonuç 'result.json' dosyasına kaydedildi ✅")


if __name__ == "__main__":
    main()

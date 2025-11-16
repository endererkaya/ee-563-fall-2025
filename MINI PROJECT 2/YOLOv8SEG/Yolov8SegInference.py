import os
import sys
import time
import cv2
from ultralytics import YOLO

# ====================
# AYARLAR
# ====================
MODEL_WEIGHTS_PATH = r"runs\\tuned_freeze_20\\weights\\best.pt"
OUT_DIR = "pred_vis"
CONF_TH = 0.6
IOU_TH = 0.5
MAX_DET = 50


def run_inference(image_paths):
   
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_WEIGHTS_PATH)

    total_start = time.time()
    print("\n===== INFERENCE BAŞLADI =====")

    for img_path in image_paths:
        print(f"\n[INFO] İnference başlıyor → {img_path}")

        start = time.time()

        results = model.predict(
            source=img_path,
            imgsz=896,
            conf=CONF_TH,
            iou=IOU_TH,
            max_det=MAX_DET,
            save=False
        )[0]

        end = time.time()
        inference_time = end - start

        # Çıktıyı (mask + box) çiz
        render_bgr = results.plot()
        scale = 0.5  # 50%
        h, w = render_bgr.shape[:2]
        render_small = cv2.resize(render_bgr, (int(w*scale), int(h*scale)))
        # Kaydet
        save_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, render_bgr)

        print(f"[OK] Kaydedildi → {save_path}")
        print(f"[TIME] İnference süresi: {inference_time:.4f} sn")

        # ======================
        #    EKRANDA GÖSTER
        # ======================
        cv2.imshow(f"Prediction - {os.path.basename(img_path)}", render_small)
        print("[INFO] Görüntü kapatmak için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    total_end = time.time()
    total_time = total_end - total_start

    print("\n===== TÜM İŞLEM BİTTİ =====")
    print(f"Toplam süre: {total_time:.4f} sn")
    print(f"Çıktılar klasörü: {OUT_DIR}")


def main():
    """
    Çalıştırma:
    python balloon_path.py img1.jpg img2.jpg img3.jpg
    """
    if len(sys.argv) < 2:
        print("KULLANIM:\n  python balloon_path.py image1.jpg image2.jpg ...")
        sys.exit(1)

    image_paths = sys.argv[1:]
    run_inference(image_paths)


if __name__ == "__main__":
    main()

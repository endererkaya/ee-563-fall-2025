import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
from balloon_utils import get_model
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Resim bulunamadı: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    return img, transform(img)


def inference(image_path, score_thresh=0.5, mask_thresh=0.5):
    # Model yükle
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("maskrcnn_final_best.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Görseli yükle
    orig_img, img_tensor = load_image(image_path)
    img_tensor = img_tensor.to(DEVICE)

    
    start = time.time()
    # Inference
    with torch.no_grad():
        outputs = model([img_tensor])

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    inference_time = end - start   # saniye

    out = outputs[0]
    masks  = out["masks"]
    boxes  = out["boxes"]
    scores = out["scores"]
    labels = out["labels"]

    # Score threshold
    keep = scores >= score_thresh
    masks  = masks[keep]
    boxes  = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    results = []
    for i in range(len(masks)):
        mask_prob = masks[i, 0].cpu().numpy()
        binary_mask = mask_prob > mask_thresh
        area = int(binary_mask.sum())

        results.append({
            "box": boxes[i].cpu().numpy().tolist(),
            "score": float(scores[i]),
            "label": int(labels[i]),
            "area": area,
            "mask": binary_mask
        })

    return orig_img, results, inference_time


def visualize(orig_img, results):
    vis = orig_img.copy()

    for r in results:
        mask = r["mask"]

        # Maskeyi kırmızı renkle kapla (transparent overlay)
        vis[mask] = [255, 0, 0]

        # BBox çiz
        x1, y1, x2, y2 = map(int, r["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{r['score']:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python infer.py image_path")
        sys.exit(1)

    image_path = sys.argv[1]
    orig_img, results, inference_time = inference(image_path)

    print(f"\nSonuçlar ({image_path}):")
    print("-------------------------------")

    if len(results) == 0:
        print("Tespit edilen nesne yok.")
        sys.exit(0)

    for i, r in enumerate(results):
        print(f"Obj {i+1}:")
        print(f"  Score: {r['score']:.3f}")
        print(f"  Label: {r['label']}")
        print(f"  Area (pixels): {r['area']}")
        print(f"  Box: {r['box']}\n")

    image_area= orig_img.shape[1] * orig_img.shape[0]
    total_balloon_ratio = sum(r["area"] for r in results)/image_area
    print(image_area)
    print(f"Total ballon ratio: {total_balloon_ratio} pixels")
    print(f"Inference Time:{inference_time:.4f}")
    # Görsel oluştur
    vis_rgb = visualize(orig_img, results)
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    # Kaydetme
    out_path = image_path.rsplit(".", 1)[0] + "_infer.jpg"
    cv2.imwrite(out_path, vis_bgr)
    print(f"Kaydedildi -> {out_path}")

    scale = 0.5 # %50 küçültme, istersen 0.3 yap %30
    h, w = vis_bgr.shape[:2]
    vis_show = cv2.resize(vis_bgr, (int(w*scale), int(h*scale)))
    # Ekranda göster
    cv2.imshow("Inference Result", vis_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

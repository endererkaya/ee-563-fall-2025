import sys
from ultralytics import YOLO
import torch
import torch.nn.functional as F

## MASK ORIGINAL IMAGE BOYUTU DEĞIŞTIRILIP RESIZE EDILIP HESAPLANDIĞI ICIN TEKRAR ORIGINAL IMAGE BOYUTUNA GETIRILMESI GEREKILIYOR
## ORANIN SAĞLIKLI HESAPLANMASI ICIN. YOLONUN ÇALIŞMA ŞEKLI IMAGEI IMGSZI BOYUTUNA INDIRIP CNNI MODELI O ŞEKILDE MASKELERI ÇIKARMAK
def main():
    # Command path arg check
    if len(sys.argv) < 2:
        print("Kullanim: python countmaskpixel.py image_path")
        sys.exit(1)

    # User Image Path
    image_path = sys.argv[1]

    # YOLO modelini yükle
    MODEL_WEIGHTS_PATH = r"runs\freeze_20\\tuned_freeze_20\\weights\\best.pt"
    model = YOLO(MODEL_WEIGHTS_PATH)
    
    # Inference
    results = model.predict(
    source=image_path,
    imgsz=896,
    conf=0.6,
    iou=0.5,
    max_det=50,
    save=False
)[0]

    h, w = results.orig_shape
    image_pixel_counts = h * w

    # No mask
    if results.masks is None:
        print("No detection")
        sys.exit(0)

    # Maskleri al (N,H,W)
    masks = results.masks.data

    ## YOLO HER IMAGEI DEFAULT OLARAK 640,640A ÇEVIRIP IŞLER MASKEYI O ŞEKILDE ÇIKARIR
    ## O YUZDEN MASKENIN PIXEL HESAPLANAN PIKSEL SAYISI IMAGEIN PIKSEL SAYISI ILE AYNI OLMAZ
    ## O SEBEPLE MASKI TEKRAR ORIGINAL IMAGE ILE AYNI ORANA GETIRMEYE ÇALIŞIYORUZ
    ## BUNU DA INTERPOLATE ILE YAPTIK SONRA TEKRAR PIKSEL HESAPLADIK
    ## BÖYLECE ORANLAR DÜZGÜN ÇIKTI

    # (N, Hm, Wm) -> (N, 1, Hm, Wm)
    masks = masks.unsqueeze(1)

    # MASKI ORIGINAL GORUNTU BOYUTUNA INTERPOLATE ILE RESIZE ET
    masks_resized = F.interpolate(masks, size=(h, w), mode="nearest")

    # (N, H, W) şekline dön
    masks_resized = masks_resized.squeeze(1)

    # 0/1’e çevir (garanti)
    masks_bin = (masks_resized > 0.5).to(torch.int)

     # HER MASK ICIN GERCEK PIKSEL SAYISINI HESAPLA
    pixel_counts = masks_bin.sum(dim=(1, 2)).cpu().tolist()

    # Sınıflar
    classes = results.boxes.cls.cpu().tolist()
    names = model.names

    # Sonuçları yazdır
    for i, count in enumerate(pixel_counts):
        cls_id = int(classes[i])
        label = names[cls_id]
        print(f"Mask {i} | class: {label} | piksel: {int(count)}")

    total_pixel_counts = sum(pixel_counts)
    total_pixel_ratio  = (total_pixel_counts/image_pixel_counts) * 100
    print(f"Total Detected Balloon Pixel Counts: {total_pixel_counts}")
    print(f"Total Detected Ballon Area Ratio is : {total_pixel_ratio:.4f}")
    
if __name__ == "__main__":
    main()

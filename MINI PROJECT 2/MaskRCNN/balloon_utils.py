import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt


# --------- CONFIG ---------
TRAIN_DIR = "data/balloon/train"
VAL_DIR   = "data/balloon/val"
BATCH_SIZE = 2
NUM_EPOCHS = 5
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def freeze_first_n_layers(model, n: int):
    """
    Mask R-CNN backbone (ResNet50) içindeki parametreleri sırayla sayar.
    İlk n parametre grubunu freeze eder.
    """
    params = []

    # Backbone: model.backbone.body (ResNet50)
    for name, param in model.backbone.body.named_parameters():
        params.append((name, param))

    # İlk n parametre grubunu dondur
    for i, (name, param) in enumerate(params):
        if i < n:
            param.requires_grad = False
            
def freeze_backbone(model):
    """Tüm backbone (ResNet + FPN) donsun."""
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

def freeze_partial(model):
    """
    ResNet içinde sadece early layer'ları (layer1 & layer2) dondur.
    Layer3 ve layer4 + ROI heads eğitimde olur.
    """
    for name, param in model.backbone.body.named_parameters():
        if "layer1" in name or "layer2" in name:
            param.requires_grad = False

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic mode (slow but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
# --------- DATASET ---------
class BalloonDataset(Dataset):
    def __init__(self, root, transforms=None):

        self.root = root
        self.transforms = transforms

        annot_path = os.path.join(root, "via_region_data.json")
        with open(annot_path) as f:
            self.annotations = json.load(f)

        self.image_infos = list(self.annotations.values())

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        info = self.image_infos[idx]
        filename = info["filename"]

        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        regions = info["regions"]
        if isinstance(regions, dict):
            regions = list(regions.values())

        boxes = []
        masks = []

        for r in regions:
            shape = r["shape_attributes"]
            xs = shape["all_points_x"]
            ys = shape["all_points_y"]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            boxes.append([xmin, ymin, xmax, ymax])

            mask = np.zeros((h, w), dtype=np.uint8)
            poly = np.array([list(zip(xs, ys))], dtype=np.int32)
            cv2.fillPoly(mask, poly, 1)
            masks.append(mask)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.stack(masks), dtype=torch.uint8)

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# --------- TRANSFORMS ---------
def get_transform(train):
    return T.ToTensor()


# --------- COLLATE FN ---------
def collate_fn(batch):
    return tuple(zip(*batch))


# --------- MODEL ---------
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model

import numpy as np
from collections import defaultdict

def compute_ap(rec, prec):
    """
    COCO'ya benzer şekilde AP (area under PR curve) hesapluyor.
    rec, prec: 1D numpy array, artan recall dizileri.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # precision'ı monotonik azalmayan yap
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # recall değiştiği noktalarda integral al
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def mask_iou_np(pred_mask, gt_masks):
    """
    pred_mask: (H, W) bool
    gt_masks: (N, H, W) bool
    """
    if gt_masks.size == 0:
        return np.zeros((0,), dtype=np.float32)

    pred_flat = pred_mask.reshape(-1)
    gt_flat = gt_masks.reshape(gt_masks.shape[0], -1)  # (N, H*W)

    inter = (gt_flat & pred_flat).sum(axis=1).astype(np.float32)
    union = (gt_flat | pred_flat).sum(axis=1).astype(np.float32) + 1e-6

    return inter / union

def evaluate_mask_map(model, data_loader, device, iou_thresholds=None, score_thresh=0.05):
    """
    Tek sınıflı (balloon) MASK mAP@50 ve mAP@50:95 hesaplar.
    - model: Mask R-CNN
    - data_loader: val_loader
    - device: DEVICE
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95

    model.eval()
    gt_masks_per_image = defaultdict(list)
    all_predictions = []  # (image_id, score, pred_mask)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())

                # GT maskleri al (N, H, W)
                gt_masks = target["masks"].cpu().numpy().astype(bool)
                gt_masks_per_image[image_id].extend(gt_masks)

                # Pred maskler ve skorlar
                pred_masks = output["masks"].cpu().numpy()  # (N, 1, H, W)
                scores = output["scores"].cpu().numpy()

                for pm, sc in zip(pred_masks, scores):
                    if sc < score_thresh:
                        continue
                    pm_bin = (pm[0] > 0.5).astype(bool)
                    all_predictions.append({
                        "image_id": image_id,
                        "score": float(sc),
                        "mask": pm_bin
                    })

    # Toplam GT sayısı
    total_gt = sum(len(v) for v in gt_masks_per_image.values())
    if total_gt == 0:
        return 0.0, 0.0

    # Skora göre sırala (yüksek -> düşük)
    all_predictions.sort(key=lambda x: x["score"], reverse=True)

    aps = []

    for t in iou_thresholds:
        tp = []
        fp = []

        # Her GT maske sadece 1 kere eşleşebilsin
        matched = {
            img_id: np.zeros(len(masks), dtype=bool)
            for img_id, masks in gt_masks_per_image.items()
        }

        for pred in all_predictions:
            img_id = pred["image_id"]
            pred_mask = pred["mask"]

            gt_list = gt_masks_per_image[img_id]
            if len(gt_list) == 0:
                fp.append(1)
                tp.append(0)
                continue

            gt_masks = np.stack(gt_list, axis=0)  # (G, H, W)
            ious = mask_iou_np(pred_mask, gt_masks)
            max_idx = np.argmax(ious)
            max_iou = ious[max_idx]

            if max_iou >= t and not matched[img_id][max_idx]:
                tp.append(1)
                fp.append(0)
                matched[img_id][max_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp = np.array(tp)
        fp = np.array(fp)

        if len(tp) == 0:
            aps.append(0.0)
            continue

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / (total_gt + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    map_50_95 = float(np.mean(aps))
    map_50 = float(aps[0])  # IoU=0.5 için

    return map_50, map_50_95

# --------- TRAIN LOOP ---------
def run_training_for_hparams(
    lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    freeze_n_layers: int,
    step_size: int,
    gamma: float,
    trial=None,
):

    train_dataset = BalloonDataset(TRAIN_DIR, get_transform(train=True))
    val_dataset   = BalloonDataset(VAL_DIR,   get_transform(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = get_model(num_classes=2)
    model.to(DEVICE)

    # ---- Integer layer freeze burada uygulanıyor ----
    print(f"  >> Freezing first {freeze_n_layers} layers in backbone")
    freeze_first_n_layers(model, freeze_n_layers)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=step_size,
    gamma=gamma,
    )

    eval_epochs = {1} | set(range(2, num_epochs + 1, 2))
    best_map50_95 = 0.0
    best_state_dict = None
    map_history = []   # her eval epoch'taki mAP@50:95 buraya girecek
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"[Trial] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        loss_history.append((epoch + 1, avg_loss))

        if (epoch + 1) in eval_epochs:
            mask_map50, mask_map50_95 = evaluate_mask_map(model, val_loader, DEVICE)
            print(f"  -> Val mAP@50: {mask_map50:.3f}, mAP@50:95: {mask_map50_95:.3f}")
            map_history.append((epoch + 1, mask_map50_95))

            if mask_map50_95 > best_map50_95:
                best_map50_95 = mask_map50_95
                best_state_dict = model.state_dict() 

    return best_map50_95, best_state_dict, map_history, loss_history


# --------- VISUALIZATION ---------
def show_detection(model, val_loader):
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(val_loader))
        img = images[0].to(DEVICE)
        pred = model([img])[0]

    # Tensor → numpy
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # OpenCV uyumlu format (uint8 + writable)
    overlay = (img_np * 255).astype(np.uint8).copy()

    # Tahminler
    scores = pred["scores"].cpu().numpy()
    masks  = pred["masks"].cpu().numpy()

    # Threshold
    keep = scores > 0.3
    masks = masks[keep]

    if len(masks) == 0:
        print("Hiç maske bulunamadı.")
        return

    # Her maske için işlemler
    for m in masks:
        mask = (m[0] > 0.5).astype(np.uint8)   # binary 0/1 maske

        # ---------- 1) Maske içini %50 siyah yap ----------
        overlay[mask == 1] = (overlay[mask == 1] * 0.5).astype(np.uint8)
        # Bu balonu yarı saydam siyahla koyulaştırır.

        # ---------- 2) Kontur çiz ----------
        mask_255 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Kalın siyah çizgi (kalınlık=4 → istersen değiştir)
        cv2.drawContours(overlay, contours, -1, (0,0,0), 8)

    # Göster
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("50% Siyah Maske + Kalın Siyah Kontur")
    plt.show()

if __name__ == "__main__":
    print("Device:", DEVICE)
    train()

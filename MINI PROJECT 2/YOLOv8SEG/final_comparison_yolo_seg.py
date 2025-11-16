from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================== PATHLER ==================
SEG_YAML_PATH = r"C:\Users\endererkaya\OneDrive\Desktop\Mini Project 2\balloon_YOLO_SEG_OUTPUT\balloon_seg.yaml"
BASE_MODEL    = r"C:\Users\endererkaya\OneDrive\Desktop\Mini Project 2\runs\base_weights\best.pt"

PROJECT = "runs/freeze_20"
EPOCHS  = 30
IMAGE_SIZE = 640

# Optuna tuned params
BEST_LR0          = 0.010196974520165896
BEST_LRF          = 0.016662784594747543
BEST_MOMENTUM     = 0.8885383128386182
BEST_WEIGHT_DECAY = 5.0191768410420506e-05
BEST_OPTIMIZER    = "SGD"

FREEZE_LIST = [10, 20]


# ================== TRAIN FUNCTIONS ==================
def train_tuned(freeze_val):
    print(f"\n=== Tuned Training (freeze={freeze_val}) ===")
    model = YOLO(BASE_MODEL)

    model.train(
        data=SEG_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        lr0=BEST_LR0,
        lrf=BEST_LRF,
        momentum=BEST_MOMENTUM,
        weight_decay=BEST_WEIGHT_DECAY,
        optimizer=BEST_OPTIMIZER,

        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        scale=0.5,

        freeze=freeze_val,
        batch=4,
        workers=0,
        device=0,
        project=PROJECT,
        name=f"tuned_freeze_{freeze_val}",
        exist_ok=True,
    )


def train_default(freeze_val):
    print(f"\n=== DEFAULT Training (freeze={freeze_val}) ===")
    model = YOLO(BASE_MODEL)

    model.train(
        data=SEG_YAML_PATH,
        epochs=EPOCHS,
        project=PROJECT,
        name=f"default_freeze_{freeze_val}",
        workers=0,
        device=0,
        exist_ok=True,
    )


# ================== EVALUATION ==================
def evaluate_all():
    results = {}

    print("\n=== Modeller Değerlendiriliyor ===")

    for mode in ["tuned", "default"]:
        for f in FREEZE_LIST:
            weights_path = fr"{PROJECT}\{mode}_freeze_{f}\weights\best.pt"

            model = YOLO(weights_path)
            metrics = model.val(
                data=SEG_YAML_PATH,
                imgsz=IMAGE_SIZE,
                batch=2,
                workers=0,
                verbose=False,
            )

            mAP = float(metrics.seg.map)
            results[(mode, f)] = mAP

            print(f"{mode:<7s} freeze={f:<3d} → mAP50-95(seg) = {mAP:.4f}")

    return results


# ================== PLOT ==================
def plot_curves():
    plt.figure(figsize=(20, 5))  # geniş yatay çizim

    # ===================== 1) RECALL =====================
    plt.subplot(1, 3, 1)
    for mode in ["tuned", "default"]:
        for f in FREEZE_LIST:
            csv_path = fr"{PROJECT}\{mode}_freeze_{f}\results.csv"
            if not os.path.exists(csv_path):
                print(f"⚠ CSV bulunamadı, atlandı: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            plt.plot(df["epoch"], df["metrics/recall(M)"],
                     marker="o", linewidth=2,
                     label=f"{mode}_freeze={f}")

    plt.title("Recall vs Freeze", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)


    # ===================== 2) mAP50 =====================
    plt.subplot(1, 3, 2)
    for mode in ["tuned", "default"]:
        for f in FREEZE_LIST:
            csv_path = fr"{PROJECT}\{mode}_freeze_{f}\results.csv"
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            plt.plot(df["epoch"], df["metrics/mAP50(M)"],
                     marker="o", linewidth=2,
                     label=f"{mode}_freeze={f}")

    plt.title("mAP50 vs Freeze", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)


    # ===================== 3) mAP50-95 =====================
    plt.subplot(1, 3, 3)
    for mode in ["tuned", "default"]:
        for f in FREEZE_LIST:
            csv_path = fr"{PROJECT}\{mode}_freeze_{f}\results.csv"
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            plt.plot(df["epoch"], df["metrics/mAP50-95(M)"],
                     marker="o", linewidth=2,
                     label=f"{mode}_freeze={f}")

    plt.title("mAP50-95 vs Freeze", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50-95")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)

    # ======================================================
    plt.suptitle("Freeze Sweep Performance Comparison", fontsize=15)
    plt.tight_layout()
    plt.show()


# ================== MAIN ==================
def main():
    # Train all models
    for f in FREEZE_LIST:
        train_tuned(f)
        train_default(f)

    # Evaluate
    results = evaluate_all()

    print("\n========== EN İYİ MODELLER ==========")
    best_key = max(results, key=results.get)
    print(f"BEST → {best_key} = {results[best_key]:.4f}")

    # Plots
    plot_curves()


if __name__ == "__main__":
    main()

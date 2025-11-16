import optuna
from ultralytics import YOLO

SEG_YAML_PATH = r"C:\Users\endererkaya\OneDrive\Desktop\Mini Project 2\balloon_YOLO_SEG_OUTPUT\balloon_seg.yaml"
PROJECT = "runs/optuna_seg_balloon"
EPOCHS = 15  


def objective(trial: optuna.Trial) -> float:
    import torch
    torch.cuda.empty_cache()

    lr0 = trial.suggest_float("lr0", 3e-5, 3e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 0.5)
    momentum = trial.suggest_float("momentum", 0.85, 0.97)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])

    image_size = trial.suggest_categorical("image_size", [640, 768, 896])

    model = YOLO(r"runs/segment/yolov8s_seg_final_train/weights/best.pt")

    run_name = f"trial_{trial.number}"

    # TRAIN
    model.train(
        data=SEG_YAML_PATH,
        epochs=EPOCHS,
        imgsz=image_size,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        batch=4,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        scale=0.5,
        workers=0,
        device=0,       # tek GPU
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    # VAL
    metrics = model.val(
        data=SEG_YAML_PATH,
        imgsz=image_size,
        verbose=False,
    )

    score = metrics.seg.map  # mAP50-95 (segmentation)
    return score


def main():
    study = optuna.create_study(
        study_name="yolov8_seg_balloon_optim_focus",
        direction="maximize",
    )
    study.optimize(objective, n_trials=200)

    print("En iyi hiperparametreler:")
    print(study.best_params)   
    print("En iyi mAP50-95 (seg):", study.best_value)


if __name__ == "__main__":
    main()

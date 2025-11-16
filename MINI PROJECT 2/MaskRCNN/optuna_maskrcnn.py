import optuna
import torch
from torch.utils.data import DataLoader

from balloon_utils import (
    BalloonDataset,
    get_transform,
    collate_fn,
    get_model,
    evaluate_mask_map,
    run_training_for_hparams,
    TRAIN_DIR,
    VAL_DIR,
    DEVICE,
    set_seed,
)

# ---------------- CONFIG ----------------
MAX_EPOCHS_PER_TRIAL = 8   # her trial için epoch sayısı
NUM_CLASSES = 2            # background + balloon


def build_dataloaders(batch_size: int):
    """Train/Val DataLoader oluşturur."""
    train_dataset = BalloonDataset(TRAIN_DIR, transforms=get_transform(train=True))
    val_dataset   = BalloonDataset(VAL_DIR,   transforms=get_transform(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,          # eval için 1 yeterli
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def train_one_trial(trial, train_loader, val_loader):
    """
    Bir Optuna trial'ı için:
      - hyperparam seç
      - model kur
      - MAX_EPOCHS_PER_TRIAL kadar train et
      - sonunda val MASK mAP@50:95 döndür
    """

    # ---------- 1) Hyperparameter sampling ----------
    # learning rate (log scale)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # momentum (0.8 - 0.99)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)

    # weight decay
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # batch size (1 veya 2) – VRAM'ine göre sınırı artırabilirsin
    batch_size = trial.suggest_categorical("batch_size", [1, 2])

    # NOT: batch size'ı değiştirdiğimiz için dataloader'ı yeniden kurmamız gerek
    train_loader, val_loader = build_dataloaders(batch_size)

    # ---------- 2) Model & optimizer ----------
    model = get_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # LR scheduler de deneyebiliriz (sabit de tutabilirsin)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, MAX_EPOCHS_PER_TRIAL // 2),
        gamma=0.1,
    )

    # ---------- 3) Training loop (kısa) ----------
    for epoch in range(MAX_EPOCHS_PER_TRIAL):
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

        avg_loss = total_loss / len(train_loader)

        # her epoch sonunda lr güncelle
        scheduler.step()

        # trial içi log (isteğe bağlı)
        trial.report(avg_loss, step=epoch)

        # Optuna pruning (ör: çok kötü giden trial'ı erken kesmek için)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ---------- 4) Epochler bitti → val MASK mAP hesapla ----------
    mask_map50, mask_map50_95 = evaluate_mask_map(model, val_loader, DEVICE)

    # İstiyorsan trial user attrs olarak kaydedebilirsin
    trial.set_user_attr("mask_map50", mask_map50)
    trial.set_user_attr("mask_map50_95", mask_map50_95)

    # Objective: mAP@50:95'i maximize etmek istiyoruz
    return mask_map50_95


def objective(trial):
    # --------------------
    # Seed (trial başında)
    # --------------------
    seed_value = 42
    set_seed(seed_value)

    # Hyperparam sampling
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2])
    freeze_n_layers = trial.suggest_int("freeze_n_layers", 0, 15)


    num_epochs = 10

    print("\n" + "=" * 60)
    print(f"Trial {trial.number} Hyperparams:")
    print(f" seed={seed_value}")
    print(f" lr={lr:.6f}")
    print(f" momentum={momentum:.3f}")
    print(f" weight_decay={weight_decay:.6f}")
    print(f" batch_size={batch_size}")
    print(f" freeze_mode={freeze_n_layers}")
    print(f" epochs={num_epochs}")

    best_map50_95 = run_training_for_hparams(
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=num_epochs,
        freeze_n_layers=freeze_n_layers,
        trial=trial,
    )

    trial.set_user_attr("best_mask_map50_95", best_map50_95)
    return best_map50_95

if __name__ == "__main__":
    # Kaç trial
    N_TRIALS = 100

    study = optuna.create_study(
        direction="maximize",
        study_name="maskrcnn_balloon_mask_map",
        sampler=optuna.samplers.TPESampler(seed=42),   # <-- SEED EKLENDİ
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
        ),
    )

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    print("\nBest trial:")
    best_trial = study.best_trial
    print("  value (best mAP@50:95):", best_trial.value)
    print("  params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    # Doğru user_attr
    print("  best_mask_map50_95:", best_trial.user_attrs.get("best_mask_map50_95"))

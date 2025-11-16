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

# En iyi bulduğun değerleri buraya yaz
BEST_LR = 3.6895280487816236e-04
BEST_MOMENTUM = 0.8753744911670475
BEST_WEIGHT_DECAY = 1.0009363719729304e-05
BEST_BATCH_SIZE = 1
BEST_FREEZE_N_LAYERS = 1

MAX_EPOCHS_PER_TRIAL = 10   # kaç epoch denemek istiyorsan
DEVICE = "cuda"         # background + balloon

def objective_scheduler(trial):
    seed_value = 42
    set_seed(seed_value)

    num_epochs = MAX_EPOCHS_PER_TRIAL  # örn: 8 veya 10

    # Sadece StepLR parametreleri
    step_size = trial.suggest_int("step_size", 2, num_epochs)
    gamma = trial.suggest_float("gamma", 0.3, 0.7)

    best_map50_95 = run_training_for_hparams(
        lr=BEST_LR,
        momentum=BEST_MOMENTUM,
        weight_decay=BEST_WEIGHT_DECAY,
        batch_size=BEST_BATCH_SIZE,
        freeze_n_layers=BEST_FREEZE_N_LAYERS,
        num_epochs=num_epochs,
        step_size=step_size,
        gamma=gamma,
        trial=trial,
    )

    trial.set_user_attr("best_mask_map50_95", best_map50_95)
    return best_map50_95

if __name__ == "__main__":
    N_TRIALS = 15  

    study = optuna.create_study(
        direction="maximize",
        study_name="stepLR_only_search",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=1,
        ),
    )

    study.optimize(objective_scheduler, n_trials=N_TRIALS, gc_after_trial=True)

    best = study.best_trial
    print("\nBest StepLR trial:")
    print(" value:", best.value)
    print(" params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(" mAP:", best.user_attrs.get("best_mask_map50_95"))
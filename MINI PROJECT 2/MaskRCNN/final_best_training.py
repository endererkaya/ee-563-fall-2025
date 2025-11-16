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
seed_value = 42
set_seed(seed_value)
num_epochs = 30
DEVICE = "cuda" 

# BEST OPTUNA VALUES
BEST_LR = 3.6895280487816236e-04
BEST_MOMENTUM = 0.8753744911670475
BEST_WEIGHT_DECAY = 1.0009363719729304e-05
BEST_BATCH_SIZE = 1
BEST_FREEZE_N_LAYERS = 1
step_size = 5
gamma = 0.5112666241284302
nesterov= True

best_final, best_weights, map_history, loss_history = run_training_for_hparams(
        lr=BEST_LR,
        momentum=BEST_MOMENTUM,
        weight_decay=BEST_WEIGHT_DECAY,
        batch_size=BEST_BATCH_SIZE,
        freeze_n_layers=BEST_FREEZE_N_LAYERS,
        num_epochs=num_epochs,
        step_size=step_size,
        gamma=gamma,
        trial=None,
    )

print("Final mAP@50:95:", best_final)
torch.save(best_weights, "maskrcnn_final_best.pth")

import matplotlib.pyplot as plt

# Loss verileri
loss_epochs = [e for e, l in loss_history]
loss_values = [l for e, l in loss_history]

# mAP verileri
map_epochs = [e for e, m in map_history]
map_values = [m for e, m in map_history]

plt.figure(figsize=(12, 4))

# ---- SOL: LOSS ----
plt.subplot(1, 2, 1)
plt.plot(loss_epochs, loss_values, marker="o", markersize=10)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# ---- SAĞ: mAP ----
plt.subplot(1, 2, 2)
plt.plot(map_epochs, map_values, marker=">", markersize=10)
plt.title("Validation mAP@50:95")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.grid(True)

plt.tight_layout()
plt.show()
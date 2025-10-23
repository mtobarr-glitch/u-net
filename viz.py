from typing import List
import os
import matplotlib.pyplot as plt


def plot_curves_png(path: str, train_losses: List[float], val_losses: List[float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

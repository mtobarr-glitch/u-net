from typing import Dict, Tuple
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .viz import plot_curves_png


class EarlyStopper:
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best = float('inf')
        self.count = 0
        self.best_state = None

    def step(self, val_loss: float, model) -> bool:
        if val_loss < self.best:
            self.best = val_loss
            self.count = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.count += 1
            return self.count > self.patience


def save_hparams(path: str, hparams: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(hparams, f)


def train_model(model, optimizer, loss_fn, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                epochs: int, save_dir: str, early_stop_patience: int = 20) -> Tuple[str, str, list, list]:
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best.pt')
    last_path = os.path.join(save_dir, 'last.pt')

    early = EarlyStopper(patience=early_stop_patience)
    train_losses, val_losses = [], []

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f'train {epoch+1}/{epochs}'):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_losses.append(running / max(len(train_loader), 1))

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f'val {epoch+1}/{epochs}'):
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                vloss += loss.item()
        vloss /= max(len(val_loader), 1)
        val_losses.append(vloss)

        # early stopping
        stop = early.step(vloss, model)
        if early.best_state is not None:
            torch.save(early.best_state, best_path)
        torch.save(model.state_dict(), last_path)
        if stop:
            break

    plot_curves_png(os.path.join(save_dir, 'curves.png'), train_losses, val_losses)
    return best_path, last_path, train_losses, val_losses

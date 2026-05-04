# Training loop for baseline and multitask models

import os
import torch

from training.losses import lighting_loss


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, config,
                 is_multitask=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.is_multitask = is_multitask
        self.device = config.device
        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_light = 0.0
        total_mat = 0.0
        n_batches = 0

        for i, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            target_light = batch['lighting'].to(self.device)
            target_mat = batch['material_params'].to(self.device)

            self.optimizer.zero_grad()

            if self.is_multitask:
                pred_light, pred_mat = self.model(images)
                loss, loss_light, loss_mat = self.criterion(
                    pred_light, pred_mat, target_light, target_mat
                )
            else:
                pred_light = self.model(images)
                loss_light = lighting_loss(
                    pred_light, target_light,
                    eps=self.config.log_eps,
                    mse_weight=getattr(self.config, "lighting_mse_weight", 1.0),
                    l1_weight=getattr(self.config, "lighting_l1_weight", 0.5),
                    ssim_weight=getattr(self.config, "lighting_ssim_weight", 0.2),
                )
                loss_mat = torch.tensor(0.0)
                loss = loss_light

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_light += loss_light.item()
            total_mat += loss_mat.item()
            n_batches += 1

            if (i + 1) % self.config.log_interval == 0:
                print(f"  batch {i+1}/{len(dataloader)} | "
                      f"loss={loss.item():.4f} light={loss_light.item():.4f} "
                      f"mat={loss_mat.item():.4f}")

        return {
            'loss': total_loss / n_batches,
            'loss_lighting': total_light / n_batches,
            'loss_material': total_mat / n_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_light = 0.0
        total_mat = 0.0
        sum_abs_err = 0.0
        total_samples = 0
        n_batches = 0

        for batch in dataloader:
            images = batch['image'].to(self.device)
            target_light = batch['lighting'].to(self.device)
            target_mat = batch['material_params'].to(self.device)

            if self.is_multitask:
                pred_light, pred_mat = self.model(images)
                loss, loss_light, loss_mat = self.criterion(
                    pred_light, pred_mat, target_light, target_mat
                )
                sum_abs_err += (pred_mat - target_mat).abs().sum().item()
            else:
                pred_light = self.model(images)
                loss_light = lighting_loss(
                    pred_light, target_light,
                    eps=self.config.log_eps,
                    mse_weight=getattr(self.config, "lighting_mse_weight", 1.0),
                    l1_weight=getattr(self.config, "lighting_l1_weight", 0.5),
                    ssim_weight=getattr(self.config, "lighting_ssim_weight", 0.2),
                )
                loss_mat = torch.tensor(0.0)
                loss = loss_light

            total_loss += loss.item()
            total_light += loss_light.item()
            total_mat += loss_mat.item()
            total_samples += images.size(0)
            n_batches += 1

        metrics = {
            'loss': total_loss / n_batches,
            'loss_lighting': total_light / n_batches,
            'loss_material': total_mat / n_batches,
        }
        if self.is_multitask and total_samples > 0:
            n_params = self.config.num_material_params
            metrics['material_mae'] = sum_abs_err / (total_samples * n_params)

        return metrics

    def fit(self, train_loader, val_loader):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_lighting": [],
            "val_loss_lighting": [],
            "train_loss_material": [],
            "val_loss_material": [],
        }

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"Epoch {epoch}/{self.config.num_epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            history["train_loss"].append(train_metrics['loss'])
            history["val_loss"].append(val_metrics['loss'])
            history["train_loss_lighting"].append(train_metrics['loss_lighting'])
            history["val_loss_lighting"].append(val_metrics['loss_lighting'])
            history["train_loss_material"].append(train_metrics['loss_material'])
            history["val_loss_material"].append(val_metrics['loss_material'])

            # Print epoch summary
            print(f"  train | loss={train_metrics['loss']:.4f} "
                  f"light={train_metrics['loss_lighting']:.4f} "
                  f"mat={train_metrics['loss_material']:.4f}")
            val_line = (f"  val   | loss={val_metrics['loss']:.4f} "
                        f"light={val_metrics['loss_lighting']:.4f} "
                        f"mat={val_metrics['loss_material']:.4f}")
            if 'material_mae' in val_metrics:
                val_line += f" mat_mae={val_metrics['material_mae']:.4f}"
            print(val_line)

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, is_best=True)
                print(f"  -> new best model (val_loss={self.best_val_loss:.4f})")

            print()

        return history

    def _save_checkpoint(self, epoch, is_best=False):
        os.makedirs(self.config.save_dir, exist_ok=True)
        if self.is_multitask:
            tag = ("multitask_film"
                   if getattr(self.config, 'multitask_arch', 'tiled') == 'film'
                   else "multitask")
        else:
            tag = "baseline"
        filename = f"{self.config.experiment_name}_{tag}_best.pt" if is_best \
            else f"{self.config.experiment_name}_{tag}_epoch{epoch}.pt"
        path = os.path.join(self.config.save_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
        }, path)

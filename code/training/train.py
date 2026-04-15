# Training loop for baseline and multitask models

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


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
            target_sh = batch['lighting'].to(self.device)
            target_mat = batch['material'].to(self.device)

            self.optimizer.zero_grad()

            if self.is_multitask:
                pred_light, pred_mat = self.model(images)
                loss, loss_light, loss_mat = self.criterion(
                    pred_light, pred_mat, target_sh, target_mat
                )
            else:
                pred_light = self.model(images)
                loss_light = torch.nn.functional.mse_loss(pred_light, target_sh)
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
        correct = 0
        total_samples = 0
        n_batches = 0

        for batch in dataloader:
            images = batch['image'].to(self.device)
            target_sh = batch['lighting'].to(self.device)
            target_mat = batch['material'].to(self.device)

            if self.is_multitask:
                pred_light, pred_mat = self.model(images)
                loss, loss_light, loss_mat = self.criterion(
                    pred_light, pred_mat, target_sh, target_mat
                )
                correct += (pred_mat.argmax(1) == target_mat).sum().item()
            else:
                pred_light = self.model(images)
                loss_light = torch.nn.functional.mse_loss(pred_light, target_sh)
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
            metrics['material_accuracy'] = correct / total_samples

        return metrics

    def _get_curriculum_phase(self, epoch):
        """Return the mix dict for the given epoch, or None if no curriculum."""
        curriculum = getattr(self.config, 'curriculum', None)
        if not curriculum:
            return None
        for last_epoch, mix in curriculum:
            if epoch <= last_epoch:
                return mix
        # Past the last defined phase — use the final one
        return curriculum[-1][1]

    def _build_train_loader(self, train_ds, mix):
        """Build a DataLoader with a WeightedRandomSampler for the given mix."""
        # Compute per-sample weights based on source tag and desired mix
        source_tags = train_ds.source_tags[train_ds.indices]
        weights = np.zeros(len(train_ds), dtype=np.float64)
        for tag, w in mix.items():
            mask = source_tags == tag
            n_tag = mask.sum()
            if n_tag > 0 and w > 0:
                weights[mask] = w / n_tag

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=int((weights > 0).sum()),
            replacement=True,
        )
        return DataLoader(
            train_ds, batch_size=self.config.batch_size, sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=(str(self.device) == "cuda"),
        )

    def fit(self, train_ds_or_loader, val_loader):
        """Train the model.

        Args:
            train_ds_or_loader: either a DataLoader (no curriculum) or a
                SphereDataset (curriculum phases will rebuild the loader).
            val_loader: validation DataLoader (always uses the full mix).
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_lighting": [],
            "val_loss_lighting": [],
            "train_loss_material": [],
            "val_loss_material": [],
        }

        # Determine if we're using curriculum learning
        use_curriculum = (hasattr(train_ds_or_loader, 'source_tags')
                          and getattr(self.config, 'curriculum', None))
        if use_curriculum:
            train_ds = train_ds_or_loader
        else:
            train_loader = train_ds_or_loader

        current_mix = None

        for epoch in range(1, self.config.num_epochs + 1):
            # Rebuild train loader when curriculum phase changes
            if use_curriculum:
                mix = self._get_curriculum_phase(epoch)
                if mix != current_mix:
                    current_mix = mix
                    train_loader = self._build_train_loader(train_ds, mix)
                    active = {k: v for k, v in mix.items() if v > 0}
                    print(f"  [curriculum] phase mix: {active}")
                    print(f"  [curriculum] loader size: {len(train_loader)} batches")

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
            if 'material_accuracy' in val_metrics:
                val_line += f" acc={val_metrics['material_accuracy']:.4f}"
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
        tag = "multitask" if self.is_multitask else "baseline"
        filename = f"{self.config.experiment_name}_{tag}_best.pt" if is_best \
            else f"{self.config.experiment_name}_{tag}_epoch{epoch}.pt"
        path = os.path.join(self.config.save_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
        }, path)

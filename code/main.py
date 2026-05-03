# Entry point for training baseline and multitask models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T

from config import config
from dataset.data import SphereDataset
from models.baseline_model import BaselineLightingNet
from models.multitask_model import MaterialAwareLightingNet
from training.losses import MultiTaskLoss
from training.train import Trainer
from evaluation.evaluator import evaluate_model, compute_all_metrics, compare_models
from utils.visualizer import (
    plot_envmap_comparison,
    plot_per_material_comparison,
    plot_training_curves,
)


def make_dataloaders(config, transform):
    train_ds = SphereDataset('train', config, transform=transform)
    val_ds = SphereDataset('val', config, transform=transform)
    test_ds = SphereDataset('test', config, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(str(config.device) == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(str(config.device) == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(str(config.device) == "cuda"),
    )
    print(f"Dataset splits: train={len(train_ds)}, val={len(val_ds)}, "
          f"test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def train_model(config, is_multitask, train_loader, val_loader):
    tag = "multitask" if is_multitask else "baseline"
    print(f"\n{'='*60}")
    print(f"Training {tag} model ({config.backbone})")
    print(f"{'='*60}\n")

    if is_multitask:
        model = MaterialAwareLightingNet(config).to(config.device)
        criterion = MultiTaskLoss(config)
    else:
        model = BaselineLightingNet(config).to(config.device)
        criterion = None  # baseline uses plain MSE in Trainer

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    trainer = Trainer(model, criterion, optimizer, scheduler, config,
                      is_multitask=is_multitask)
    history = trainer.fit(train_loader, val_loader)

    # Save training curves
    curves_path = os.path.join(config.save_dir,
                               f"{config.experiment_name}_{tag}_curves.png")
    plot_training_curves(history, curves_path, title=f"{tag.title()} Training Curves")
    print(f"  Training curves saved to {curves_path}")

    return model


def _load_checkpoint(model, config, tag):
    """Load <experiment>_<tag>_best.pt into `model` and move it to device."""
    path = os.path.join(config.save_dir,
                        f"{config.experiment_name}_{tag}_best.pt")
    print(f"Loading {tag} checkpoint from {path}")
    ckpt = torch.load(path, map_location=config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(config.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train lighting estimation models")
    parser.add_argument('--mode', choices=['baseline', 'multitask', 'both'],
                        default='both', help='Which model(s) to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone from config')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load saved checkpoints and run '
                             'evaluation + visualizations only.')
    args = parser.parse_args()

    # Apply overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.backbone is not None:
        config.backbone = args.backbone

    # Device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {config.device}")

    # Seed
    torch.manual_seed(config.seed)

    # Transforms
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, test_loader = make_dataloaders(config, transform)

    baseline_model = None
    multitask_model = None

    if args.eval_only:
        if args.mode in ('baseline', 'both'):
            baseline_model = _load_checkpoint(BaselineLightingNet(config), config,
                                              tag='baseline')
        if args.mode in ('multitask', 'both'):
            multitask_model = _load_checkpoint(MaterialAwareLightingNet(config),
                                               config, tag='multitask')
    else:
        if args.mode in ('baseline', 'both'):
            baseline_model = train_model(config, is_multitask=False,
                                         train_loader=train_loader, val_loader=val_loader)

        if args.mode in ('multitask', 'both'):
            multitask_model = train_model(config, is_multitask=True,
                                          train_loader=train_loader, val_loader=val_loader)

    # --- Evaluation on test set ---
    print(f"\n{'='*60}")
    print("Evaluating on test set")
    print(f"{'='*60}\n")

    baseline_metrics = None
    multitask_metrics = None

    baseline_results = None
    multitask_results = None

    if baseline_model is not None:
        baseline_results = evaluate_model(baseline_model, test_loader, config.device,
                                          is_multitask=False)
        baseline_metrics = compute_all_metrics(
            baseline_results,
            material_param_names=config.material_param_names,
        )
        print("Baseline test metrics:")
        agg = baseline_metrics["aggregate"]
        print(f"  log_mse:    {agg['log_mse_mean']:.4f} ± {agg['log_mse_ci95_half']:.4f}")
        print(f"  linear_mse: {agg['linear_mse_mean']:.4f} ± {agg['linear_mse_ci95_half']:.4f}")
        print(f"  psnr_log:   {agg['psnr_log_mean']:.2f} ± {agg['psnr_log_ci95_half']:.2f} dB")
        print()

    if multitask_model is not None:
        multitask_results = evaluate_model(multitask_model, test_loader, config.device,
                                           is_multitask=True)
        multitask_metrics = compute_all_metrics(
            multitask_results,
            material_param_names=config.material_param_names,
        )
        print("Multitask test metrics:")
        agg = multitask_metrics["aggregate"]
        print(f"  log_mse:    {agg['log_mse_mean']:.4f} ± {agg['log_mse_ci95_half']:.4f}")
        print(f"  linear_mse: {agg['linear_mse_mean']:.4f} ± {agg['linear_mse_ci95_half']:.4f}")
        print(f"  psnr_log:   {agg['psnr_log_mean']:.2f} ± {agg['psnr_log_ci95_half']:.2f} dB")
        if "material_param_mae" in multitask_metrics:
            mae = multitask_metrics["material_param_mae"]
            print(f"  material_param_mae (mean): {mae['mean']:.4f}")
            for name, v in mae["per_param"].items():
                print(f"    {name:<14} {v:.4f}")
        print()

    # Side-by-side comparison (only when both were trained)
    if baseline_metrics is not None and multitask_metrics is not None:
        compare_models(baseline_metrics, multitask_metrics,
                       material_param_names=config.material_param_names)

    # --- Visualizations ---
    vis_dir = os.path.join("result", config.experiment_name)

    # Per-material comparison chart
    if baseline_metrics is not None and multitask_metrics is not None:
        chart_path = os.path.join(vis_dir, "per_material_comparison.png")
        plot_per_material_comparison(baseline_metrics, multitask_metrics, chart_path)
        print(f"Per-material chart saved to {chart_path}")

    # Envmap GT vs predicted comparisons for a few test samples
    for tag, raw_results in [("baseline", baseline_results),
                              ("multitask", multitask_results)]:
        if raw_results is None:
            continue
        n_vis = min(5, len(raw_results["pred_env"]))
        for i in range(n_vis):
            pred = raw_results["pred_env"][i]
            target = raw_results["target_env"][i]
            env_path = os.path.join(vis_dir, f"{tag}_envmap_{i}.png")
            plot_envmap_comparison(pred, target, env_path,
                                   title=f"{tag.title()} Sample {i}")

    # Save metrics to JSON. Drop per-sample arrays (they're large and only
    # needed in-memory for the paired t-test, which we persist separately).
    os.makedirs(config.save_dir, exist_ok=True)
    results_path = os.path.join(config.save_dir,
                                f"{config.experiment_name}_results.json")

    def _strip_per_sample(m):
        if m is None:
            return None
        out = {k: v for k, v in m.items() if k != "per_sample"}
        return out

    saved = {}
    if baseline_metrics is not None:
        saved["baseline"] = _strip_per_sample(baseline_metrics)
    if multitask_metrics is not None:
        saved["multitask"] = _strip_per_sample(multitask_metrics)

    # Persist the paired t-test on per-sample angular errors
    if baseline_metrics is not None and multitask_metrics is not None:
        from evaluation.metrics import paired_ttest
        b_log = baseline_metrics.get("per_sample", {}).get("log_mse")
        m_log = multitask_metrics.get("per_sample", {}).get("log_mse")
        if b_log is not None and m_log is not None and len(b_log) == len(m_log):
            saved["paired_ttest_log_mse"] = paired_ttest(b_log, m_log)

    with open(results_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

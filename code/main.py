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
    render_sphere_comparison, plot_sh_coefficients,
    plot_per_material_comparison, plot_training_curves,
    render_combined_comparison, plot_combined_sh,
)


def make_datasets_and_loaders(config, transform):
    train_ds = SphereDataset('train', config, transform=transform)
    val_ds = SphereDataset('val', config, transform=transform)
    test_ds = SphereDataset('test', config, transform=transform)

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
    return train_ds, val_loader, test_loader


def train_model(config, is_multitask, train_ds, val_loader):
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
    history = trainer.fit(train_ds, val_loader)

    # Save training curves
    curves_path = os.path.join("result", config.experiment_name,
                               f"{tag}_curves.png")
    plot_training_curves(history, curves_path, title=f"{tag.title()} Training Curves")
    print(f"  Training curves saved to {curves_path}")

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

    train_ds, val_loader, test_loader = make_datasets_and_loaders(config, transform)

    baseline_model = None
    multitask_model = None

    if args.mode in ('baseline', 'both'):
        baseline_model = train_model(config, is_multitask=False,
                                     train_ds=train_ds, val_loader=val_loader)

    if args.mode in ('multitask', 'both'):
        multitask_model = train_model(config, is_multitask=True,
                                      train_ds=train_ds, val_loader=val_loader)

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
        baseline_metrics = compute_all_metrics(baseline_results)
        print("Baseline test metrics:")
        agg = baseline_metrics["aggregate"]
        print(f"  angular_error_mean:  {agg['angular_error_mean']:.2f} deg")
        print(f"  angular_error_median: {agg['angular_error_median']:.2f} deg")
        print(f"  intensity_error:     {agg['intensity_error_mean']:.4f}")
        print(f"  sh_mse:              {agg['sh_mse_mean']:.4f}")
        print()

    if multitask_model is not None:
        multitask_results = evaluate_model(multitask_model, test_loader, config.device,
                                           is_multitask=True)
        multitask_metrics = compute_all_metrics(multitask_results)
        print("Multitask test metrics:")
        agg = multitask_metrics["aggregate"]
        print(f"  angular_error_mean:  {agg['angular_error_mean']:.2f} deg")
        print(f"  angular_error_median: {agg['angular_error_median']:.2f} deg")
        print(f"  intensity_error:     {agg['intensity_error_mean']:.4f}")
        print(f"  sh_mse:              {agg['sh_mse_mean']:.4f}")
        if "material_accuracy" in multitask_metrics:
            print(f"  material_accuracy:   {multitask_metrics['material_accuracy']:.4f}")
        print()

    # Side-by-side comparison (only when both were trained)
    if baseline_metrics is not None and multitask_metrics is not None:
        compare_models(baseline_metrics, multitask_metrics)

    # --- Visualizations ---
    vis_dir = os.path.join("result", config.experiment_name)

    # Per-material comparison chart
    if baseline_metrics is not None and multitask_metrics is not None:
        chart_path = os.path.join(vis_dir, "per_material_comparison.png")
        plot_per_material_comparison(baseline_metrics, multitask_metrics, chart_path)
        print(f"Per-material chart saved to {chart_path}")

    # Sphere renders + SH bar charts for a few test samples
    for tag, raw_results in [("baseline", baseline_results),
                              ("multitask", multitask_results)]:
        if raw_results is None:
            continue
        n_vis = min(5, len(raw_results["pred_sh"]))
        for i in range(n_vis):
            pred = raw_results["pred_sh"][i]
            target = raw_results["target_sh"][i]

            sphere_path = os.path.join(vis_dir, f"{tag}_sphere_{i}.png")
            render_sphere_comparison(pred, target, sphere_path,
                                     title=f"{tag.title()} Sample {i}")

            sh_path = os.path.join(vis_dir, f"{tag}_sh_{i}.png")
            plot_sh_coefficients(pred, target, sh_path,
                                  title=f"{tag.title()} SH Coefficients - Sample {i}")

    # Combined comparison visualizations (when both models were trained)
    if baseline_results is not None and multitask_results is not None:
        test_ds = SphereDataset('test', config)
        n_vis = min(5, len(baseline_results["pred_sh"]))
        for i in range(n_vis):
            real_idx = test_ds.indices[i]
            img_path = test_ds.image_paths[real_idx]

            combo_path = os.path.join(vis_dir, f"combined_sphere_{i}.png")
            render_combined_comparison(
                img_path,
                baseline_results["pred_sh"][i],
                multitask_results["pred_sh"][i],
                baseline_results["target_sh"][i],
                combo_path,
                title=f"Sample {i}",
            )

            sh_combo_path = os.path.join(vis_dir, f"combined_sh_{i}.png")
            plot_combined_sh(
                baseline_results["pred_sh"][i],
                multitask_results["pred_sh"][i],
                baseline_results["target_sh"][i],
                sh_combo_path,
                title=f"SH Coefficients - Sample {i}",
            )
        print(f"Combined comparison images saved to {vis_dir}")

    # Save metrics to JSON
    os.makedirs(config.save_dir, exist_ok=True)
    results_path = os.path.join(config.save_dir,
                                f"{config.experiment_name}_results.json")
    saved = {}
    if baseline_metrics is not None:
        saved["baseline"] = baseline_metrics
    if multitask_metrics is not None:
        saved["multitask"] = multitask_metrics
    with open(results_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

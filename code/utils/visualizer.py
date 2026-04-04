"""Visualize predicted vs ground truth lighting by rendering a sphere"""

# TODO: Implement visualization utilities
#   - render_sphere_comparison(pred_sh, target_sh, save_path):
#       Render a sphere under predicted and GT lighting side-by-side.
#       Could use matplotlib + simple Lambertian shading from SH,
#       or call Blender headless with the SH-derived environment map.
#
#   - plot_sh_coefficients(pred_sh, target_sh, save_path):
#       Bar chart comparing predicted vs GT SH coefficients per band/channel.
#
#   - plot_training_curves(train_log, save_path):
#       Loss curves (total, lighting, material) over epochs for train and val.

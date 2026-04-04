# Spherical harmonics utilities for lighting analysis

# TODO: Implement SH helper functions
#   - sh_to_dominant_direction(sh_coeffs):
#       Already implemented in evaluation/metrics.py as dominant_light_direction().
#       Could re-export or extend here for broader use.
#
#   - sh_to_irradiance_map(sh_coeffs, resolution=64):
#       Reconstruct a lat-long irradiance map from SH coefficients for visualization.
#
#   - reconstruct_environment(sh_coeffs, resolution=256):
#       Approximate environment map from band-0/1/2 SH coefficients.

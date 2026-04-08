"""
Synthetic dataset generator 
===================================================================================
Renders spheres with distinct Principled BSDF material categories (diffuse, glossy,
metallic, rough_metallic, dielectric) lit by HDRI environment maps.  Outputs images
with descriptive filenames and a metadata.json containing ground-truth 27-dim
spherical-harmonics coefficients, material type/label, and material parameters.

Usage
-----
    blender --background --python code/data_gen.py -- [options]

Options
-------
    --resolution  INT   Image resolution in pixels   (default 256)
    --num_images  INT   Total images to generate      (default 10000)
    --seed        INT   Random seed                   (default 42)
    --output_dir  PATH  Output directory              (default dataset/renders)
    --hdri_dir    PATH  Root HDRI folder              (default dataset/hdris)
    --samples     INT   Cycles samples per pixel      (default 128)

Example
--------------------
    blender --background --python code/data_gen.py -- --num_images 20 --resolution 128 --samples 32
"""

import bpy
import os
import sys
import json
import math
import random
import re
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI  (arguments after the "--" separator belong to this script)
# ---------------------------------------------------------------------------
import argparse

_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_p = argparse.ArgumentParser(description="Generate synthetic sphere renders")
_p.add_argument("--resolution", type=int, default=256)
_p.add_argument("--num_images", type=int, default=10000)
_p.add_argument("--seed", type=int, default=42)
_p.add_argument("--output_dir", type=str, default=None)
_p.add_argument("--hdri_dir", type=str, default=None)
_p.add_argument("--samples", type=int, default=128)
args = _p.parse_args(_argv)

# Paths (resolve relative to project root = parent of code/)
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path(os.getcwd())
_PROJECT_DIR = _SCRIPT_DIR.parent

OUTPUT_DIR = Path(args.output_dir) if args.output_dir else _PROJECT_DIR / "dataset" / "renders"
HDRI_DIR = Path(args.hdri_dir) if args.hdri_dir else _PROJECT_DIR / "dataset" / "hdris"
IMAGE_DIR = OUTPUT_DIR / "images"

# Material definitions
# Ranges given as (min, max) are sampled uniformly; scalars are fixed.
MATERIAL_DEFS = {
    "diffuse": {
        "label": 0,
        "metallic": 0.0,
        "roughness": (0.7, 1.0),
        "specular": 0.0,
        "transmission": 0.0,
        "ior": 1.45,
    },
    "glossy": {
        "label": 1,
        "metallic": 0.0,
        "roughness": (0.0, 0.15),
        "specular": 0.8,
        "transmission": 0.0,
        "ior": 1.45,
    },
    "metallic": {
        "label": 2,
        "metallic": 1.0,
        "roughness": (0.0, 0.15),
        "specular": 0.5,
        "transmission": 0.0,
        "ior": 1.45,
    },
    "rough_metallic": {
        "label": 3,
        "metallic": 1.0,
        "roughness": (0.4, 0.9),
        "specular": 0.5,
        "transmission": 0.0,
        "ior": 1.45,
    },
    "dielectric": {
        "label": 4,
        "metallic": 0.0,
        "roughness": (0.0, 0.1),
        "specular": 0.5,
        "transmission": (0.5, 1.0),
        "ior": (1.3, 1.7),
    },
}
MAT_NAMES = list(MATERIAL_DEFS.keys())

# ---------------------------------------------------------------------------
# Named colour palette  (values are linear-sRGB for Blender)
# ---------------------------------------------------------------------------
NAMED_COLORS = {
    "red":       (0.80, 0.05, 0.05),
    "scarlet":   (0.90, 0.15, 0.05),
    "orange":    (0.90, 0.35, 0.05),
    "amber":     (0.85, 0.55, 0.10),
    "yellow":    (0.90, 0.80, 0.10),
    "lime":      (0.50, 0.80, 0.10),
    "green":     (0.10, 0.60, 0.10),
    "emerald":   (0.10, 0.70, 0.35),
    "teal":      (0.10, 0.55, 0.50),
    "cyan":      (0.10, 0.70, 0.75),
    "sky":       (0.30, 0.50, 0.90),
    "blue":      (0.10, 0.15, 0.80),
    "indigo":    (0.20, 0.10, 0.60),
    "purple":    (0.50, 0.10, 0.70),
    "magenta":   (0.75, 0.10, 0.55),
    "pink":      (0.90, 0.40, 0.50),
    "white":     (0.90, 0.90, 0.90),
    "cream":     (0.90, 0.85, 0.70),
    "gray":      (0.50, 0.50, 0.50),
    "charcoal":  (0.15, 0.15, 0.15),
    "brown":     (0.35, 0.18, 0.05),
    "tan":       (0.70, 0.55, 0.35),
    "copper":    (0.72, 0.45, 0.20),
    "slate":     (0.40, 0.45, 0.50),
}
COLOR_NAMES = list(NAMED_COLORS.keys())


# =========================================================================
# Spherical-harmonics utilities
# =========================================================================

"""Compute order-2 real SH coefficients from an equirectangular HDR image.

    Uses the standard real SH basis (band 0-2, 9 functions) integrated over
    the sphere via discrete summation.

    Parameters
    ----------
    blender_image : bpy.types.Image
        Already-loaded Blender image (equirectangular .hdr/.exr).

    Returns
    -------
    np.ndarray, shape (9, 3)
        SH coefficients per basis function (rows) per RGB channel (cols).
        Row order: Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1,
                   Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2.
"""
def compute_sh_coefficients(blender_image):
    
    w, h = blender_image.size
    pixels = np.array(blender_image.pixels[:], dtype=np.float32).reshape(h, w, 4)
    rgb = pixels[:, :, :3]

    # Pixel grid → spherical coordinates
    j_arr = np.arange(w, dtype=np.float32)
    i_arr = np.arange(h, dtype=np.float32)
    jj, ii = np.meshgrid(j_arr, i_arr)

    theta = np.pi * (ii + 0.5) / h           # polar angle   [0, pi]
    phi = 2.0 * np.pi * (jj + 0.5) / w       # azimuth       [0, 2*pi]

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    # Solid-angle element for equirectangular projection
    d_omega = sin_t * (np.pi / h) * (2.0 * np.pi / w)

    # Real SH basis functions (band 0-2)
    basis = [
        0.282095 * np.ones_like(theta),        # l=0  m= 0
        0.488603 * y,                           # l=1  m=-1
        0.488603 * z,                           # l=1  m= 0
        0.488603 * x,                           # l=1  m=+1
        1.092548 * x * y,                       # l=2  m=-2
        1.092548 * y * z,                       # l=2  m=-1
        0.315392 * (3.0 * z * z - 1.0),        # l=2  m= 0
        1.092548 * x * z,                       # l=2  m=+1
        0.546274 * (x * x - y * y),             # l=2  m=+2
    ]

    coeffs = np.zeros((9, 3), dtype=np.float64)
    for bi, Y in enumerate(basis):
        weighted = Y * d_omega
        for c in range(3):
            coeffs[bi, c] = np.sum(rgb[:, :, c] * weighted)

    return coeffs


"""Z-axis rotation of SH coefficients.

    Applies the analytic rotation that matches Blender's Mapping-node
    rotation_euler.z = *angle* on the environment texture.

    For each band the (m, -m) coefficient pairs undergo a 2-D rotation by
    m * angle.  The m = 0 coefficients are unchanged.

    Parameters
    ----------
    coeffs : np.ndarray, shape (9, 3)
    angle  : float, radians

    Returns
    -------
    np.ndarray, shape (9, 3)
    """
def rotate_sh_z(coeffs, angle):
    out = coeffs.copy()
    ca, sa = np.cos(angle), np.sin(angle)
    c2a, s2a = np.cos(2 * angle), np.sin(2 * angle)

    # l=1  pair (idx 1 = m-1, idx 3 = m+1)
    out[1] = ca * coeffs[1] - sa * coeffs[3]
    out[3] = sa * coeffs[1] + ca * coeffs[3]

    # l=2  m=+-1 pair (idx 5 = m-1, idx 7 = m+1)
    out[5] = ca * coeffs[5] - sa * coeffs[7]
    out[7] = sa * coeffs[5] + ca * coeffs[7]

    # l=2  m=+-2 pair (idx 4 = m-2, idx 8 = m+2)
    out[4] = c2a * coeffs[4] - s2a * coeffs[8]
    out[8] = s2a * coeffs[4] + c2a * coeffs[8]

    return out


# =========================================================================
# HDRI discovery
# =========================================================================

def discover_hdris(root):
    """Return [(Path, category_str), ...] for every .hdr/.exr under root,
    skipping accidental duplicate downloads like 'name (1).hdr'."""
    found = []
    for category in ("puresky", "scene"):
        cat_dir = root / category
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.iterdir()):
            if f.suffix.lower() not in (".hdr", ".exr"):
                continue
            if re.search(r"\(\d+\)", f.stem):       # skip "(1)", "(2)" duplicates
                continue
            found.append((f, category))
    return found


def hdri_short_name(filename):
    """Derive a concise, readable tag from an HDRI filename.

    Examples:
        sunset_fairway_2k.hdr           -> sunset_fairway
        belfast_sunset_puresky_2k.hdr   -> belfast_sunset
    """
    name = Path(filename).stem
    name = re.sub(r"_\d+k$", "", name)              # strip resolution tag
    name = name.replace("_puresky", "")              # strip puresky tag
    name = re.sub(r"\s*\(\d+\)$", "", name)          # strip duplicate markers
    return name.strip().replace(" ", "_")


# =========================================================================
# Blender scene helpers
# =========================================================================

    """Remove every object and material from the scene."""
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)

    """Configure Cycles renderer, colour management, and GPU if available."""
def setup_render(resolution, samples):
    sc = bpy.context.scene

    sc.render.engine = "CYCLES"
    sc.render.resolution_x = resolution
    sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.film_transparent = False

    sc.cycles.samples = samples
    sc.cycles.use_denoising = True

    sc.view_settings.view_transform = "Filmic"

    # Try to enable GPU rendering if available (NVIDIA first, then AMD / Apple / Intel)
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is not None:
        cp = addon.preferences
        for device_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cp.compute_device_type = device_type
                cp.get_devices()
                gpu_devs = [d for d in cp.devices if d.type != "CPU"]
                if gpu_devs:
                    for d in cp.devices:
                        d.use = True
                    sc.cycles.device = "GPU"
                    print(f"Render device: {device_type} "
                          f"({', '.join(d.name for d in gpu_devs)})")
                    return
            except Exception:
                continue
    sc.cycles.device = "CPU"
    print("Render device: CPU")

"""Place a camera at (0, -4, 0) looking at the origin, framing a unit sphere."""
def create_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, -4, 0)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = cam_obj
    return cam_obj


"""Create a smooth-shaded UV sphere (radius 1) at the origin."""
def create_sphere():
    mesh = bpy.data.meshes.new("SphereMesh")
    obj = bpy.data.objects.new("Sphere", mesh)
    bpy.context.collection.objects.link(obj)

    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    bm.to_mesh(mesh)
    bm.free()

    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj

"""Build the world node tree: HDRI for lighting, neutral gray for camera.

    Uses the Light-Path "Is Camera Ray" trick so the HDRI lights the sphere
    via diffuse/glossy/transmission rays, but the camera sees a neutral 18 %
    gray background.

    Returns
    -------
    (env_node, mapping_node, bg_hdri_node) for later per-frame updates.
    """
def build_world_shader():
    
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    link = tree.links.new

    n_coord = tree.nodes.new("ShaderNodeTexCoord")
    n_map = tree.nodes.new("ShaderNodeMapping")
    n_env = tree.nodes.new("ShaderNodeTexEnvironment")
    n_bg_hdri = tree.nodes.new("ShaderNodeBackground")
    n_bg_black = tree.nodes.new("ShaderNodeBackground")
    n_lpath = tree.nodes.new("ShaderNodeLightPath")
    n_mix = tree.nodes.new("ShaderNodeMixShader")
    n_out = tree.nodes.new("ShaderNodeOutputWorld")

    n_bg_black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    n_bg_black.inputs["Strength"].default_value = 1.0

    link(n_coord.outputs["Generated"], n_map.inputs["Vector"])
    link(n_map.outputs["Vector"], n_env.inputs["Vector"])
    link(n_env.outputs["Color"], n_bg_hdri.inputs["Color"])
    link(n_lpath.outputs["Is Camera Ray"], n_mix.inputs["Fac"])
    link(n_bg_hdri.outputs["Background"], n_mix.inputs[1])   # fac=0 → HDRI
    link(n_bg_black.outputs["Background"], n_mix.inputs[2])   # fac=1 → gray
    link(n_mix.outputs["Shader"], n_out.inputs["Surface"])

    return n_env, n_map, n_bg_hdri


# =========================================================================
# Material application
# =========================================================================

def apply_material(obj, mat_type, color_name, rng):
    """Create / update a Principled BSDF material on *obj*.

    Returns a dict of the sampled parameter values (for metadata).
    """
    defn = MATERIAL_DEFS[mat_type]
    rgb = NAMED_COLORS[color_name]

    def _sample(v):
        return rng.uniform(*v) if isinstance(v, tuple) else v

    roughness = _sample(defn["roughness"])
    transmission = _sample(defn["transmission"])
    ior = _sample(defn["ior"])

    # Reuse a single material slot
    mat = bpy.data.materials.get("SphereMat")
    if mat is None:
        mat = bpy.data.materials.new("SphereMat")
    mat.use_nodes = True

    pr = mat.node_tree.nodes.get("Principled BSDF")
    if pr is None:
        mat.node_tree.nodes.clear()
        pr = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        out_node = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(pr.outputs["BSDF"], out_node.inputs["Surface"])

    pr.inputs["Base Color"].default_value = (*rgb, 1.0)
    pr.inputs["Metallic"].default_value = defn["metallic"]
    pr.inputs["Roughness"].default_value = roughness
    pr.inputs["IOR"].default_value = ior

    # Handle Blender 3.x vs 4.x input renames
    spec_key = ("Specular IOR Level" if "Specular IOR Level" in pr.inputs
                else "Specular")
    pr.inputs[spec_key].default_value = defn["specular"]

    trans_key = ("Transmission Weight" if "Transmission Weight" in pr.inputs
                 else "Transmission")
    pr.inputs[trans_key].default_value = transmission

    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    return {
        "base_color_rgb": list(rgb),
        "metallic": defn["metallic"],
        "roughness": round(roughness, 4),
        "specular": defn["specular"],
        "transmission": round(transmission, 4),
        "ior": round(ior, 4),
    }


# =========================================================================
# Main generation loop
# =========================================================================

def main():
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Discover HDRIs ----
    hdris = discover_hdris(HDRI_DIR)
    if not hdris:
        sys.exit(f"ERROR: no HDRIs found under {HDRI_DIR}")
    print(f"Found {len(hdris)} HDRIs")

    # ---- Scene setup ----
    clear_scene()
    setup_render(args.resolution, args.samples)
    create_camera()
    sphere = create_sphere()
    env_node, map_node, bg_node = build_world_shader()

    # ---- Pre-load HDRIs and compute SH coefficients ----
    print("Pre-computing SH coefficients for all HDRIs ...")
    sh_cache = {}      # str(path) -> (9, 3) array
    img_cache = {}     # str(path) -> bpy.types.Image
    for hdri_path, _ in hdris:
        key = str(hdri_path)
        print(f"  {hdri_path.name}")
        img = bpy.data.images.load(str(hdri_path))
        img_cache[key] = img
        sh_cache[key] = compute_sh_coefficients(img)
    print(f"SH pre-computation complete.\n")

    # ---- Render loop ----
    metadata = []

    try:
        for idx in range(args.num_images):
            # Random selections
            hdri_path, hdri_cat = rng.choice(hdris)
            mat_type = rng.choice(MAT_NAMES)
            color_name = rng.choice(COLOR_NAMES)
            rotation_deg = rng.randint(0, 359)
            rotation_rad = math.radians(rotation_deg)
            strength = round(rng.uniform(0.5, 2.0), 4)

            # Descriptive filename
            short = hdri_short_name(hdri_path.name)
            filename = (f"{idx:05d}_{mat_type}_{color_name}"
                        f"_{short}_rot{rotation_deg:03d}.png")

            # Material
            mat_params = apply_material(sphere, mat_type, color_name, rng)

            # HDRI environment
            env_node.image = img_cache[str(hdri_path)]
            map_node.inputs["Rotation"].default_value = (0, 0, rotation_rad)
            bg_node.inputs["Strength"].default_value = strength

            # Render
            bpy.context.scene.render.filepath = str(IMAGE_DIR / filename)
            bpy.ops.render.render(write_still=True)

            # Ground-truth SH coefficients (rotated + intensity-scaled)
            sh = rotate_sh_z(sh_cache[str(hdri_path)], rotation_rad) * strength
            # Flatten to 27-dim: [R0..R8, G0..G8, B0..B8]
            sh_flat = sh.T.flatten().tolist()

            metadata.append({
                "filename": filename,
                "material_type": mat_type,
                "material_label": MATERIAL_DEFS[mat_type]["label"],
                "color_name": color_name,
                **mat_params,
                "hdri_file": hdri_path.name,
                "hdri_category": hdri_cat,
                "hdri_short_name": short,
                "rotation_z_deg": rotation_deg,
                "rotation_z_rad": round(rotation_rad, 6),
                "world_strength": strength,
                "sh_coefficients": [round(v, 6) for v in sh_flat],
            })

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[{idx + 1:>{len(str(args.num_images))}}"
                      f"/{args.num_images}] {filename}")

    finally:
        # Always save metadata, even on interrupt (partial is better than none)
        meta_path = OUTPUT_DIR / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved {len(metadata)} records -> {meta_path}")
        print(f"Images -> {IMAGE_DIR}")


if __name__ == "__main__":
    main()

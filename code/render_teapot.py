"""Headless Blender script: render the Utah teapot under one HDRI envmap.

Invoked as a subprocess by code/utils/teapot_compare.py:

    blender --background --python code/render_teapot.py -- \
        --teapot dataset/renders/utah_teapot.obj \
        --hdri PATH \
        --output OUT.png \
        [--rotation_z_deg 0.0] [--strength 1.0] \
        [--resolution 256] [--samples 32]

For ground-truth renders, --hdri points at the original .hdr/.exr in
dataset/hdris/, with --rotation_z_deg and --strength taken from the
sample's metadata. For predicted renders, --hdri points at the
.hdr file written from the model's output (rotation/strength already
baked into the prediction), so pass rotation=0 and strength=1.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy


# ---------------------------------------------------------------------------
# CLI (arguments after "--" belong to this script)
# ---------------------------------------------------------------------------
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--teapot", required=True)
_p.add_argument("--hdri", required=True)
_p.add_argument("--output", required=True)
_p.add_argument("--rotation_z_deg", type=float, default=0.0)
_p.add_argument("--strength", type=float, default=1.0)
_p.add_argument("--resolution", type=int, default=512)
_p.add_argument("--samples", type=int, default=32)
args = _p.parse_args(_argv)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


def setup_render(resolution, samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = resolution
    sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.compression = 0
    sc.render.film_transparent = False
    sc.render.use_persistent_data = True

    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.adaptive_min_samples = max(8, samples // 8)
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 4
    sc.cycles.transmission_bounces = 4
    sc.cycles.transparent_max_bounces = 4
    sc.cycles.volume_bounces = 0

    sc.view_settings.view_transform = "Filmic"

    addon = bpy.context.preferences.addons.get("cycles")
    if addon is not None:
        cp = addon.preferences
        for dt in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cp.compute_device_type = dt
                cp.get_devices()
                if [d for d in cp.devices if d.type != "CPU"]:
                    for d in cp.devices:
                        d.use = True
                    sc.cycles.device = "GPU"
                    if dt == "OPTIX":
                        try:
                            sc.cycles.denoiser = "OPTIX"
                        except Exception:
                            pass
                    return
            except Exception:
                continue
    sc.cycles.device = "CPU"


def import_teapot(path):
    """Import OBJ across Blender 3.x / 4.x and return the mesh object."""
    before = set(bpy.data.objects)
    try:
        bpy.ops.wm.obj_import(filepath=path)         # Blender 3.4+
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=path)      # Older
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        sys.exit(f"[render_teapot] No mesh imported from {path}")
    # If the OBJ contains multiple objects, join them so we have one teapot.
    bpy.ops.object.select_all(action="DESELECT")
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize_object(obj, target_size=2.0, sit_on_ground=True):
    """Center the object at origin and scale so its longest dimension equals
    target_size. Optionally translate so its base sits at z=0."""
    bpy.context.view_layer.update()
    bbox_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [p.x for p in bbox_world]
    ys = [p.y for p in bbox_world]
    zs = [p.z for p in bbox_world]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    bottom = min(zs) if sit_on_ground else (min(zs) + max(zs)) / 2.0
    obj.location = (obj.location.x - cx, obj.location.y - cy,
                    obj.location.z - bottom)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    bbox_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [p.x for p in bbox_world]
    ys = [p.y for p in bbox_world]
    zs = [p.z for p in bbox_world]
    extent = max(max(xs) - min(xs),
                 max(ys) - min(ys),
                 max(zs) - min(zs))
    if extent > 1e-6:
        s = target_size / extent
        obj.scale = (s, s, s)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def apply_ceramic_material(obj):
    """Slightly glossy off-white ceramic-style Principled BSDF.

    Tuned to read like glazed porcelain under HDRI lighting: bright nearly-
    white diffuse, low (but not mirror) roughness so highlights are sharp
    and clearly carry environment color, IOR ~1.5 like glass/glaze.
    """
    mat = bpy.data.materials.new("TeapotMat")
    mat.use_nodes = True
    pr = mat.node_tree.nodes.get("Principled BSDF")
    if pr is None:
        mat.node_tree.nodes.clear()
        pr = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(pr.outputs["BSDF"], out.inputs["Surface"])

    pr.inputs["Base Color"].default_value = (0.92, 0.92, 0.90, 1.0)
    pr.inputs["Metallic"].default_value = 0.0
    pr.inputs["Roughness"].default_value = 0.15
    pr.inputs["IOR"].default_value = 1.55

    #  Blender 3.x / 4.x saftey just in case
    spec_key = ("Specular IOR Level" if "Specular IOR Level" in pr.inputs
                else "Specular")
    pr.inputs[spec_key].default_value = 0.6

    # Optional clearcoat for a subtle glaze pop where the input exists.
    for coat_key, coat_val in [("Coat Weight", 0.3), ("Clearcoat", 0.3)]:
        if coat_key in pr.inputs:
            pr.inputs[coat_key].default_value = coat_val
            break
    for coat_r_key in ("Coat Roughness", "Clearcoat Roughness"):
        if coat_r_key in pr.inputs:
            pr.inputs[coat_r_key].default_value = 0.05
            break

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -5.0, 2.2)
    cam_obj.rotation_euler = (math.radians(72), 0.0, 0.0)
    bpy.context.scene.camera = cam_obj


def add_ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    plane = bpy.context.object
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    pr = mat.node_tree.nodes.get("Principled BSDF")
    pr.inputs["Base Color"].default_value = (0.32, 0.32, 0.32, 1.0)
    pr.inputs["Roughness"].default_value = 0.75
    plane.data.materials.append(mat)


def setup_world_hdri(hdri_path, rotation_deg, strength):
    sc = bpy.context.scene
    world = sc.world
    if world is None:
        world = bpy.data.worlds.new("World")
        sc.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    link = tree.links.new

    n_coord = tree.nodes.new("ShaderNodeTexCoord")
    n_map = tree.nodes.new("ShaderNodeMapping")
    n_env = tree.nodes.new("ShaderNodeTexEnvironment")
    n_bg = tree.nodes.new("ShaderNodeBackground")
    n_out = tree.nodes.new("ShaderNodeOutputWorld")

    img = bpy.data.images.load(hdri_path)
    n_env.image = img
    n_map.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    n_bg.inputs["Strength"].default_value = strength

    link(n_coord.outputs["Generated"], n_map.inputs["Vector"])
    link(n_map.outputs["Vector"], n_env.inputs["Vector"])
    link(n_env.outputs["Color"], n_bg.inputs["Color"])
    link(n_bg.outputs["Background"], n_out.inputs["Surface"])


def main():
    if not Path(args.teapot).exists():
        sys.exit(f"[render_teapot] Teapot OBJ not found: {args.teapot}")
    if not Path(args.hdri).exists():
        sys.exit(f"[render_teapot] HDRI not found: {args.hdri}")

    clear_scene()
    setup_render(args.resolution, args.samples)
    add_camera()

    teapot = import_teapot(args.teapot)
    normalize_object(teapot, target_size=2.0, sit_on_ground=True)
    apply_ceramic_material(teapot)

    add_ground_plane()
    setup_world_hdri(args.hdri, args.rotation_z_deg, args.strength)

    bpy.context.scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)
    print(f"[render_teapot] wrote {args.output}")


if __name__ == "__main__":
    main()

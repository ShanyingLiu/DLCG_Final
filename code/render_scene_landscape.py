"""One-off: render a landscape scene with a sphere + Utah teapot floating
in an HDRI environment (no platform), using Cycles.

Usage:
  blender --background --python code/render_scene_landscape.py
"""

import math
import sys
from pathlib import Path

import bpy


HDRI = "dataset/hdris/scene/autumn_hill_view_2k.hdr"
TEAPOT_OBJ = "dataset/renders/utah_teapot.obj"
OUT_PATH = "result/scene_landscape.png"
RES_X, RES_Y = 1920, 1080
SAMPLES = 128


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x = RES_X
    sc.render.resolution_y = RES_Y
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.film_transparent = False

    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.max_bounces = 8
    sc.cycles.glossy_bounces = 6
    sc.cycles.transmission_bounces = 6

    sc.view_settings.view_transform = "Filmic"

    addon = bpy.context.preferences.addons.get("cycles")
    if addon is not None:
        cp = addon.preferences
        for dt in ("OPTIX", "CUDA", "METAL", "HIP", "ONEAPI"):
            try:
                cp.compute_device_type = dt
                cp.get_devices()
                if [d for d in cp.devices if d.type != "CPU"]:
                    for d in cp.devices:
                        d.use = True
                    sc.cycles.device = "GPU"
                    return
            except Exception:
                continue
    sc.cycles.device = "CPU"


def import_teapot(path):
    before = set(bpy.data.objects)
    try:
        bpy.ops.wm.obj_import(filepath=path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=path)
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        sys.exit("Teapot import failed")
    bpy.ops.object.select_all(action="DESELECT")
    for o in new:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize_centered(obj, target_size=2.0):
    """Center at origin, scale longest dim to target_size, center vertically."""
    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [p.x for p in bbox]; ys = [p.y for p in bbox]; zs = [p.z for p in bbox]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    obj.location = (obj.location.x - cx, obj.location.y - cy,
                    obj.location.z - cz)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [p.x for p in bbox]; ys = [p.y for p in bbox]; zs = [p.z for p in bbox]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if extent > 1e-6:
        s = target_size / extent
        obj.scale = (s, s, s)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def make_ceramic_material(name="Ceramic"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    pr = mat.node_tree.nodes.get("Principled BSDF")
    pr.inputs["Base Color"].default_value = (0.92, 0.92, 0.90, 1.0)
    pr.inputs["Metallic"].default_value = 0.0
    pr.inputs["Roughness"].default_value = 0.18
    pr.inputs["IOR"].default_value = 1.5
    spec = "Specular IOR Level" if "Specular IOR Level" in pr.inputs else "Specular"
    pr.inputs[spec].default_value = 0.6
    for k, v in [("Coat Weight", 0.25), ("Clearcoat", 0.25)]:
        if k in pr.inputs:
            pr.inputs[k].default_value = v
            break
    return mat


def make_metal_material(name="Metal"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    pr = mat.node_tree.nodes.get("Principled BSDF")
    pr.inputs["Base Color"].default_value = (0.85, 0.78, 0.65, 1.0)  # warm bronze-ish
    pr.inputs["Metallic"].default_value = 1.0
    pr.inputs["Roughness"].default_value = 0.12
    return mat


def add_sphere(location, radius=1.1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=128,
                                         ring_count=64, location=location)
    obj = bpy.context.object
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def add_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    # Slightly elevated, framing both objects with the horizon visible behind.
    cam_obj.location = (0.0, -12.0, 0.4)
    cam_obj.rotation_euler = (math.radians(89), 0.0, 0.0)
    bpy.context.scene.camera = cam_obj


def setup_world(hdri_path, strength=1.0, rotation_deg=0.0):
    sc = bpy.context.scene
    world = sc.world or bpy.data.worlds.new("World")
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

    n_env.image = bpy.data.images.load(hdri_path)
    n_map.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    n_bg.inputs["Strength"].default_value = strength

    link(n_coord.outputs["Generated"], n_map.inputs["Vector"])
    link(n_map.outputs["Vector"], n_env.inputs["Vector"])
    link(n_env.outputs["Color"], n_bg.inputs["Color"])
    link(n_bg.outputs["Background"], n_out.inputs["Surface"])


def main():
    if not Path(HDRI).exists():
        sys.exit(f"HDRI not found: {HDRI}")
    if not Path(TEAPOT_OBJ).exists():
        sys.exit(f"Teapot OBJ not found: {TEAPOT_OBJ}")

    clear_scene()
    setup_render()
    add_camera()
    setup_world(HDRI, strength=1.0, rotation_deg=220.0)

    # Teapot on the right, floating
    teapot = import_teapot(TEAPOT_OBJ)
    normalize_centered(teapot, target_size=2.4)
    teapot.rotation_euler = (math.radians(90), 0.0, math.radians(-30))
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    teapot.location = (2.0, 0.0, 0.0)
    teapot.data.materials.clear()
    teapot.data.materials.append(make_ceramic_material("TeapotMat"))
    for poly in teapot.data.polygons:
        poly.use_smooth = True

    # Sphere on the left, floating
    sphere = add_sphere(location=(-2.0, 0.0, 0.0), radius=1.1)
    sphere.data.materials.clear()
    sphere.data.materials.append(make_metal_material("SphereMat"))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = OUT_PATH
    bpy.ops.render.render(write_still=True)
    print(f"[scene] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

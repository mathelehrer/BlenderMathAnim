"""
Generate the three pictures used by the ``fractal_jump`` scene
(``scene_hat_tile.py``):

    media/raster/hat_fractal_jump_red.png    tiling at the red grid shift
    media/raster/hat_fractal_jump_blue.png   tiling at the blue grid shift
    media/raster/hat_fractal_jump_diff.png   per-pixel |red - blue| difference map

The two tilings are produced by ``LabbeSelingerColorModifier`` with exactly the
same parameters as the live left/right windows of the scene (grid_size,
grid_filter, shifts, color_scheme), rendered flat (no extrusion, no grid dots)
from a top-down orthographic camera whose ortho_scale matches the live window
crop: window edge (4.5 world units) / carrier scale (0.07) = 64.29 lattice
units.  Lighting/render setup (HDRI, engine, exposure) reuses the exact same
``_setup_render()`` helper ``HatTileScene.fractal_jump`` calls, so the two
pre-rendered tiles are lit identically to the live windows they sit between.
``_setup_render()`` already renders with a transparent film, so all three
PNGs keep that alpha channel as-is (transparent outside the tiling's actual
silhouette, not just outside its square footprint) instead of being
composited onto a fake opaque background.

Run headless with the workspace venv:

    .venv/bin/python video_hat_tile/fractal_jump_images.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# utils.constants derives all media paths from os.getcwd() -> must chdir first
os.chdir(HERE)
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

from objects.hat_tile import LabbeSelingerColorModifier  # noqa: E402
from perform.scene import define_materials  # noqa: E402
from video_hat_tile.scene_hat_tile import _setup_render  # noqa: E402

# --- keep these in sync with HatTileScene.fractal_jump -----------------------
RED_SHIFT = Vector([0.01, 0.01, 0])
BLUE_SHIFT = Vector([0.015, 0.015, 0])
GRID_SIZE = 100
GRID_FILTER = 5000
WINDOW = 4.5          # backdrop edge length of an upper window (world units)
CARRIER_SCALE = 0.05  # live tiling carrier object scale
ORTHO_SCALE = WINDOW / CARRIER_SCALE
RESOLUTION = 1200
OUT_DIR = os.path.join(HERE, "media", "raster")


def _set_value(mod, label, value):
    for node in mod.tree.nodes:
        if node.label == label:
            node.outputs[0].default_value = value
            return
    raise KeyError(label)


def build_carrier(name, shift):
    mod = LabbeSelingerColorModifier(color_scheme=0, shift=shift,
                                     grid_size=GRID_SIZE, grid_filter=GRID_FILTER)
    # flat picture: no extrusion, no grid dots
    _set_value(mod, "ExtrudeScale", 0.0)
    _set_value(mod, "GridRadius", 0.0)
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new(name, 'NODES')
    modifier.node_group = mod.tree
    return obj


def setup_scene():
    scene = bpy.context.scene
    for obj in list(scene.collection.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("OrthoCam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = ORTHO_SCALE
    cam = bpy.data.objects.new("OrthoCam", cam_data)
    cam.location = (0, 0, 30)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # set_hdri_background() edits bpy.data.worlds[-1] -- Blender keeps that
    # collection name-sorted (not creation-order), so a leftover default
    # "World" would sort after "HDRIWorld" and silently become the one
    # patched.  Removing every existing world first makes ours the only (and
    # so unambiguously last) one.  It also needs one node pre-built (use_nodes
    # =True auto-creates 'Background' + 'World Output'; 'Light Path' must be
    # added by hand) -- _setup_render() then rewires this into the same HDRI
    # environment-texture rig (and the same EEVEE engine/exposure/sample
    # count) that HatTileScene.fractal_jump uses for the live left/right
    # tiling windows, so these pre-rendered tiles match them.
    for stale_world in list(bpy.data.worlds):
        bpy.data.worlds.remove(stale_world)
    world = bpy.data.worlds.new("HDRIWorld")
    world.use_nodes = True
    world.node_tree.nodes.new(type='ShaderNodeLightPath')
    scene.world = world

    _setup_render()
    # _setup_render() only wires HDRI + engine; the Filmic/High Contrast look
    # itself is set once in perform.scene.initialize_blender(), which this
    # standalone script never calls.  Without it every real scene renders
    # noticeably more saturated/punchy than Blender's stock AgX default this
    # process starts with -- match it explicitly so colours correspond too.
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'

    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    return scene


def render(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    define_materials()  # provides the named base materials (hat00.., drawing, ...)
    scene = setup_scene()

    red = build_carrier("RedTiling", RED_SHIFT)
    blue = build_carrier("BlueTiling", BLUE_SHIFT)

    raw_red = os.path.join(OUT_DIR, "_raw_jump_red.png")
    raw_blue = os.path.join(OUT_DIR, "_raw_jump_blue.png")

    blue.hide_render = True
    render(scene, raw_red)
    blue.hide_render = False
    red.hide_render = True
    render(scene, raw_blue)

    from PIL import Image
    img_a = np.asarray(Image.open(raw_red).convert("RGBA"), dtype=np.float32) / 255.0
    img_b = np.asarray(Image.open(raw_blue).convert("RGBA"), dtype=np.float32) / 255.0

    # red/blue already carry the render's own alpha (film_transparent=True in
    # _setup_render() -> transparent outside the tiling's actual silhouette,
    # not just outside its square footprint) -- move the raw renders straight
    # to their final names instead of compositing them onto a fake background
    final_red = os.path.join(OUT_DIR, "hat_fractal_jump_red.png")
    final_blue = os.path.join(OUT_DIR, "hat_fractal_jump_blue.png")
    os.replace(raw_red, final_red)
    os.replace(raw_blue, final_blue)
    print("wrote", final_red)
    print("wrote", final_blue)

    # the diff picture is no longer used by the scene (fractal_jump computes
    # it procedurally in a shader now) but is kept for reference: colour is
    # the per-channel |a-b| on the straight RGB, alpha is wherever either
    # tiling was actually rendered (the alpha union of the two)
    diff_rgb = np.clip(np.abs(img_a[..., :3] - img_b[..., :3]) * 1.5, 0, 1)
    alpha_a, alpha_b = img_a[..., 3], img_b[..., 3]
    diff_alpha = alpha_a + alpha_b - alpha_a * alpha_b
    diff_rgba = np.dstack([diff_rgb, diff_alpha])
    diff_path = os.path.join(OUT_DIR, "hat_fractal_jump_diff.png")
    Image.fromarray((diff_rgba * 255).astype(np.uint8), "RGBA").save(diff_path)
    print("wrote", diff_path)

    changed = np.abs(img_a[..., :3] - img_b[..., :3]).max(axis=-1)
    print("difference stats: mean %.4f, pixels > 0.2: %.1f%%"
          % (changed.mean(), 100.0 * (changed > 0.2).mean()))


if __name__ == "__main__":
    main()
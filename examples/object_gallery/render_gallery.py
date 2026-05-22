"""Build + render every sub-scene of a Scene class, one still each.
blender -b --python run_all.py -- <module> <Class> <out_dir> [samples W H]
"""
import sys, os, importlib

REPO = "/home/jmartin/PycharmProjects/BlenderMathAnim"
sys.path.insert(0, REPO)
import bpy


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except Exception:
        return None
    for backend in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = (d.type in (backend, 'CPU'))
                return backend
        except Exception:
            pass
    return None


argv = sys.argv[sys.argv.index("--") + 1:]
module_name, class_name, out_dir = argv[:3]
samples = int(argv[3]) if len(argv) > 3 else 64
W = int(argv[4]) if len(argv) > 4 else 800
H = int(argv[5]) if len(argv) > 5 else 450
os.makedirs(out_dir, exist_ok=True)

mod = importlib.import_module(module_name)
cls = getattr(mod, class_name)
from utils.constants import FRAME_RATE

names = list(cls().sub_scenes.keys())
print("SUBSCENES:", names)
results = {}
for name in names:
    print("\n==================  building", name, " ==================")
    try:
        scene = cls()
        scene.create(name=name, resolution=[W, H], start_at_zero=True)
        sc = bpy.context.scene
        backend = enable_gpu()
        if backend:
            sc.cycles.device = 'GPU'
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        sc.render.resolution_x = W
        sc.render.resolution_y = H
        sc.render.resolution_percentage = 100
        t0 = getattr(scene, "t0", None)
        target = int(t0 * FRAME_RATE) if t0 else sc.frame_end
        target = max(sc.frame_start, min(target, sc.frame_end))
        sc.frame_set(target)
        out_png = os.path.join(out_dir, name + ".png")
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(filepath=out_png)
        ok = os.path.exists(out_png)
        results[name] = (ok, target, backend)
        print("RENDERED", name, "->", out_png, "frame", target, "gpu", backend)
    except Exception as e:
        import traceback
        traceback.print_exc()
        results[name] = (False, None, None)

print("\n==================  SUMMARY  ==================")
for n, r in results.items():
    print(f"  {n}: {r}")

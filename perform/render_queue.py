"""
Render a list of blender files as animations in the background, one after the other.

The script lists all video_* and short_* directories of the BlenderMathAnimWorkspace. After a video is chosen,
it lists the sub scenes defined in the Scene subclasses of that video (the keys of ``self.sub_scenes``; only the
scene_*.py files are read, if there are any) and asks which of them to render. The blender file of a sub scene
is expected at

    <workspace>/<video_dir>/media/blend/<SceneClass>_<sub_scene>.blend

(that is where Scene.save puts it when the scene script is run from the video directory). Sub scenes without a
blender file are skipped with a warning. Each file is rendered with ``blender -b <file> -a``, so frame range,
resolution and output directory (/filme/<video>_<scene>/) are the ones stored in the file.
The next render starts when the previous one has finished.

Usage:
    python perform/render_queue.py                          # interactive
    python perform/render_queue.py elliptic_curves "0-9"    # video and selection given
    python perform/render_queue.py --blender ~/local/blender-5.1.2-linux-x64/blender

The blender executable is taken from --blender, else from the environment variable BLENDER,
else the newest ~/local/blender-*/blender, else 'blender' on the PATH.
"""
import argparse
import ast
import glob
import os
import re
import shutil
import subprocess
import sys
import time

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USAGE = ("Usage: <enter> -> 0 | 3 -> number 3 | 0-9 or [0-9] -> numbers 0 to 9 (inclusive)"
         " | 1,4,7-9 -> a list | name -> by name")


def find_blender():
    """
    The blender executable: $BLENDER, else the newest ~/local/blender-*/blender, else 'blender' on the PATH.
    """
    if os.environ.get("BLENDER"):
        return os.environ["BLENDER"]

    def version(path):
        match = re.search(r"blender-(\d+)\.(\d+)\.(\d+)", path)
        return tuple(int(v) for v in match.groups()) if match else (0, 0, 0)

    candidates = [path for path in glob.glob(os.path.expanduser("~/local/blender-*/blender"))
                  if os.access(path, os.X_OK) and "alpha" not in path and "beta" not in path]
    if candidates:
        return max(candidates, key=version)
    return shutil.which("blender")


def video_directories(workspace=WORKSPACE_DIR):
    """
    All video_* and short_* directories of the workspace, sorted by name.
    """
    return sorted(name for name in os.listdir(workspace)
                  if (name.startswith("video_") or name.startswith("short_"))
                  and os.path.isdir(os.path.join(workspace, name)))


def sub_scenes_of_video(video_dir):
    """
    Find the Scene subclasses in the python files of a video directory and the names of their sub scenes,
    without importing the files (no bpy needed). Only the scene_*.py files are read when the directory has any,
    so helper classes in other files (e.g. riemann_surfaces.py) don't push their sub scenes into the list;
    older videos without a scene_*.py file fall back to all python files.

    :return: list of (class_name, sub_scene_name) in the order of definition
    """
    scenes = []
    paths = sorted(glob.glob(os.path.join(video_dir, "scene_*.py"))) or sorted(glob.glob(os.path.join(video_dir, "*.py")))
    for path in paths:
        try:
            with open(path) as file:
                tree = ast.parse(file.read(), filename=path)
        except (SyntaxError, UnicodeDecodeError) as error:
            print("Warning: could not parse", path, ":", error)
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in ast.walk(node):
                if not isinstance(statement, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Attribute) and target.attr == "sub_scenes"
                           and isinstance(target.value, ast.Name) and target.value.id == "self"
                           for target in statement.targets):
                    continue
                for name in sub_scene_names(statement.value):
                    scenes.append((node.name, name))
                break
    return scenes


def sub_scene_names(value):
    """
    The keys of a sub_scenes definition: OrderedDict([('name', {...}), ...]) or {'name': {...}, ...}
    """
    if isinstance(value, ast.Dict):
        return [key.value for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
    if isinstance(value, ast.Call) and value.args and isinstance(value.args[0], (ast.List, ast.Tuple)):
        names = []
        for element in value.args[0].elts:
            if isinstance(element, ast.Tuple) and element.elts and isinstance(element.elts[0], ast.Constant):
                names.append(element.elts[0].value)
        return names
    return []


def parse_selection(selection, names):
    """
    Translate a selection string into a list of entries of names.

    ""  -> [names[0]]; "3" -> [names[3]]; "0-9" or "[0-9]" -> names[0..9] (inclusive);
    "1,4,7-9" -> a list; an entry of names is selected by itself.
    """
    selection = selection.strip().strip("[]").strip()
    if len(selection) == 0:
        return names[:1]
    selected = []
    for part in selection.split(","):
        part = part.strip()
        if part in names:
            selected.append(part)
        elif "-" in part:
            first, last = (int(bound) for bound in part.split("-", 1))
            step = 1 if last >= first else -1
            selected += [names[i] for i in range(first, last + step, step)]
        else:
            selected.append(names[int(part)])
    return selected


def choose(names, prompt, selection=None):
    """
    Print the numbered names, the usage and ask for a selection (unless it is given).
    """
    for i, name in enumerate(names):
        print(i, name)
    if selection is None:
        print(USAGE)
        selection = input(prompt)
    return parse_selection(selection, names)


def render_files(blend_files, blender):
    """
    Render each blender file as an animation in the background, one after the other.

    :return: list of (file, return code, seconds)
    """
    results = []
    for i, path in enumerate(blend_files):
        print("\n==== [%d/%d] rendering %s" % (i + 1, len(blend_files), path), flush=True)
        start = time.time()
        code = subprocess.run([blender, "-b", path, "-a"]).returncode
        elapsed = time.time() - start
        print("==== finished %s with return code %d after %.0f s" % (os.path.basename(path), code, elapsed),
              flush=True)
        results.append((path, code, elapsed))
    return results


def main():
    parser = argparse.ArgumentParser(description="Render blender files of a video consecutively in the background.")
    parser.add_argument("video", nargs="?", help="video directory, e.g. video_elliptic_curves or elliptic_curves")
    parser.add_argument("selection", nargs="?", help="sub scenes, e.g. 3, 0-9, [0-9], 1,4,7-9")
    parser.add_argument("--blender", default=None, help="blender executable")
    args = parser.parse_args()

    blender = args.blender or find_blender()
    if not blender:
        sys.exit("No blender executable found; use --blender or set BLENDER.")
    print("Blender:", blender)

    videos = video_directories()
    if args.video:
        video = args.video if args.video in videos else "video_" + args.video
        if video not in videos:
            sys.exit("Unknown video directory: " + args.video)
        print("Video:", video)
    else:
        video = choose(videos, "Choose video:")[0]
    video_dir = os.path.join(WORKSPACE_DIR, video)

    scenes = sub_scenes_of_video(video_dir)
    if not scenes:
        sys.exit("No Scene subclass with sub_scenes found in " + video_dir)
    several_classes = len({cls for cls, _ in scenes}) > 1
    labels = [cls + "." + name if several_classes else name for cls, name in scenes]
    print("\nScenes of", video)
    print("A selection of several scenes renders them one after the other.")
    selected = choose(labels, "Choose scene(s):", args.selection)

    blend_dir = os.path.join(video_dir, "media", "blend")
    blend_files = []
    for label in selected:
        cls, name = scenes[labels.index(label)]
        path = os.path.join(blend_dir, cls + "_" + name + ".blend")
        if os.path.isfile(path):
            blend_files.append(path)
        else:
            print("Warning: skipping %s, no blender file %s" % (label, path))

    if not blend_files:
        sys.exit("Nothing to render.")
    print("\nRender queue:")
    for path in blend_files:
        print("  ", path)

    results = render_files(blend_files, blender)
    print("\nSummary:")
    for path, code, elapsed in results:
        print("  %-8s %6.0f s  %s" % ("ok" if code == 0 else "FAILED", elapsed, os.path.basename(path)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")

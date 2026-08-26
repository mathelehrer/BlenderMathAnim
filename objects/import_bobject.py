"""Append a finished asset out of a .blend file that does not belong in the library.

:class:`~objects.derived_objects.pencil.Pencil` is the pattern this generalises:
``BObject.from_file`` appends by name out of ``files/blend/primitives``, the
library of small, reusable props that ships with the repo. That is the wrong
home for an asset that is either large (a photogrammetry drum is 160 MB of mesh
and textures) or specific to one video - it would be carried by every checkout
for the sake of a single shot.

Such assets live next to the video that uses them, in its own ``media/blend``
folder, which is what :data:`~utils.constants.BLEND_DIR` points at (the
constants are built on ``os.getcwd()``, and a scene runs from its video
directory). :class:`ImportBObject` appends from there.
"""
import os

import bpy

from objects.bobject import BObject
from utils.constants import BLEND_DIR


class ImportBObject(BObject):
    r"""An asset appended from a .blend file, wrapped as a :class:`BObject`.

    Usage is one line - the drum of ``video_interferences`` is::

        drum = ImportBObject("drum", objects="drum", name="Drum",
                             scale=3.5, location=[0, 0, -0.72])
        drum.appear(begin_time=0, transition_time=1)

    which appends the object called ``drum`` out of ``media/blend/drum.blend``,
    parents it to an empty, and moves and scales *that* - so the asset arrives
    with its own transform intact and the scene positions it from the outside.

    **The materials come with it, and are kept.** Appending an *object* pulls
    its mesh, its materials and the images those reference along with it (the
    images in a Meshy/photogrammetry export are packed, so there is no path to
    repair). Nothing is painted over them: the wrapper is built with
    ``no_material=True`` and the parts are only given a colour if ``colors`` is
    passed, which is the difference between this class and ``from_file`` -
    ``from_file`` exists to *recolour* a grey primitive, this one exists to
    keep a textured asset as it is.

    **What is appended.** ``objects`` names them; leaving it out takes every
    object of a type in ``types``, which defaults to meshes alone. That default
    is what makes ``ImportBObject("drum")`` do the right thing on an export
    that also carries the artist's ``Camera`` and ``Light`` - they would
    otherwise be appended into the scene and light it.

    **Linking.** The appended objects are deliberately *not* linked into the
    scene collection here; :meth:`BObject.appear` does that, at the time the
    asset is supposed to show up, exactly as it does for an object built from
    scratch. Until then they are unlinked datablocks, so a scene that never
    calls ``appear`` never has them in shot.

    :param filename: name of the .blend in ``directory``, with or without the
        extension. An absolute path is taken as it stands.
    :param objects: name of the object to append, or a list of names. ``None``
        (the default) appends every object whose type is in ``types``.
    :param directory: where to look, :data:`~utils.constants.BLEND_DIR`
        (``<video>/media/blend``) by default.
    :param types: object types to keep when ``objects`` is not given. ``None``
        keeps all of them.
    :param colors: one palette colour per appended object, applied in the order
        the objects come back. Passing this *replaces* the asset's own
        materials; the default ``None`` keeps them.
    :param kwargs: forwarded to :class:`BObject` - ``name``, ``location``,
        ``rotation_euler``, ``scale``, ... all act on the wrapper, and so on
        the asset as a whole.
    """

    def __init__(self, filename, objects=None, directory=None, types=('MESH',),
                 colors=None, **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name', None)
        self.filepath = self.resolve(filename, directory)

        appended = self.append(self.filepath, objects=objects, types=types)
        if name is None:
            name = os.path.splitext(os.path.basename(self.filepath))[0]

        self.parts = []
        for i, obj in enumerate(appended):
            part_name = name + '_' + obj.name
            if colors is None:
                # no_material: the appended material is the point of the asset
                self.parts.append(BObject(obj=obj, name=part_name,
                                          no_material=True))
            else:
                color = colors[i] if i < len(colors) else colors[-1]
                self.parts.append(BObject(obj=obj, name=part_name, color=color))

        super().__init__(children=self.parts, name=name, no_material=True,
                         **kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def resolve(filename, directory=None):
        """The full path of ``filename``, and an error naming what is there."""
        path = filename if filename.endswith('.blend') else filename + '.blend'
        if not os.path.isabs(path):
            path = os.path.join(BLEND_DIR if directory is None else directory,
                                path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no blend file %s. ImportBObject looks in %s, the media/blend "
                "folder of the video the scene is run from"
                % (path, os.path.dirname(path)))
        return path

    @classmethod
    def contents(cls, filename, directory=None):
        """The object names in a .blend, without appending anything.

        A .blend that came out of an exporter rarely names its parts the way
        the scene would like to call them, and this is how to find out what
        they *are* called::

            print(ImportBObject.contents("drum"))   # ['Light', 'drum', 'Camera']
        """
        path = cls.resolve(filename, directory)
        with bpy.data.libraries.load(path, link=False) as (source, _):
            return list(source.objects)

    @staticmethod
    def append(filepath, objects=None, types=('MESH',)):
        """Append objects from ``filepath`` and return the new bpy objects.

        ``bpy.data.libraries.load`` rather than ``bpy.ops.wm.append``: the
        operator links what it appends into the active collection there and
        then, which takes the timing of the asset's appearance out of the
        scene's hands, and it needs a context that a headless run does not
        always have.

        Selecting by type can only happen *after* the load, since the library
        header lists names and not types - so a camera in the file is appended
        and then removed again, which costs nothing.
        """
        with bpy.data.libraries.load(filepath, link=False) as (source, target):
            available = list(source.objects)
            if objects is None:
                wanted = available
            else:
                wanted = [objects] if isinstance(objects, str) else list(objects)
                missing = [obj for obj in wanted if obj not in available]
                if missing:
                    raise KeyError("%s holds no object called %s. It holds %s"
                                   % (os.path.basename(filepath),
                                      ", ".join(missing), ", ".join(available)))
            target.objects = wanted

        appended = [obj for obj in target.objects if obj is not None]
        if objects is None and types is not None:
            keep = [obj for obj in appended if obj.type in types]
            for obj in appended:
                if obj not in keep:
                    bpy.data.objects.remove(obj, do_unlink=True)
            appended = keep
        if not appended:
            raise KeyError("nothing to append from %s"
                           % os.path.basename(filepath))
        return appended

    # ------------------------------------------------------------------
    def part(self, name):
        """The appended part whose original name is (or ends in) ``name``."""
        for part in self.parts:
            if part.ref_obj.name == name or part.ref_obj.name.endswith('_' + name):
                return part
        raise KeyError("%s has no part %s, only %s"
                       % (self.name, name,
                          ", ".join(p.ref_obj.name for p in self.parts)))

    def __getitem__(self, index):
        return self.parts[index]

    def __len__(self):
        return len(self.parts)

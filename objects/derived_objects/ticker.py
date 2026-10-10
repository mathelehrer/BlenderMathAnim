import numpy as np

from interface import ibpy
from objects.bobject import BObject
from objects.plane import Plane
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME


class Ticker(BObject):
    """A news ticker: a line of formulas running from right to left across a band.

    The entries are typeset as one line, separated by ``separator``, and
    run at constant speed from the band's right end until the last of them
    has left at its left end. Nothing clips the line, so the band is meant to
    span the frame - its edges are where the text enters and leaves.

    Band and line are children of an empty that carries the location; the
    band stands in the x-z plane, facing the camera along -y.

    :param entries: LaTeX strings, one per item.
    :param width: length of the band.
    :param height: height of the band.
    :param separator: LaTeX between two entries.
    :param color: color of the text.
    :param band_color: color of the band.
    :param band_alpha: opacity of the band.
    :param text_size: text size of the line.
    :param text_shift: vertical offset of the line in the band, to centre
        the glyphs, whose anchor is not their middle.

    Example::

        ticker = Ticker([r"a_0=1.0", r"a_1=-2.2499997"], width=15,
                        location=(0, 0, -3.8))
        ticker.appear(begin_time=0)
        ticker.run(begin_time=0, transition_time=20)
    """

    def __init__(self, entries, width=15, height=0.4, separator=r"\qquad ", color="text",
                 band_color="joker", band_alpha=0.15, text_size="small", text_shift=-0.05,
                 **kwargs):
        self.kwargs = kwargs
        name = self.get_from_kwargs('name', 'Ticker')
        self.width = width
        self.band_alpha = band_alpha
        self.text_shift = text_shift

        self.band = Plane(u=[-width / 2, width / 2], v=[-height / 2, height / 2],
                          rotation_euler=[np.pi / 2, 0, 0], color=band_color, resolution=1,
                          name=name + "Band")
        self.line = SimpleTexBObject(separator.join(entries), text_size=text_size, color=color,
                                     aligned="left", location=(width / 2, -0.01, text_shift),
                                     name=name + "Line")
        # the line's extent along x, from its glyphs' boxes in its own frame
        letters = self.line.letters
        self.length = self.line.ref_obj.scale.x * (
                max(l.ref_obj.location.x + max(c[0] for c in l.ref_obj.bound_box) for l in letters)
                - min(l.ref_obj.location.x + min(c[0] for c in l.ref_obj.bound_box) for l in letters))

        super().__init__(children=[self.band, self.line], name=name, no_material=True, **kwargs)

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, **kwargs):
        """Fade the band in; the line only shows once it runs."""
        super().appear(begin_time=begin_time, transition_time=transition_time)
        self.band.appear(alpha=self.band_alpha, begin_time=begin_time,
                         transition_time=transition_time)
        return begin_time + transition_time

    def run(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        """Run the line through once, entering on the right and gone on the left.

        :return: the time the last entry has left the band.
        """
        self.line.write(begin_time=begin_time, transition_time=0, writing=False)
        self.line.move_to(target_location=(-self.width / 2 - self.length, -0.01, self.text_shift),
                          begin_time=begin_time, transition_time=transition_time)
        # a ticker runs at constant speed
        ibpy.set_linear_fcurves(self.line)
        return begin_time + transition_time

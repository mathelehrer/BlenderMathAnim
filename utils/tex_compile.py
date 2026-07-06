"""
latex -> dvi -> svg compilation helpers.

Moved verbatim out of ``objects/tex_bobject.py`` so that modules that only
need the compile pipeline (e.g. ``objects/token_mapping.py``) can use it
without importing the bpy-dependent tex object machinery.  All results are
cached in ``TEX_DIR``/``SVG_DIR`` keyed by :func:`hashed_tex`.
"""

import hashlib
import os

from utils.constants import SVG_DIR, TEX_DIR, TEX_TEXT_TO_REPLACE


def tex_to_svg_file(expression, template_tex_file, typeface, text_only, recreate=False):
    path = os.path.join(
        SVG_DIR,
        # tex_title(expression, typeface)
        hashed_tex(expression, typeface)
    ) + ".svg"
    if not recreate and os.path.exists(path):
        return path

    tex_file = generate_tex_file(expression, template_tex_file, typeface, text_only, recreate)
    dvi_file = tex_to_dvi(tex_file, recreate)
    return dvi_to_svg(dvi_file, recreate)


def get_null():
    if os.name == "nt":
        return "NUL"
    return "/dev/null"


def dvi_to_svg(dvi_file, recreate):
    """
    Converts a dvi, which potentially has multiple slides, into a
    directory full of enumerated svgs corresponding with these slides.
    Returns a list of PIL Image objects for these images sorted as they
    where in the dvi
    """

    result = dvi_file.replace(".dvi", ".svg")
    result = result.replace("tex", "svg")  # change directory for the svg file
    print('svg: ', result)
    if recreate or not os.path.exists(result):
        commands = [
            "dvisvgm",
            dvi_file,
            "-n",
            "-v",
            "3",
            "-o",
            result
        ]
        os.system(" ".join(commands))
    return result


def hashed_tex(expression, typeface):
    string = expression + typeface
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]


def generate_tex_file(expression, template_tex_file, typeface, text_only, recreate):
    result = os.path.join(
        TEX_DIR,
        # tex_title(expression, typeface)
        hashed_tex(expression, typeface)
    ) + ".tex"

    if recreate or not os.path.exists(result):
        print("Writing \"%s\" to %s" % (
            "".join(expression), result
        ))
        with open(template_tex_file, "r") as infile:
            body = infile.read()
            # I add an H to every expression to give a common reference point
            # for all expressions, then hide the H character. This is necessary
            # for consistent alignment of tex curves in blender, because
            # blender's import svg bobject sets the object's origin depending
            # on the expression itself, not according to a typesetting reference
            # frame.
            if text_only:
                expression = 'H ' + expression
            else:
                expression = '\\text{H} ' + expression
            body = body.replace(TEX_TEXT_TO_REPLACE, expression)
        with open(result, "w") as outfile:
            outfile.write(body)
    return result


def tex_to_dvi(tex_file, recreate):
    result = tex_file.replace(".tex", ".dvi")
    if recreate or not os.path.exists(result):
        commands = [
            "latex",
            "-interaction=batchmode",
            "-halt-on-error",
            "-output-directory=" + TEX_DIR,
            tex_file
        ]
        exit_code = os.system(" ".join(commands))
        if exit_code != 0:
            latex_output = ''
            log_file = tex_file.replace(".tex", ".log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    latex_output = f.read()
            raise Exception(
                "Latex error converting to dvi. "
                "See log output above or the log file: %s" % log_file)
    return result

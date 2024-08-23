import os
from copy import deepcopy

from utils.constants import TEMPLATE_TEX_FILE, TEX_DIR, TEX_TEXT_TO_REPLACE, IMG_DIR


class ImageCreator:
    def __init__(self,text,count,prefix=''):
        name = prefix+"_" + f'{count:03}'

        template = deepcopy(TEMPLATE_TEX_FILE)
        if not os.path.exists(template):
            raise Warning(r'Can\'t find template tex file for that font.')

        self.path = self.tex_to_png_file(text, template, 'default', name)
        print(self.path)

    def get_image_path(self):
        return self.path

    def tex_to_png_file(self, expression, template_tex_file, typeface, name):
        path = os.path.join(
            IMG_DIR,
            name
        ) + ".png"
        if os.path.exists(path):
            return path

        tex_file = self.generate_tex_file(expression, template_tex_file, typeface, name)
        dvi_file = self.tex_to_dvi(tex_file)
        svg_file = self.dvi_to_svg(dvi_file)
        return self.svg_to_png(svg_file)

    def generate_tex_file(self, expression, template_tex_file, typeface, name):
        result = os.path.join(
            TEX_DIR,
            name
        ) + ".tex"

        if not os.path.exists(result):
            print("Writing \"%s\" to %s" % (
                "".join(expression), result
            ))
            with open(template_tex_file, "r") as infile:
                body = infile.read()
                body = body.replace(TEX_TEXT_TO_REPLACE, expression)
            with open(result, "w") as outfile:
                outfile.write(body)
        return result

    def tex_to_dvi(self, tex_file):
        result = tex_file.replace(".tex", ".dvi")
        if not os.path.exists(result):
            commands = [
                "latex",
                "-interaction=batchmode",
                "-halt-on-error",
                "-output-directory=" + TEX_DIR,
                tex_file  # ,
                # ">",
                # get_null()
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

    def dvi_to_svg(self, dvi_file):
        """
        Converts a dvi, which potentially has multiple slides, into a
        directory full of enumerated svgs corresponding with these slides.
        Returns a list of PIL Image objects for these images sorted as they
        where in the dvi
        """

        result = dvi_file.replace(".dvi", ".svg")
        result = result.replace("tex", "svg")  # change directory for the svg file
        print('svg: ', result)
        if not os.path.exists(result):
            commands = [
                "dvisvgm",
                dvi_file,
                "-n",
                "-v",
                "3",
                "-o",
                result
                # Not sure what these are for, and it seems to work without them
                # so commenting out for now
                # ,
                # ">",
                # get_null()
            ]
            os.system(" ".join(commands))
        return result

    def svg_to_png(self, svg_file):
        """
        Converts a svg to png
        """

        result = svg_file.replace(".svg", ".png")
        result = result.replace("svg", "raster")  # change directory for the svg file
        print('png: ', result)

        # if not os.path.exists(result):
        #     commands = [
        #         "inkscape",
        #         "-z",
        #         "-f",
        #         svg_file,
        #         "-w 1024",
        #         "-e",
        #         result
        #     ]
        #     os.system(" ".join(commands))

        # adjusted to new inkscape version

        if not os.path.exists(result):
            commands = [
                "inkscape",
                svg_file,
                "-w 1024",
                "-o",
                result
            ]
            os.system(" ".join(commands))
        return result


if __name__ == '__main__':
    ImageCreator()

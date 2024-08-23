import mathutils
import numpy as np

from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME


class BMatrix(SimpleTexBObject):

    def similar(self, a, b, EPS):
        if np.abs(a - b) < EPS:
            return True
        else:
            return False

    def __init__(self, entries, pre_word="",after_word="", **kwargs):
        self.kwargs = kwargs
        self.entries = entries

        if 'mapping' in kwargs:
            mapping = kwargs['mapping']
        else:
            mapping = None

        name = self.get_from_kwargs('name','Matrix')

        rows = entries.shape[0]
        cols = entries.shape[1]
        structure_string = "{"
        for i in range(cols):
            structure_string += "c "
        structure_string += "}"
        entry_string = ""
        for row in range(rows):
            row_string = ""
            for col in range(cols):
                if col > 0:
                    sep = "&"
                else:
                    sep = ""
                raw_entry = entries[row, col]
                entry = raw_entry
                if mapping:
                    for key in mapping:
                        if self.similar(raw_entry, key, 0.01):
                            entry = mapping[key]
                            break
                row_string += sep + str(entry)
            if row < rows - 1:
                row_string += r"\\"
            entry_string += row_string
        latex = pre_word + r"\left(\begin{array}" + structure_string + entry_string + r"\end{array}\right)"+after_word
        super().__init__(latex, name=name, **kwargs)

    def get_mathutils_matrix(self):
        return self.entries

    def __str__(self):
        rows = self.entries.shape[0]
        cols = self.entries.shape[1]
        entry_string = "["
        for row in range(rows):
            row_string = "["
            for col in range(cols):
                if col > 0:
                    sep = ","
                else:
                    sep = ""
                raw_entry = self.entries[row, col]
                entry = np.round(raw_entry * 10) / 10
                row_string += sep + str(entry)
            row_string += r"]"
            entry_string += row_string + "]"
        return entry_string

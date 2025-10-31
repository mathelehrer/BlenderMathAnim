import os

import numpy as np

from objects.bobject import BObject
from objects.tex_bobject import SimpleTexBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE
from utils.string_utils import find_all
from utils.utils import flatten


class FunctionText:
    def __init__(self, header, content, lines):
        """
        :param header: the header of the function
        :param content: the body of the function
        :param lines: the range of lines within the class
        """
        self.header = header
        self.content = content
        self.lines = lines
        self.empty_lines=[]
        self.parse()

    def parse(self):
        pass

    def number_of_lines(self):
        n_lines = 1  # header
        n_lines += len(self.content)
        return n_lines

    def show(self):
        print(self.header)
        for line in self.content:
            print(line)


def prepare_for_latex(text):
    tmp = text.replace('_', r'\_')
    tmp = tmp.replace('{', r'\{')
    tmp = tmp.replace('}', r'\}')
    tmp = tmp.replace('%', r'\%')
    return tmp


def get_leading_spaces(text):
    spaces = 0
    while text[0] == ' ':
        spaces += 1
        text = text[1:]
    return spaces

def prepare_colors(text, isheader=False):
    builtin = ['import', 'as', 'from', 'def', 'class', 'if', 'not', 'return', 'else', 'None', 'and', 'or', 'elif',
               'for', 'in', 'while', 'yield']
    builtin2 = ['super', 'range', 'len', 'print', 'list', 'int', 'open', 'enumerate']
    self = 'self'
    override = '__'
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    operator = ['>','<','+','-','*','/','**','%']
    text = text.strip()  # remove leading spaces
    words = text.split(' ')
    n_letters = 0
    for word in words:
        n_letters += len(word)

    colors = ['text'] * n_letters
    pos = 0
    for word in words:
        if ',' in word:
            idx = word.find(',')
            colors[pos + idx] = 'code_builtin'
        if override in word and '(' in word:  # exclude __main__ from being colored
            ids = list(find_all(word, "__"))
            print(ids)
            for i in range(ids[0], ids[1] + 2):
                colors[pos + i] = 'code_override'
        elif isheader and '(' in word:
            id0 =word.find('(')
            for i in range(0,id0):
                colors[pos+i]='example'
        if self in word:
            index = word.find(self)
            for i in range(index, index + 4):
                colors[pos + i] = 'code_self'
        if "'" in word:
            ids = list(find_all(word, "'"))
            for i in range(ids[0], ids[1] + 1):
                colors[pos + i] = 'code_string'
        for digit in digits:
            if digit in word:
                indices = [pos for pos, char in enumerate(word) if char == digit]
                for index in indices:
                    colors[pos + index] = 'code_number'
                    if index < len(word) - 1:
                        if word[index + 1] in ['j', '.']:
                            colors[pos + index + 1] = 'code_number'
        for bi in builtin:
            if bi in word:
                index = word.find(bi)
                isword = True
                if index > 0:  # make sure there are not letters before the builtin word
                    if word[index - 1] not in ['=', '(', ' ']:
                        isword = False
                if len(bi) < len(word) - index:  # make sure there are not letters after the builtin word
                    if word[index +len(bi)] not in ['=', ')',':']:
                        isword = False
                if isword:
                    for i in range(index, index + len(bi)):
                        colors[pos + i] = 'code_builtin'
        for bi in builtin2:
            if bi in word:
                index = word.find(bi)
                isword = True
                if index > 0:
                    if word[index - 1] not in ['=', '(', ' ']:
                        isword = False
                if isword:
                    for i in range(index, index + len(bi)):
                        colors[pos + i] = 'code_builtin2'
        if '=' in word:
            if not isheader:  # keywords are only highlighted in function calls and not in function definitions
                index = word.find("=")
                if index > 0:
                    if word[index - 1] not in operator :
                        bracket_index = word.find('(')
                        if -1 < bracket_index < index:
                            start = bracket_index + 1
                        else:
                            start = 0
                        for i in range(start, index):
                            colors[pos + i] = 'code_keyword'
        pos += len(word)

    return colors


class ClassText:
    def __init__(self, header, content):
        self.header = header
        self.functions = []
        self.parse(content)

    def parse(self, content):
        fcn = []
        header_line=0 # for functions without def (main)
        for l,line in enumerate(content):
            if 'def' in line:
                if not fcn:
                    fcn = [line]
                    header_line = l
                else:
                    self.functions.append(FunctionText(fcn[0], fcn[1:],lines=range(header_line,l))) # exclusive right boundary
                    fcn = [line]
                    header_line = l
            else:
                fcn.append(line)
        if fcn:
            self.functions.append(FunctionText(fcn[0], fcn[1:],lines=range(header_line,len(content)))) # exclusive right boundary
    def number_of_lines(self):
        n_lines = 1  # header + empty line
        for fcn in self.functions:
            n_lines += 1
            n_lines += fcn.number_of_lines()
        return n_lines

    def show(self):
        print(self.header)
        for function in self.functions:
            function.show()

    def create_simple_tex_b_objects(self, **kwargs):
        tex_header = prepare_for_latex(self.header)
        colors = prepare_colors(self.header, isheader=True)
        objects = [SimpleTexBObject(r"\text{\tt " + tex_header + "}", color=colors, **kwargs)]
        for fcn in self.functions:
            fcn_tex_header = prepare_for_latex(fcn.header)
            colors = prepare_colors(fcn.header, isheader=True)
            objects.append(SimpleTexBObject(r"\text{\tt " + fcn_tex_header + "}", color=colors, **kwargs))
            for line in fcn.content:
                tex_line = prepare_for_latex(line)
                colors = prepare_colors(line)
                objects.append(SimpleTexBObject(r"\text{\tt " + tex_line + "}", color=colors, **kwargs))
        return objects

    def indents(self):
        indents = [0]
        for fcn in self.functions:
            indents.append(get_leading_spaces(fcn.header))
            for line in fcn.content:
                indents.append(get_leading_spaces(line))
        return indents

    def headers(self):
        headers = [0]
        line = 0
        for fcn in self.functions:
            line += 1
            headers.append(line)
            line += len(fcn.content)
        return headers


class CodeParser(BObject):
    '''
    The CodeParser has limitations
    1.) no comments
    '''

    def __init__(self, filename, **kwargs):
        with open(filename) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.replace("\n", "")

        # split classes
        imports = []
        remaining_lines = []

        for line in lines:
            if len(line) == 0:
                line='!'
            if 'import' in line or 'from' in line:
                imports.append(line)
            else:
                remaining_lines.append(line)

        self.classes = []
        cls = None
        remaining_lines2 = []
        for line in remaining_lines:
            if not cls and not 'class' in line:
                remaining_lines2.append(line)
            elif not cls and 'class' in line:
                cls = [line]  # add first line to class
            elif cls:
                if line[0] == ' ' or line[0]=='!':  # tabs in the file get converted in to spaces
                    cls.append(line)
                elif 'class' in line:
                    self.classes.append(ClassText(cls[0], cls[1:]))
                    cls = [line]
                else:
                    self.classes.append(ClassText(cls[0], cls[1:]))
                    cls = None
                    remaining_lines2.append(line)
            else:
                remaining_lines2.append(line)

        # if len(remaining_lines2) > 0:
        #     self.classes.append(ClassText(remaining_lines2[0], remaining_lines2[1:]))
        #     self.remaining_lines = remaining_lines2

        if cls is not None:
            self.classes.append(ClassText(cls[0], cls[1:]))

        self.indents = []
        self.headers = []
        for cls in self.classes:
            self.indents.append(cls.indents())
            self.headers.append(cls.headers())

        self.objects = self.create_simple_tex_b_objects(**kwargs)
        super().__init__(objects=flatten(self.objects), **kwargs)

    def create_simple_tex_b_objects(self, **kwargs):
        objects = []
        for cls in self.classes:
            objects.append(cls.create_simple_tex_b_objects(**kwargs))
        return objects

    def write(self, code_display, class_index=0, function=None, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME,
              **kwargs):

        if 'back' in kwargs:
            back = kwargs.pop('back')
        else:
            back=False

        if 'indent' in kwargs:
            indent = kwargs.pop('indent')
        else:
            indent = 0
        if function is None:
            lines = range(0, len(self.objects[class_index]))
            all = True

        elif function is not None:
            if len(self.classes)>0:
                cls = self.classes[class_index]
                fcn = cls.functions[function]
                lines = fcn.lines
                all = False
            else:
                all = True

        if len(self.indents)>0:
            indents = self.indents[class_index]
        else:
            indents = 0

        # calculate time per char
        sum = 0
        chars = []
        if all:
            chars.append(len(cls.header.replace(" ","")))
            sum+=chars[-1]
            for fcn in cls.functions:
                chars.append(len(fcn.header.strip().replace(" ","")))
                sum+=chars[-1]
                for line in fcn.content:
                    chars.append(len(line.strip().replace(" ","")))
                    sum += chars[-1]
        else:
            fcn = cls.functions[function]
            if function == 0:
                chars.append(len(cls.header.strip().replace(" ","")))
                lines =range(lines[0],lines[-1]+2) # extend range to include the class declaration
                sum=chars[-1]
            else:
                lines = range(lines[0]+1,lines[-1]+2) # shift the range by one accounting for the class declaration
            chars.append(len(fcn.header.strip().replace(" ","")))
            for line in fcn.content:
                chars.append(len(line.strip().replace(" ","")))
                sum += chars[-1]

        dt = np.minimum(1 / 50, transition_time / sum)
        t0=begin_time
        for i, line in enumerate(lines):
            if len(self.objects[class_index][line].letters)==1:# work around to create empty lines
               code_display.add_empty_line()
            elif back:
                t0 = code_display.write_text_in_back(self.objects[class_index][line], begin_time=t0,
                                             transition_time=chars[i] * dt,
                                             indent=indent + indents[line] * 0.15)
            else:
                t0 = code_display.write_text(self.objects[class_index][line], begin_time=t0, transition_time=chars[i]*dt,
                                    indent=indent + indents[line] * 0.15)
            t0+=2/FRAME_RATE # add two frames at the end of each line
        return begin_time + dt*sum

    def write_in_back(self, code_display, class_index=0, function=None, begin_time=0,
                      transition_time=DEFAULT_ANIMATION_TIME,**kwargs):
        return self.write(code_display, class_index=class_index, function=function, begin_time=begin_time,
                      transition_time=transition_time,back=True,**kwargs)

    def appear(self, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME, clear_data=False, silent=False):
        n_lines = 0
        for cls in self.objects:
            n_lines += len(cls)
        dt = transition_time / n_lines
        t0 = begin_time

        for cls in self.objects:
            for line in cls:
                t0 = line.write(begin_time=t0, transition_time=dt)

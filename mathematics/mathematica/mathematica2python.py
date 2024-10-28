import os

from sympy.parsing.mathematica import parse_mathematica

from utils.constants import LOC_FILE_DIR

if __name__ == '__main__':
    with open(os.path.join(LOC_FILE_DIR,"fittingfunctions.dat")) as data:
        raw = ""
        for line in data:
            raw+=line

        raw = raw.replace("<|","")
        raw = raw.replace("|>","")
        raw = raw.replace("*^","e")
        parts = raw.split(",")
        dict = {}
        for part in parts:
            sub_parts = part.split("->")
            dict[int(sub_parts[0])]=parse_mathematica(sub_parts[1])
        print(dict)
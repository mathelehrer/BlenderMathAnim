import os
from random import choice

from mathematics.parsing.parser import parse_permutation
from utils.constants import GLOBAL_DATA_DIR


def create_coxB3_word_dictionary():
    dict = {}
    with open(os.path.join(GLOBAL_DATA_DIR,"coxB3.dat"), "r") as f:
        content = f.read()
        for line in content.splitlines():
            parts = line.split(":")
            parts = parts[1].split("->")
            word = parts[0].strip()
            # if len(word)==0:
            #     word=r"\varepsilon"
            permutation = parse_permutation(parts[1])
            dict[permutation]=word
    return dict

def create_coxB3_permutation_dictionary():
    dict = {}
    with open(os.path.join(GLOBAL_DATA_DIR, "coxB3.dat"), "r") as f:
        content = f.read()
        for line in content.splitlines():
            parts = line.split(":")
            parts = parts[1].split("->")
            word = parts[0].strip()
            if len(word)==0:
                word=r"\varepsilon"
            permutation = parse_permutation(parts[1])
            dict[word] = permutation
    return dict

def random_coxB3_permutation():
    return choice(permutations)

word2permutation = create_coxB3_permutation_dictionary()
permutations = list(word2permutation.values())

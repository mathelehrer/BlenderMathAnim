"""
I want to convert a number between 0 and 100000
into five tuples of the form (-10,8,-3,7,3)
Is there a way to do this to collect all possible five tuples

Try the reverse way first
"""
from mathematics.mathematica.mathematica import tuples


def tuple2index(tup, base):
    b = 1
    s = 0
    for t in reversed(list(tup)):
        s += b * t
        b *= base
    return s


def index2tuple(index, base):
    digs = [0, 0, 0, 0, 0]
    b = base ** 4
    for i in range(5):
        digs[i] = index // b
        index -= digs[i] * b
        b = int(b / base)
    return tuple(digs)


if __name__ == '__main__':
    digits = [-1, 0, 1]
    tups = tuples(digits, 5)
    set = set(tuple2index(t, len(digits)) for t in tups)
    print(len(set))

    for s in set:
        print(s, index2tuple(s, len(digits)))

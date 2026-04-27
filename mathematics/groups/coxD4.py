from mathematics.groups.coxeter_group import CoxeterGroup4D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector, EpsilonTensor

COXD4_SIGNATURES = {
    "x3o3o *b3o": [1, 0, 0, 0],
    "o3o3x *b3o": [1, 0, 0, 0],
    "o3o3o *b3x": [1, 0, 0, 0],
    "o3x3o *b3o": [0, 1, 0, 0],
    "x3o3x *b3o": [1, 0, 1, 0],
    "x3o3o *b3x": [1, 0, 1, 0],
    "o3o3x *b3x": [1, 0, 1, 0],
    "x3x3o *b3o": [1, -1, 0, 0],
    "o3x3x *b3o": [1, -1, 0, 0],
    "o3x3o *b3x": [1, -1, 0, 0],
    "x3o3x *b3x": [1, 0, 1, 1],
    "x3x3x *b3o": [1, -1, 1, 0],
    "x3x3o *b3x": [1, -1, 1, 0],
    "o3x3x *b3x": [1, -1, 1, 0],
    "x3x3x *b3x": [1, -1, 1, 1],
}

COXD4_SEEDS ={
    "x3o3x *b3x": FVector.parse("[1,-3,1,1]"),
    "x3o3x *b3o": FVector.parse("[1,-2,0,1]"),
    "x3o3o *b3x": FVector.parse("[1,-2,0,1]"),
    "o3o3x *b3x": FVector.parse("[1,-2,0,1]"),
    "x3o3o *b3o": FVector.parse("[1,-1,0,0]"),
    "o3o3x *b3o": FVector.parse("[1,-1,0,0]"),
    "o3o3o *b3x": FVector.parse("[1,-1,0,0]"),
    "x3x3x *b3x": FVector.parse("[1,-5,1,1]"),
    "x3x3x *b3o": FVector.parse("[1,-4,0,1]"),
    "x3x3o *b3x": FVector.parse("[1,-4,0,1]"),
    "o3x3x *b3x": FVector.parse("[1,-4,0,1]"),
    "x3x3o *b3o": FVector.parse("[1,-3,0,0]"),
    "o3x3x *b3o": FVector.parse("[1,-3,0,0]"),
    "o3x3o *b3x": FVector.parse("[1,-3,0,0]"),
    "o3x3o *b3o": FVector.parse("[0,2,0,0]"),
}

_DEFAULT_PATH = "../geometry/data/"


class CoxD4(CoxeterGroup4D):
    def __init__(self, path=None):
        self.size=192
        zero = QR.from_integers(0, 1, 0, 1)
        one = QR.from_integers(1, 1, 0, 1)
        half = QR.from_integers(1, 2, 0, 1)

        normals = [
            FVector([one, zero, zero, zero]),
            FVector([half, half, half, half]),
            FVector([zero, zero,one, zero]),
            FVector([zero,zero,zero,one]),
            ]

        identity = FMatrix([[one, zero, zero, zero], [zero, one, zero, zero],
                            [zero, zero, one, zero], [zero, zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]
        epsilon = EpsilonTensor(4)

        super().__init__("coxD4", normals, generators, zero, epsilon, identity,
                         path=path or _DEFAULT_PATH,seeds=COXD4_SEEDS,signatures=COXD4_SIGNATURES)

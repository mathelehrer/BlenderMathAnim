from mathematics.groups.coxeter_group import CoxeterGroup4D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector, EpsilonTensor

COXH4_SIGNATURES = {
    "x3o3o5o": [1, 0, 0, 0],
    "o3o3o5x": [0, 0, 0, 1],
    "o3x3o5o": [0, 1, 0, 0],
    "o3o3x5o": [0, 0, 1, 0],
    "x3x3o5o": [1, -1, 0, 0],
    "x3o3o5x": [1, 0, 0, -1],
    "o3o3x5x": [0, 0, 1, -1],
    "o3x3o5x": [0, 1, 0, 1],
    "o3x3x5o": [0, 1, -1, 0],
    "x3o3x5o": [1, 0, 1, 0],
    "x3x3x5o": [1, -1, 1, 0],
    "x3x3o5x": [1, -1, 0, -1],
    "x3o3x5x": [1, 0, 1, -1],
    "o3x3x5x": [0, 1, -1, 1],
    "x3x3x5x": [1, -1, 1, -1]
}

COXH4_SEEDS ={
    "x3o3o5o": FVector.parse("[1,0,1/2+1/2*r5,-3/2-1/2*r5]"),
    "o3o3o5x": FVector.parse("[0,0,-3-1*r5,3+1*r5]"),
    "o3x3o5o": FVector.parse("[0,0,-1-1*r5,3+r5]"),
    "o3o3x5o": FVector.parse("[0,1,5/2+3/2*r5,-7/2-3/2*r5]"),
    "x3x3o5o": FVector.parse("[1,0,3/2+3/2*r5,-9/2-3/2*r5]"),
    "x3o3o5x": FVector.parse("[1,0,7/2+3/2*r5,-9/2-3/2*r5]"),
    "o3o3x5x": FVector.parse("[0,1,11/2+5/2*r5,-13/2-5/2*r5]"),
    "o3x3o5x": FVector.parse("[0,0,-4-2*r5,6+2*r5]"),
    "o3x3x5o": FVector.parse("[0,-1,-7/2-5/2*r5,13/2+5/2*r5]"),
    "x3o3x5o": FVector.parse("[1,1,3+2*r5,-5-2*r5]"),
    "x3x3x5o": FVector.parse("[1,1,4+3*r5,-8-3*r5]"),
    "x3x3o5x": FVector.parse("[1,0,9/2+5/2*r5,-15/2-5/2*r5]"),
    "x3o3x5x": FVector.parse("[1,1,6+3*r5,-8-3*r5]"),
    "o3x3x5x": FVector.parse("[0,-1,-13/2-7/2*r5,19/2+7/2*r5]"),
    "x3x3x5x": FVector.parse("[1,1,7+4*r5,-11-4*r5]"),
}

_DEFAULT_PATH = "../geometry/data/"

class CoxH4(CoxeterGroup4D):
    def __init__(self, path=None):
        self.size = 14400
        zero = QR.from_integers(0, 1, 0, 1)
        one = QR.from_integers(1, 1, 0, 1)
        half = QR.from_integers(1, 2, 0, 1)

        normals = [
            FVector([one, zero, zero, zero]),
            FVector([half, half, half, half]),
            FVector([zero, one, zero, zero]),
            FVector([zero,
                     QR.from_integers(1, 4, 1, 4),
                     -half,
                     QR.from_integers(1, 4, -1, 4)]),
        ]

        identity = FMatrix([[one, zero, zero, zero], [zero, one, zero, zero],
                            [zero, zero, one, zero], [zero, zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]
        epsilon = EpsilonTensor(4)

        super().__init__("coxH4", normals, generators, zero, epsilon, identity,
                         path=path or _DEFAULT_PATH,seeds=COXH4_SEEDS,signatures=COXH4_SIGNATURES)

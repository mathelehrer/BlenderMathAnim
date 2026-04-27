from mathematics.groups.coxeter_group import CoxeterGroup4D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector, EpsilonTensor

COXF4_SIGNATURES = {
    "x3o4o3o": [1, 0, 0, 0],
    "o3x4o3o": [0, 1, 0, 0],
    "o3o4o3x": [1, 0, 0, 0],
    "o3o4x3o": [0, 1, 0, 0],
    "x3x4o3o": [1, -1, 0, 0],
    "x3o4x3o": [1, 0, 1, 0],
    "o3x4o3x": [1, 0, 1, 0],
    "o3o4x3x": [1, -1, 0, 0],
    "x3o4o3x": [1, 0, 0, -1],
    "o3x4x3o": [0, 1, -1, 0],
    "x3x4x3o": [1, -1, 1, 0],
    "o3x4x3x": [1, -1, 1, 0],
    "x3x4o3x": [1, -1, 0, -1],
    "x3o4x3x": [1, -1, 0, -1],
    "x3x4x3x": [1, -1, 1, -1]
}

COXF4_SEEDS = {
    "o3o4o3x": FVector.parse("[1,1,-r2,0]"),
    "x3o4o3o": FVector.parse("[1,1,-r2,0]"),
    "x3o4o3x": FVector.parse("[1,1+r2,-1-r2,-1]"),
    "o3x4x3o": FVector.parse("[0,-2-2*r2,2+2*r2,0]"),
    "o3o4x3o": FVector.parse(" [0,-2,2*r2,0]"),
    "o3x4o3o": FVector.parse(" [0,-2,2*r2,0]"),
    "x3x4o3o": FVector.parse("[1,3,-3*r2,0]"),
    "o3o4x3x": FVector.parse("[1,3,-3*r2,0]"),
    "o3x4o3x": FVector.parse("[1,1+2*r2,-2-r2,0]"),
    "x3o4x3o": FVector.parse("[1,1+2*r2,-2-r2,0]"),
    "o3x4x3x": FVector.parse("[1,3+2*r2,-2-3*r2,0]"),
    "x3x4x3o": FVector.parse("[1,3+2*r2,-2-3*r2,0]"),
    "x3x4o3x": FVector.parse("[1,3+r2,-1-3*r2,-1]"),
    "x3o4x3x": FVector.parse("[1,3+r2,-1-3*r2,-1]"),
    "x3x4x3x": FVector.parse("[1,3+3*r2,-3-3*r2,-1]"),
}

_DEFAULT_PATH = "../geometry/data/"


class CoxF4(CoxeterGroup4D):
    def __init__(self, path=None):
        self.size = 1152
        zero = QR.from_integers(0, 1, 0, 1, root_modulus=2, root_string="r2")
        one = QR.from_integers(1, 1, 0, 1, root_modulus=2, root_string="r2")
        half = QR.from_integers(1, 2, 0, 1, root_modulus=2, root_string="r2")
        root2inv = QR.from_integers(0, 1, 1, 2, root_modulus=2, root_string="r2")

        normals = [
            FVector([one, zero, zero, zero]),
            FVector([half, half, root2inv, zero]),
            FVector([zero, root2inv, half, half]),
            FVector([zero, zero, zero, one])
        ]

        identity = FMatrix([[one, zero, zero, zero], [zero, one, zero, zero],
                            [zero, zero, one, zero], [zero, zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]
        epsilon = EpsilonTensor(4, root_modulus=2, root_string="r2")

        super().__init__("coxF4", normals, generators, zero, epsilon, identity,
                         path=path or _DEFAULT_PATH,seeds=COXF4_SEEDS,signatures=COXF4_SIGNATURES)

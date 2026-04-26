from mathematics.groups.coxeter_group import CoxeterGroup4D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector, EpsilonTensor

COXB4_SIGNATURES = {
    "x3o3o4o": [0, 1, 1, 0],
    "o3o3o4x": [0, 0, 0, 1],
    "o3x3o4o": [0, 1, 0, 0],
    "o3o3x4o": [0, 0, 1, 0],
    "x3x3o4o": [1, -1, 0, 0],
    "x3o3o4x": [0, 0, 1, 1],
    "o3o3x4x": [0, 0, 1, -1],
    "o3x3o4x": [1, 0, -1, 1],
    "o3x3x4o": [0, 1, -1, 0],
    "x3o3x4o": [1, 1, -1, 0],
    "x3x3x4o": [1, -1, 1, 0],
    "x3x3o4x": [1, -1, 0, -1],
    "x3o3x4x": [1, 0, 1, -1],
    "o3x3x4x": [0, 1, -1, 1],
    "x3x3x4x": [1, -1, 1, -1]
}

_DEFAULT_PATH = "../geometry/data/"


class CoxB4(CoxeterGroup4D):
    def __init__(self, path=None):
        self.size = 384
        zero = QR.from_integers(0, 1, 0, 1, root_modulus=2, root_string="r2")
        one = QR.from_integers(1, 1, 0, 1, root_modulus=2, root_string="r2")
        half = QR.from_integers(1, 2, 0, 1, root_modulus=2, root_string="r2")
        root2inv = QR.from_integers(0, 1, 1, 2, root_modulus=2, root_string="r2")

        normals = [
            FVector([one, zero, zero, zero]),
            FVector([half, half, root2inv, zero]),
            FVector([zero, zero, root2inv, root2inv]),
            FVector([zero, zero, zero, one])
        ]

        identity = FMatrix([[one, zero, zero, zero], [zero, one, zero, zero],
                            [zero, zero, one, zero], [zero, zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]
        epsilon = EpsilonTensor(4, root_modulus=2, root_string="r2")

        super().__init__("coxB4", normals, generators, zero, epsilon, identity,
                         path=path or _DEFAULT_PATH)

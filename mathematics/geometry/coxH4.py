from mathematics.geometry.coxeter_group import CoxeterGroup4D
from mathematics.geometry.field_extensions import QR, FMatrix, FVector, EpsilonTensor

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

_DEFAULT_PATH = "data/"


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
                         path=path or _DEFAULT_PATH)

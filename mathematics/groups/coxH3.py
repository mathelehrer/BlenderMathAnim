from mathematics.groups.coxeter_group import CoxeterGroup3D
from mathematics.algebra.field_extensions import QR, FMatrix, FVector

COXH3_SEEDS = {
    "o3o5x": FVector.parse("[1, 1, -1]"),
    "x3o5x": FVector.parse("[1, 1,2+r5]"),
    "x3x5x": FVector.parse("[1, 1,-3-2*r5]"),
    "x3o5o": FVector.parse("[1, 0,1/2+1/2*r5]"),
    "x3x5o": FVector.parse("[1, 0,-3/2-3/2*r5]"),
    "o3x5x": FVector.parse("[0, 1,-5/2-3/2*r5]"),
    "o3x5o": FVector.parse("[0, 0,1+r5]")
}

COXH3_SIGNATURES = {
    "DODECA": [1, 1, 1],
    "RHOMBICOSIDODECA": [1, -1, 1],
    "TRUNC_ICOSIDODECA": [1, 1, -1],
    "ICOSA": [1, 0, 1],
    "TRUNC_ICOSA": [1, 0, -1],
    "TRUNC_DODECA": [0, 1, -1],
    "ICOSIDODECA": [0, 0, 1],
    "o3o5x": [1, 1, 1],
    "x3o5o": [1, 0, 1],
    "o3x5o": [0, 0, 1],
    "o3x5x": [0, 1, -1],
    "x3x5o": [1, 0, -1],
    "x3o5x": [1, -1, 1],
    "x3x5x": [1, 1, -1],

}

COXH3_NAMES = {
    "DODECA": "o3o5x",
    "RHOMBICOSIDODECA": "x3o5x",
    "TRUNC_ICOSIDODECA": "x3x5x",
    "ICOSA": "x3o5o",
    "TRUNC_ICOSA": "x3x5o",
    "TRUNC_DODECA": "o3x5x",
    "ICOSIDODECA": "o3x5o"
}

_DEFAULT_PATH = "../mathematics/geometry/data/"


class CoxH3(CoxeterGroup3D):
    def __init__(self, path=None):
        self.size=120
        zero = QR.from_integers(0, 1, 0, 1)
        one = QR.from_integers(1, 1, 0, 1)
        half = QR.from_integers(1, 2, 0, 1)

        normals = [
            FVector([one, zero, zero]),
            FVector([zero, one, zero]),
            FVector([half,
                     QR.from_integers(1, 4, 1, 4),
                     QR.from_integers(-1, 4, 1, 4)]),
        ]

        identity = FMatrix([[one, zero, zero], [zero, one, zero], [zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]

        super().__init__("coxH3", normals, generators,
                         path=path or _DEFAULT_PATH,
                         seeds=COXH3_SEEDS, signatures=COXH3_SIGNATURES,
                         name_dict=COXH3_NAMES)

from mathematics.geometry.coxeter_group import CoxeterGroup3D
from mathematics.geometry.field_extensions import QR, FMatrix, FVector

COXP6_SEEDS = {
    "PRISM3": FVector.parse("[1,0,2/3*r3]"),
    "PRISM6": FVector.parse("[1,1,-r3]"),
}

COXP6_SIGNATURES = {
    "PRISM3": (1, 0, 1),
    "PRISM6": (1, 1, -1),
}

COXP6_TYPES = {
    (1, 0, 1): "PRISM3",
    (1, 1, -1): "PRISM6",
}

_DEFAULT_PATH = "../mathematics/geometry/data/"


class CoxP6(CoxeterGroup3D):
    def __init__(self, path=None, no_elements=False):
        self.size=12
        zero = QR.from_integers(0, 1, 0, 1, 3, "r3")
        one = QR.from_integers(1, 1, 0, 1, 3, "r3")
        half = QR.from_integers(1, 2, 0, 1, 3, "r3")

        normals = [
            FVector([one, zero, zero]),
            FVector([zero, one, zero]),
            FVector([zero, half, QR.from_integers(0, 1, 1, 2, 3, "r3")]),
        ]

        identity = FMatrix([[one, zero, zero], [zero, one, zero], [zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]

        super().__init__("coxP6", normals, generators,
                         path=path or _DEFAULT_PATH,
                         load_elements=not no_elements,
                         seeds=COXP6_SEEDS, types=COXP6_TYPES)

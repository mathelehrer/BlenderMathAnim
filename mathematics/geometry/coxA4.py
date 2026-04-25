from mathematics.geometry.coxeter_group import CoxeterGroup4D
from mathematics.geometry.field_extensions import QR, FMatrix, FVector, EpsilonTensor

COXA4_SIGNATURES = {
    "o3o3o3x": [0, 0, 0, 1],
    "x3o3o3o": [0, 0, 0, 1],
    "o3o3x3o": [0, 1, 0, 0],
    "o3x3o3o": [0, 1, 0, 0],
    "o3o3x3x": [0, 0, 1, -1],
    "x3x3o3o": [0, 0, 1, -1],
    "x3o3o3x": [0, 1, 1, -1],
    "o3x3x3o": [0, 1, -1, 0],
    "o3x3o3x": [0, 1, 0, 1],
    "x3o3x3o": [0, 1, 0, 1],
    "x3o3x3x": [1, 0, -1, 1],
    "x3x3o3x": [1, 0, -1, 1],
    "o3x3x3x": [0, 1, -1, 1],
    "x3x3x3o": [0, 1, -1, 1],
    "x3x3x3x": [1, -1, 1, -1]
}

_DEFAULT_PATH = "data/"


class CoxA4(CoxeterGroup4D):
    def __init__(self, path=None):
        self.size = 120
        zero = QR.from_integers(0, 1, 0, 1, root_modulus=5, root_string="r5")
        one = QR.from_integers(1, 1, 0, 1, root_modulus=5, root_string="r5")
        half = QR.from_integers(1, 2, 0, 1, root_modulus=5, root_string="r5")

        normals = [
            FVector([one, zero, zero, zero]),
            FVector([half, QR.from_integers(-1, 4, 1, 4, 5, "r5"),
                     QR.from_integers(1, 4, 1, 4, 5, "r5"), zero]),
            FVector([zero, QR.from_integers(1, 4, 1, 4, 5, "r5"),
                     QR.from_integers(-1, 4, 1, 4, 5, "r5"), half]),
            FVector([zero, zero, zero, one])
        ]

        identity = FMatrix([[one, zero, zero, zero], [zero, one, zero, zero],
                            [zero, zero, one, zero], [zero, zero, zero, one]])
        generators = [FMatrix((identity - (n * n) - (n * n)).components) for n in normals]
        epsilon = EpsilonTensor(4, root_modulus=5, root_string="r5")

        super().__init__("coxA4", normals, generators, zero, epsilon, identity,
                         path=path or _DEFAULT_PATH)

    def _suffix(self, signature):
        if isinstance(signature, tuple):
            signature = list(signature)
        return "".join(str(i) for i in signature)

    def check_normals(self, signature):
        if isinstance(signature, str):
            signature = COXA4_SIGNATURES[signature]
        vertices = self.point_cloud(signature)
        cells = self.get_cells(signature)
        faces = self.get_faces(signature)
        for cell, normal in cells.items():
            face1 = None
            face2 = None
            cell_set = set(cell)
            for face in faces:
                face_set = set(face)
                if face_set < cell_set and face1 is None:
                    face1 = list(face_set)
                elif face_set < cell_set and face2 is None:
                    face2 = list(face_set)
                    break
            v1 = vertices[face1[0]]
            v2 = vertices[face1[1]]
            v3 = vertices[face1[2]]
            common = set(face1).intersection(set(face2))
            different = set(face2).difference(common)
            v4 = vertices[different.pop()]
            e1 = v1 - v4
            e2 = v2 - v4
            e3 = v3 - v4
            tensor = e1 * e2 * e3
            n = self._epsilon.contract(tensor, axes=[[1, 2, 3], [0, 1, 2]])
            print(n, normal, normal.dot(e1), normal.dot(e2), normal.dot(e3),
                  e1.dot(n), e2.dot(n), e3.dot(n))

    def get_cells_in_conjugacy_classes(self, signature=None):
        if signature is None:
            signature = COXA4_SIGNATURES["x3x3x3x"]
        if isinstance(signature, str):
            signature = COXA4_SIGNATURES[signature]
        return super().get_cells_in_conjugacy_classes(signature)

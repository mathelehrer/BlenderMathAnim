# we generate the full coxeter group of type H3
# see https://en.wikipedia.org/wiki/Coxeter_group#Table_of_all_Coxeter_groups
from mathematics.geometry.cell600 import QR5, FTensor, FMatrix, FVector


class CoxH3:
    def __init__(self):

        zero = QR5.from_integers(0,1,0,1)
        one = QR5.from_integers(1,1,0,1)
        half = QR5.from_integers(1,2,0,1)
        two = QR5.from_integers(2,1,0,1)

        # normal vectors
        normals = [
                    FVector([one,zero,zero]),
                   FVector([zero,one,zero]),
                   FVector([QR5.from_integers(1,2,0,1),QR5.from_integers(1,4,1,4),QR5.from_integers(-1,4,1,4)]),
                   ]
        # normal vectors
        normals = [
                    FVector([one,zero,zero]),
                   FVector([zero,one,zero]),
                   FVector([QR5.from_integers(1,2,0,1),QR5.from_integers(1,4,1,4),QR5.from_integers(-1,4,1,4)]),
                   ]

        # the generators of H3 are given by
        identity =  FMatrix([[one,zero,zero],[zero,one,zero],[zero,zero,one]])
        generators=[identity-(n*n)-(n*n) for n in normals]
        # cast to matrix for proper matrix multiplication
        generators = [FMatrix(g.components) for g in generators]

        # generators = [
        #     FMatrix([[-one,zero,zero],[zero,one,zero],[zero,zero,one]]),
        #     FMatrix([[one,zero,zero],[zero,-one,zero],[zero,zero,one]]),
        #     FMatrix([[half,QR5.from_integers(-1,4,-1,4),QR5.from_integers(1,4,-1,4)],
        #              [QR5.from_integers(-1,4,-1,4),QR5.from_integers(1,4,-1,4),-half],
        #              [QR5.from_integers(1,4,-1,4),-half,QR5.from_integers(1,4,1,4)]])]

        [print(g) for g in generators]

        # generate the group from the generators
        new_elements =generators
        elements = set(generators)
        while len(new_elements)>0:
            next_elements = []
            for h in new_elements:
                for g in generators:
                    element = g*h
                    if element not in elements:
                        next_elements.append(element)
                        elements.add(element)
                        # print(len(elements),end=" ")
            new_elements = next_elements
            # print(len(new_elements))

        # try, whether the center of all normals is a good candidate to generate
        # the truncated icosidodecahedron

        center = sum(normals,FVector([zero,zero,zero]))

        self. point_cloud = [element@center for element in elements]

    def get_point_cloud(self):
        return [p.real() for p in self.point_cloud]





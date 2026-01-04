from __future__ import annotations

import random
from fractions import Fraction


class QR5:
    def __init__(self,x:Fraction,y:Fraction):
        self.x=x
        self.y=y

    def __add__(self,other:QR5)->QR5:
        return QR5(self.x+other.x,self.y+other.y)

    def __sub__(self,other:QR5)->QR5:
        return QR5(self.x-other.x,self.y-other.y)

    def __mul__(self,other:QR5)->QR5:
        return QR5(self.x*other.x+5*self.y*other.y,
                   self.x*other.y+self.y*other.x)

    def __truediv__(self,other:QR5)->QR5:
        return (QR5(1/other.norm(),Fraction(0,1))
                *(self*other.conj()))

    def __eq__(self,other:QR5)->bool:
        return self.x==other.x and self.y==other.y

    def __str__(self)->str:
        return str(self.x)+str(self.y)+"*r5"

    def __neg__(self)->QR5:
        return QR5(-self.x,-self.y)

    def norm(self) -> Fraction:
        return self.x ** 2 - 5 * self.y ** 2

    def conj(self) -> QR5:
        return QR5(self.x, -self.y)

if __name__ == '__main__':
    for i in range(1000):
        z = QR5(Fraction(random.randint(1,100),1),Fraction(random.randint(1,100),2))
        w = QR5(Fraction(random.randint(1,100),3),Fraction(random.randint(1,100),4))

        assert z+w==w+z, "Something wrong with addition"
        assert z-w==-(w-z), "Something wrong with subtraction"
        assert (z*z.conj()).x==z.norm(), "Something wrong with norm"
        assert (z*z.conj()).y==Fraction(0,1)
        assert z/w*w==z, "Something wrong with "+str(z)+" and "+str(w)




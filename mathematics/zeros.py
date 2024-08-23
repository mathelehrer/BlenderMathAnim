import math
from random import random
import numpy as np

from mathematics.polynomial import Polynomial


def chop(x,precision=6):
    if isinstance(x,list):
        return [chop(e) for e in x]
    if isinstance(x,complex):
        return chop(np.real(x))+chop(np.imag(x))*1j
    if isinstance(x,float):
        return round(x*10**precision)/10**precision
    return x

def random_complex(radius=1):
    x = -radius+random()*radius*2
    y=  -radius+random()+radius*2
    return x+1j*y

def zeros_of_f(f,fp,n,attempts = 100,radius=2):
    """
        find n different roots of the function f
        make sure that the number of roots exists

        the default number of attempts is 100,
        if the number of roots cannot be found a warning is
        presented
    """
    tol = 1e-8

    attempt = 0
    zeros = set()

    while attempt<attempts and len(zeros)<n:
        x0 = random_complex(radius=radius)
        delta = np.Infinity
        count = 0
        while delta>tol and count<100:
            x=x0-f(x0)/fp(x0)
            delta = abs(x-x0)
            x0=x
            count+=1
        if not np.isnan(x):
            zeros = zeros.union({chop(x)})
        attempt += 1

    return zeros



if __name__ == '__main__':
    # print(zeros_of_f(lambda z:z**2+1,lambda z:2*z,2))
    pol0 = Polynomial([0,1])
    pol1 = pol0*pol0+pol0
    pol2 = pol1*pol1+pol0
    pol3 = pol2*pol2+pol0
    print(zeros_of_f(pol3.to_function(),pol3.derivative().to_function(),8))

    print(pol3)
    print(pol3.derivative())


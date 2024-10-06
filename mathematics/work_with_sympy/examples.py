from sympy import symbols, sin, cos, simplify


def replace_example():
    x, y = symbols('x y')
    f = (1-x**2)**(5/2)
    # Use sympy.replace() method
    gfg = simplify(f.replace((1-x**2), y**2))
    print(gfg)

if __name__ == '__main__':
    replace_example()
a_k = (1.0, -2.2499997, 1.2656208, -0.3163866,
       0.0444479, -0.0039444, 0.00021)
b_k = (0.79788456, -0.00000077, -0.0055274, -0.00009512,
       0.00137237, -0.00072805, 0.00014476)
c_k = (-0.04166397, -0.00003954, 0.00262573,
       -0.00054125, -0.00029333, 0.00013558)

def horner_rpn(coefficients, variable):
    terms = [str(coefficients[-1])]
    for c in coefficients[-2::-1]:
        terms.append(variable + ',*,' + str(c) + ',+')
    return ','.join(terms)

def bessel_j0_rpn(x):
    aux = {}
    aux['xs'] = x + ',3,min'
    aux['ts'] = 'xs,3,/,2,**'
    aux['j0_small'] = horner_rpn(a_k, 'ts')
    aux['xl'] = x + ',3,max'
    aux['t'] = '3,xl,/'
    aux['f0'] = horner_rpn(b_k, 't')
    c_sum = horner_rpn(c_k, 't')
    aux['th'] = 'xl,0.78539816,-,' + c_sum + ',t,*,+'
    aux['amp'] = 'f0,xl,sqrt,/'
    aux['small_x'] = x + ',3,<'
    aux['large_x'] = '1,small_x,-'
    aux['j0_large'] = 'amp,th,cos,*'
    aux['j0'] = 'j0_small,small_x,*,j0_large,large_x,*,+'
    return aux

def bessel_node_group(tree):
    aux = bessel_j0_rpn('x')
    return make_function(tree, name='BesselJ0',
                         functions={'J0': 'j0'},
                         aux_functions=aux,
                         inputs=['x'], outputs=['J0'],
                         scalars=['x', 'J0'] + list(aux))

import numpy as np

pi = np.pi
def polygon_function(phi,r,n):
    """
    returns the function that traces out the shape of a regular n-gon,
    whose vertices are located on a circle with radius r

    :param phi: angular coordinate
    :param r: radius of the surrounding circle
    :param n: number of vertices
    :return:
    """

    z0 = r*np.exp(2j*pi/n* np.floor(n*phi/2/pi))
    z1 = r*np.exp(2j*pi/n*np.ceil(n*phi/2/pi))

    if np.abs(z1-z0)==0:
        return z0
    return (z0*np.conj(z1)-z1*np.conj(z0))/(np.conj(z1-z0)-(z1-z0)*np.exp(-2j*phi))



def length_of_curve(curve,domain=[0,1],resolution=100):
    t = domain[0]
    dt = (domain[1]-domain[0])/resolution
    start=curve(t)
    length=0
    for i in range(1,resolution+1):
        t+=dt
        loc = curve(t)
        length+=(loc-start).length
        start = loc
    return length


def integrate_curve_up_to_length(curve,length=1,domain=[0,1],start_length=0,t0=0,resolution=100):
    dt = (domain[1]-domain[0])/resolution
    loc0 = curve(t0)
    while start_length<length:
        t0+=dt
        loc = curve(t0)
        start_length+=(loc-loc0).length
        loc0=loc
    return t0


def create_curve_map(curve,domain=[0,1],resolution=100):
    '''
    creates a dictionary that maps the length of the curve to the curve parameter
    :param curve:
    :return:
    '''
    dictionary = {}
    full_length=length_of_curve(curve,domain=domain,resolution=resolution)

    t = domain[0]
    dt = (domain[1] - domain[0]) / resolution
    start = curve(t)
    length = 0
    dictionary[np.round(length/full_length*resolution)/resolution]=t
    for i in range(1, resolution + 1):
        t += dt
        loc = curve(t)
        length += (loc - start).length
        start = loc
        dictionary[length]=t

    return dictionary

if __name__ == '__main__':
    for i in range(10):
        phi = 2*pi/10*i
        print(polygon_function(phi,1,6))
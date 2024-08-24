import unittest

import numpy as np
from numpy.linalg import inv
from sympy.physics.quantum.circuitplot import pyplot

from video_apollonian.indras_utils.circle import IndraCircle
from video_apollonian.indras_utils.indra_generating_algorithms import ThetaModel, fixed_point_of, GrandMasRecipe, k_of, \
    g_of, SchottkyFamily, quick_fixed_points_of, ApollonianModel, BlenderModel
from video_apollonian.indras_utils.mymath import moebius_on_circle, moebius_on_point

def plot(set,range=[[-1,1],[-1,1]],colors=None):
    # circles = [c_a,c_b,c_A,c_B]
    # now, one has to find touching circles among the two different sets of circle families
    pyplot.gca().set_aspect('equal')
    pyplot.gca().set_xlim(tuple(range[0]))
    pyplot.gca().set_ylim(tuple(range[1]))
    for i,c in enumerate(set):
        if colors is not None:
            if len(colors)>i:
                color=colors[i]
            else:
                color = colors[-1]
        else:
            color='black'
        if c.r < np.inf:
            pyplot.gca().add_patch(pyplot.Circle((np.real(c.c), np.imag(c.c)), c.r, fill=False,color=color))
    pyplot.show()
class MyTestCase(unittest.TestCase):

    def test_fundamental_domain(self):
        model = GrandMasRecipe(ta=2.01,tb=2.01)
        [a, b, A, B] = model.get_generators()

        com = a@b@A@B
        tr = com[0][0]+com[1][1]
        self.assertAlmostEqual(tr,-2)

        k = k_of(b)
        g = g_of(b)
        gi = inv(g)

        c_b = moebius_on_circle(gi, IndraCircle(0, np.sqrt(k)))
        c_B = moebius_on_circle(gi, IndraCircle(0, 1/np.sqrt(k)))

        print(c_b,moebius_on_circle(b,c_B))
        print(c_B,moebius_on_circle(B,c_b))

        # for all real values of tb>2 the two circles c_b and c_B touch the x-axis
        # therefore the x-axis is taken to be the boundary c_A

        circles = [c_b,c_B]

        pyplot.gca().set_aspect('equal')
        pyplot.gca().set_xlim((-1, 1))
        pyplot.gca().set_ylim((-1, 1))
        [pyplot.gca().add_patch(pyplot.Circle((np.real(c.c),np.imag(c.c)),c.r,fill=False)) for c in circles]
        pyplot.show()

    def test_construct_fundamental_domain(self):

        model = GrandMasRecipe(ta=4.1, tb=2.05)
        [a, b, A, B] = model.get_generators()

        com = a @ b @ A @ B
        tr = com[0][0] + com[1][1]
        self.assertAlmostEqual(tr, -2)

        kb = k_of(b)
        gb = g_of(b)
        gbi = inv(gb)

        c_b = moebius_on_circle(gbi, IndraCircle(0, np.sqrt(kb)))
        c_B = moebius_on_circle(gbi, IndraCircle(0, 1/np.sqrt(kb)))

        print(c_b,moebius_on_circle(b,c_B))
        print(c_B,moebius_on_circle(B,c_b))

        # push circles into scaling frame of a
        ka = k_of(a)
        ga = g_of(a)
        gai = inv(ga)
        image_1 = moebius_on_circle(ga,c_b)
        image_2 = moebius_on_circle(ga,c_B)

        circles=[image_1,image_2]
        #simple geometric arguments, if you look at the picture
        ka1 = np.abs(image_1.c)-image_1.r
        ka2 = np.abs(image_1.c)+image_1.r

        # create family of scaling circles
        circles2=[IndraCircle(0,r/10) for r in range(1,60,2)]

        circles+=circles2

        pyplot.gca().set_aspect('equal')
        pyplot.gca().set_xlim((-3, 3))
        pyplot.gca().set_ylim((-3, 3))
        for c in circles:
            if c.r<np.inf:
                pyplot.gca().add_patch(pyplot.Circle((np.real(c.c), np.imag(c.c)), c.r, fill=False))
        pyplot.show()

    def test_fundamental_domain_final(self):
        model = GrandMasRecipe(ta=2.01, tb=2.01)
        [a, b, A, B] = model.get_generators()

        com = a @ b @ A @ B
        tr = com[0][0] + com[1][1]
        self.assertAlmostEqual(tr, -2)

        kb = k_of(b)
        gb = g_of(b)
        gbi = inv(gb)

        c_b = moebius_on_circle(gbi, IndraCircle(0, np.sqrt(kb)))
        c_B = moebius_on_circle(gbi, IndraCircle(0, 1 / np.sqrt(kb)))

        print(c_b, moebius_on_circle(b, c_B))
        print(c_B, moebius_on_circle(B, c_b))

        # push circles into scaling frame of a
        ga = g_of(a)
        gai = inv(ga)
        image_1 = moebius_on_circle(ga, c_b)

        # simple geometric arguments, if you look at the picture
        ka1 = np.abs(image_1.c) - image_1.r
        ka2 = np.abs(image_1.c) + image_1.r

        c_A = moebius_on_circle(gai,IndraCircle(0,ka1))
        c_a = moebius_on_circle(gai,IndraCircle(0,ka2))

        # plot in the normal frame of b
        c_Ab = moebius_on_circle(gb, c_A)
        c_ab = moebius_on_circle(gb,c_a)

        circles =  [IndraCircle(0,3*k/100) for k in range(0,100)]
        circles.append(c_Ab)
        circles.append(c_ab)


        # if np.abs(c_A.c)>10000:
        #     circles = [c_b,c_B,c_a] # ,c_A]
        # else:
        #     circles = [c_b, c_B, c_a, c_A]
        pyplot.gca().set_aspect('equal')
        pyplot.gca().set_xlim((0, 2))
        pyplot.gca().set_ylim((0, 2))
        for c in circles:
            if c.r < np.inf:
                pyplot.gca().add_patch(pyplot.Circle((np.real(c.c), np.imag(c.c)), c.r, fill=False))
        pyplot.show()


    def test_fundamental_domain_final(self):
        model = SchottkyFamily(y=1,k=1)
        [a, b, A, B] = model.get_generators()

        com = a @ b @ A @ B
        tr = com[0][0] + com[1][1]
        self.assertAlmostEqual(tr, -2)

        print(a[0][0]+a[1][1])
        print(b[0][0]+b[1][1])

        kb = k_of(b)
        gb = g_of(b)
        gbi = inv(gb)

        c_b = moebius_on_circle(gbi, IndraCircle(0, np.sqrt(kb)))
        c_B = moebius_on_circle(gbi, IndraCircle(0, 1 / np.sqrt(kb)))

        print(c_b, moebius_on_circle(b, c_B))
        print(c_B, moebius_on_circle(B, c_b))

        # push circles into scaling frame of a
        ga = g_of(a)
        gai = inv(ga)
        image_1 = moebius_on_circle(ga, c_b)

        # simple geometric arguments, if you look at the picture
        ka1 = np.abs(image_1.c) - image_1.r
        ka2 = np.abs(image_1.c) + image_1.r

        c_A = moebius_on_circle(gai,IndraCircle(0,ka1))
        c_a = moebius_on_circle(gai,IndraCircle(0,ka2))

        # plot in the normal frame of b
        c_Ab = moebius_on_circle(gb, c_A)
        c_ab = moebius_on_circle(gb,c_a)

        circles =  [IndraCircle(0,3*k/100) for k in range(0,100)]
        circles.append(c_Ab)
        circles.append(c_ab)


        # if np.abs(c_A.c)>10000:
        #     circles = [c_b,c_B,c_a] # ,c_A]
        # else:
        #     circles = [c_b, c_B, c_a, c_A]
        pyplot.gca().set_aspect('equal')
        pyplot.gca().set_xlim((-1, 1))
        pyplot.gca().set_ylim((-1, 1))
        for c in circles:
            if c.r < np.inf:
                pyplot.gca().add_patch(pyplot.Circle((np.real(c.c), np.imag(c.c)), c.r, fill=False))
        pyplot.show()


    def test_isometric_circles(self):
        model = GrandMasRecipe(ta=2.1+3j, tb=3.1-0.5j)
        [a, b, B, A] = model.get_generators()

        # isometric circles are the loci of all points where the derivative of the transformations is 1
        # a'(z)=1

        iso_a = IndraCircle(-a[1][1]/a[1][0],np.abs(1/a[1][0]))
        iso_A = IndraCircle(-A[1][1]/A[1][0],np.abs(1/A[1][0]))
        iso_b = IndraCircle(-b[1][1]/b[1][0],np.abs(1/b[1][0]))
        iso_B = IndraCircle(-B[1][1]/B[1][0],np.abs(1/B[1][0]))

        circles = [iso_a,iso_b,iso_A,iso_B]
        plot(circles)

    def test_fundamental_domain_complex_traces_from_commutator_fixed_points(self):
        model = GrandMasRecipe(ta=2.001+0.5j,tb=3.001)
        [a, b, A, B] = model.get_generators()
        com = a @ b @ A @ B
        tr = com[0][0] + com[1][1]
        #self.assertAlmostEqual(tr, -2)
        # check commutator trace (closed fractal curve condition)
        com = np.matmul(a,np.matmul(b,np.matmul(A,B)))
        com2 = B @ a @ b @ A
        com3 = A @ B @ a @ b
        com4 = b @ A @ B @ a

        p = fixed_point_of(com)
        s =fixed_point_of(com2)
        r = fixed_point_of(com3)
        q = fixed_point_of(com4)

        c_fix = IndraCircle.circle_from_three_points(p,s,r)


        colors =['red','red','red','red','blue','blue','blue','blue','green']

        ps = [p, s, r, q]
        rs = [0.03, 0.03, 0.03, 0.03]
        circles = [IndraCircle(m,r) for m,r in zip(ps,rs)]
        circles.append(c_fix)

        plot(circles,range=[[-3,3],[-3,3]],colors = colors)
    def test_fundamental_domain_complex_traces(self):
        model = GrandMasRecipe(ta=2.1+0.5j, tb=2.1)
        [a, b, B, A] = model.get_generators()

        # check commutator trace (closed fractal curve condition)
        com = a @ b @ A @ B
        tr = com[0][0] + com[1][1]
        # self.assertAlmostEqual(tr, -2)

        [f1,f2]=quick_fixed_points_of(b)
        test1 = moebius_on_point(b, f1) - f1
        test2 = moebius_on_point(b, f2) - f2
        self.assertAlmostEqual(np.abs(test1), 0)
        self.assertAlmostEqual(np.abs(test2), 0)

        kb = k_of(b)
        gb = g_of(b)
        gbi = inv(gb)

        # push circles into scaling frame of a
        ka = k_of(a)
        ga = g_of(a)
        gai = inv(ga)

        l = (kb+1)/(kb-1)

        t = gb@gai

        # general solution for the radius ra in the normal frame
        # alpha *ra^2+beta*ra+gamma=0

        alpha = np.abs(t[0][0]*t[1][0])**2
        beta = np.real(-(2*np.real(np.conj(t[0][0]*t[1][1])*t[0][1]*t[1][0])+l*np.conj(l)))
        gamma = np.abs(t[1][0]*t[1][1])**2

        print(alpha,beta,gamma)

        # usually, one finds two positive solutions that correspond to the two possible radii
        ra1 = np.sqrt((-beta-np.sqrt(beta**2-4*alpha*gamma))/2/alpha)
        ra2 = np.sqrt((-beta+np.sqrt(beta**2-4*alpha*gamma))/2/alpha)

        # the corresponding radii rb can be easily calculated
        rb1 = 2*ra1/np.abs((kb-1))
        rb2 = 2*ra2/np.abs((kb-1))

        # calculate the Schottky discs

        c_a = moebius_on_circle(gai,IndraCircle(0,ra1))
        c_A = moebius_on_circle(gai,IndraCircle(0,ra2))
        c_b = moebius_on_circle(gbi,IndraCircle(0,rb1))
        c_B = moebius_on_circle(gbi,IndraCircle(0,rb2))

        # look at things in the normal frame of b
        circles=[
            moebius_on_circle(gb,c_a),
            moebius_on_circle(gb,c_A),
                 ]
        circles+=[IndraCircle(0,3*i/10) for i in range(0,10)]

        # circles = [c_a,c_b,c_A,c_B]
        # now, one has to find touching circles among the two different sets of circle families
        pyplot.gca().set_aspect('equal')
        pyplot.gca().set_xlim((-4, 4))
        pyplot.gca().set_ylim((-4,4))
        for c in circles:
            if c.r < np.inf:
                pyplot.gca().add_patch(pyplot.Circle((np.real(c.c), np.imag(c.c)), c.r, fill=False))
        pyplot.show()

if __name__ == '__main__':
    unittest.main()

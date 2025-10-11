import random

def multidimensional_gaussian(dim=3):
    """
    return a random multidimensional gaussian distribution

    :param dim:
    :return:
    """
    for i in range(100):
        print(random.gauss(0, 1))



if __name__ == '__main__':
    multidimensional_gaussian(3)

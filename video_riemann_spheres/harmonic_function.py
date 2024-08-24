def my_func(z):
    sum = 0
    for k in range(1,1000):
        sum = sum+z/k/(k+z)
    return sum
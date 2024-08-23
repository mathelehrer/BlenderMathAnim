import csv

with open('/home/jmartin/blendermath/video_mandelbrot/data/coefficientList10.dat','r') as f:
    data = [float(row) for row in f.read().split('\n')]
print(data)

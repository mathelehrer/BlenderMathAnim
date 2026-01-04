# check possible arrangements of polygons that form polyhedra, ie. their angle sum is less than 360
# start with three faces at one vertex

def vertex_angle(n):
    return (180*n-360)/n

for i in range(11,16):
    print(i,vertex_angle(i))


max = 0
max_n = 1000
for a in range(3,max_n+1):
    for b in range(a,max_n+1):
        for c in range(b+1,max_n+1):
            alpha = vertex_angle(a)
            beta = vertex_angle(b)
            gamma = vertex_angle(c)
            va = alpha + beta + gamma
            if va < 360:
                if va > max:
                    if a%2==0 and b%2==0 and c%2==0:
                        max = va
                        print(a,b,c,va)
            else:
                break


max = 0
max_n = 2
for a in range(3,max_n+1):
    for b in range(a,max_n+1):
        for c in range(b,max_n+1):
            for d in range(c,max_n+1):
                alpha = vertex_angle(a)
                beta = vertex_angle(b)
                gamma = vertex_angle(c)
                delta = vertex_angle(d)
                va = alpha + beta + gamma+delta
                if va < 360:
                    if va > max:
                        max = va
                        print(a,b,c,d,va)
                else:
                    break

max = 0
for a in range(3,max_n+1):
    for b in range(a,max_n+1):
        for c in range(b,max_n+1):
            for d in range(c,max_n+1):
                for e in range(c, max_n+1):
                    alpha = vertex_angle(a)
                    beta = vertex_angle(b)
                    gamma = vertex_angle(c)
                    delta = vertex_angle(d)
                    epsilon = vertex_angle(e)
                    va = alpha + beta + gamma+delta+epsilon
                    if va < 360:
                        if va > max:
                            max = va
                            print(a,b,c,d,e,va)
                    else:
                        break
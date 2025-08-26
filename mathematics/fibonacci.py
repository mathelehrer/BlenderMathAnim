for i in range(200):
    if i==0:
        a=1
        print(a)
    elif i==1:
        b=1
        print(b)
    else:
        c=a+b
        a=b
        b=c
        print(c)
import os

PATH = "../mathematics/numbers/data"

def save_primes(limit,primes):
    filename = os.path.join(PATH,"primes_up_to_"+str(limit)+".data")
    with open(filename,"w") as f:
        for p in primes:
            f.write(str(p)+"\n")

def read_primes(limit):
    filename = os.path.join(PATH,"primes_up_to_"+str(limit)+".data")
    if os.path.exists(filename):
        with open(filename,"r") as f:
            primes = []
            for line in f:
                primes.append(int(line.strip()))
        return primes

    primes = [2]
    for i in range(3,limit+1,2):
        is_prime = True
        for p in primes:
            if i%p==0:
                is_prime = False
                break
            if p>i**0.5:
                break
        if is_prime:
            primes.append(i)

    save_primes(limit,primes)
    return primes

if __name__ == '__main__':
    PATH = os.path.join("../",PATH)
    print(read_primes(1000000))
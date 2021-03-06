import autodiff as ad
import numpy as np

def f(x):
    return 2 * (x**2)

def g(x):
    return (3*(x[0]**2)) + (2*x[1])

if __name__ == '__main__':
    df = ad.grad(f)
    dg = ad.grad(g)
    print(df(5)) # 4(5) = 20
    print(dg([0, 0])) # [6(0), 2] = [0, 2]
    print(dg(np.array([5, 92]))) # [6(5), 2] = [30, 2]

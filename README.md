# Autodiff
**NOTE: AUTODIFF IS NOW A PART OF [KIWIML](https://github.com/oriyonay/kiwiml/)**
An automatic, lightweight, multidimensional differentiator in python :)

**Features**:
* Supports multidimensional differentiation
* Returns a gradient function
* NumPy Compatible

**Easy to use**:
```python
import autodiff as ad
import numpy as np

# define some functions:
def f(x):
    return 2 * (x**2)

def g(x):
    return (3*(x[0]**2)) + (2*x[1])

if __name__ == '__main__':
    # create gradient functions for f() and g():
    df = ad.grad(f)
    dg = ad.grad(g)
    
    # call the functions:
    print(df(5)) # 4(5) = 20
    print(dg([0, 0])) # [6(0), 2] = [0, 2]
    print(dg(np.array([5, 92]))) # [6(5), 2] = [30, 2]
```

**Notes**:
* Autodiff can take functions with multiple arguments, but will only differentiate with respect to the first one.

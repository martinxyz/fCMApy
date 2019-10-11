import numpy as np
from fCSA import fCSA

def f(x):
    return np.sum(x**2)

def f_noise(x):
    return f(x) + np.random.normal()

n = 100
explicit_noise_handling = False

optimizer = fCSA(np.ones(n), noise_adaptation=explicit_noise_handling)
for i in range(10000):
    if not explicit_noise_handling:
        optimizer.rate = n/(n + i)
    optimizer.step(f_noise)
    print(i, f(optimizer.mean), optimizer.avg_loss, np.sqrt(optimizer.variance), optimizer.rate)

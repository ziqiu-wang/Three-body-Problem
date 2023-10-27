import numpy as np
import scipy.optimize as opt

# minimize r(x), where x may be time t or angle phi
def find_min(r,a,b):
    root = opt.brent(r, brack=(a,b), full_output=True)
    xmin, peri = root[:2]   # peri = pericenter distance
    return xmin, peri

# Keplerian problem:
# smaj is semi-major axis, smin semi-minor axis.
# Very hard to find direct dependence of r on t,
# so instead minimize r(phi) first.
def r_k(phi):
    return (smaj*(1-e**2))/(1+e*np.cos(phi))

# initializations
smaj = 1
smin = 0.5
e = np.sqrt(1-smin**2/smaj**2)

# find first minimum
a = -1
b = 1
xmin, peri = find_min(r_k,a,b)
print(xmin, peri)

# find second minimum
a = 1
b = 6
xmin2, peri2 = find_min(r_k,a,b)
print(xmin2, peri2)

'''Result: xmin is nearly 0, and xmin2 is nearly 2pi, as expected'''

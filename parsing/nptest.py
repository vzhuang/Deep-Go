import numpy as np

blah = np.zeros([10, 10])

def modify(smt):
    smt[0][0] = 10

modify(blah)
print blah

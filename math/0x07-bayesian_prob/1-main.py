#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    intersection = __import__('1-intersection').intersection

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
"""
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./1-main.py 
[0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
 5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
 8.67650639e-50 9.14515194e-80 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$
"""

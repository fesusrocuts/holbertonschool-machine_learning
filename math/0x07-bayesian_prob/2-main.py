#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    marginal = __import__('2-marginal').marginal

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
"""
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./2-main.py 
0.008229580791426582
alexa@ubuntu-xenial:0x07-bayesian_prob$
"""

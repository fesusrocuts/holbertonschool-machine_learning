#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here
"""
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html
These are subplot grid parameters encoded as a single integer.
"111" means "1x1 grid, first subplot" and
"234" means "2x3 grid, 4th subplot".
"""
plt.subplot(111)
plt.plot(x, y1, 'r--', x, y2, 'g-')
plt.ylabel('Fraction Remaining')
plt.xlabel('Time (years)')
plt.title('Exponential Decay of Radioactive Elements')
plt.legend(['C-14', 'Ra-226'])
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.show()

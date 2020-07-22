#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
width = 0.5
people = ['Farrah', 'Fed', 'Felicia']
f = ['apples', 'bananas', 'oranges', 'peaches']
c = ['red', 'yellow', '#ff8000', '#ffe5b4']
pos = [None, fruit[0], fruit[0] + fruit[1], fruit[0] + fruit[1] + fruit[2]]

plt.yticks(np.arange(0, 90, 10))
plt.ylim(0, 80)
plt.ylabel('Quantity of Fruit')
plt.suptitle('Number of Fruit per Person')
v = plt.bar(people, fruit[0], width=width, color=c[0], label=f[0], bottom=pos[0])
x = plt.bar(people, fruit[1], width=width, color=c[1], label=f[1], bottom=pos[1])
y = plt.bar(people, fruit[2], width=width, color=c[2], label=f[2], bottom=pos[2])
z = plt.bar(people, fruit[3], width=width, color=c[3], label=f[3], bottom=pos[3])
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
plt.legend(handles=[v, x, y, z])
plt.show()

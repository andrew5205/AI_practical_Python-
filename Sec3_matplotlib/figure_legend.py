

import matplotlib.pyplot as plt 
import numpy as np 


x = np.linspace(0,11,10)


fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x,x, label='X vs X')
ax.plot(x,x**2, label='X vs X^2')
ax.legend(loc=0)

        # loc=(coordinates)
# ax.legend(loc=(1.1,0.5))




""" 
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html 
Location String	Location Code
'best'	0
'upper right'	1
'upper left'	2
'lower left'	3
'lower right'	4
'right'	5
'center left'	6
'center right'	7
'lower center'	8
'upper center'	9
'center'	10
"""


plt.show()




















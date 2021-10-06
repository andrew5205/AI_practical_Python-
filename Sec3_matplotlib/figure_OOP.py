

import matplotlib.pyplot as plt 
import numpy as np 


""" plt.figure(), plain canvas size 432 X 288 pixels """


a = np.linspace(0, 10, 11)
b = a ** 4

x = np.arange(0,10)
y = 2 * x


# 1.generate a fig 
fig = plt.figure()

# 2. add_axes - plt.add_axes(x, y, width, length) - left bottom point
axes1 = fig.add_axes([0,0,1,1])

# axes.set_XXX()
axes1.set_xlim(0,8)
axes1.set_ylim(0, 8000)
axes1.set_xlabel("A")
axes1.set_ylabel('B')
axes1.set_title("Power of 4")


# 3.plot() 
axes1.plot(a,b)



# smaller axes - zoom in
axes2 = fig.add_axes([0.2, 0.5, 0.25, 0.25])
axes2.plot(a,b)

# axes.set_XXX()
axes2.set_xlim(1,2)
axes2.set_ylim(0, 30)
axes2.set_xlabel("A")
axes2.set_ylabel('B')
axes2.set_title("Zoom in")




""" figure parameters"""
fig2 = plt.figure(figsize=(2, 2), dpi=100)

axes = fig2.add_axes([0,0,1,1])

axes.plot(a,b)

# fig2.savefig('named_fig.png', bbox_inches='tight)

plt.show()

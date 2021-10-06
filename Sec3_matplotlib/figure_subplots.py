
""" 
plt.subplots()  VS figure.add_subplot()

fig, axes = plt.figure(2,2)
axes[row][colm].plot() 

########################################
fig = plt.figure()
chart1 = fig.add_subplot(121)
ax1.plot()

chart2 = fig.add_subplot(121)
ax2.plot()

...

"""




import matplotlib.pyplot as plt 
import numpy as np 


# fig, ax = plt.subplots(nrows, ncolms)
# ax is ndarrays

a = np.linspace(0,10,11)
b = a ** 4

x = np.arange(0, 10)
y = 2 * x


fig, axes = plt.subplots(2,2)
# axes.plot()

axes[0][0].plot(x,y)
axes[1][1].plot(a,b)

axes[1][1].set_xlabel('x label 1,1')
axes[1][1].set_title('title for 1,1 block')
axes[1][1].setYlim(2,10)

fig.suptitle('main title on top of the fig')

plt.tight_layout()

fig.savefig('name.png', bbox='tight')




################################################################
fig, axes = plt.subplots(nrows=3, ncols=1)
""" 
# looping can be used for plt.subplots(), 
# simply loop adding axes[i][j]

for ax in axes:
    ax.plot(x,y)
"""

axes[0].plot(x,y)
axes[1].plot(x,y)
axes[2].plot(x,y)

axes[0].set_xlabel('xlabel 0')

plt.tight_layout()


# # manual layout 
# # https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
# fig.subplots_adjust(left=None,
#     bottom=None,
#     right=None,
#     top=None,
#     wspace=0.9,
#     hspace=0.1,)


plt.show()








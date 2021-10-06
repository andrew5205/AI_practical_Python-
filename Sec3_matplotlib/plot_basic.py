
""" 
https://matplotlib.org/
https://matplotlib.org/2.0.2/gallery.html
"""


import matplotlib.pyplot as plt 
import numpy as np 

x = np.arange(0,10)
y = x * 2


plt.plot(x, y)

# set label
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# set limits
plt.xlim(0, 10)
plt.ylim(0, 20)

plt.title("my title")

# # save a fig 
# plt.savefig('./fig/plt_basic.png')



plt.show()




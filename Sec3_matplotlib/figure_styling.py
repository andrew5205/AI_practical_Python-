
""" 
https://matplotlib.org/stable/api/markers_api.html
"""

import matplotlib.pyplot as plt 
import numpy as np 


x = np.linspace(0,11,10)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(x,x, color='#33FF9C', label= "X vs X", lw=5, ls='-.', marker='s', ms=9, 
        markerfacecolor='red', markeredgewidth=2, markeredgecolor='orange')        # RGB HEX code 

# # lw == linewidth, ls == linestytle, ms == markersize
# ax.plot(x,x+3, color='#265D12', label= "X vs X", linewidth=3, linestyle='--', markersize=9)

# custom line stytle
lines = ax.plot(x,x+3, color='#D927C9', label= "X vs X+3", linewidth=3)
lines[0].set_dashes([1,2,1,2,10,2])      # [solid, blank, solid, blank]



plt.show()


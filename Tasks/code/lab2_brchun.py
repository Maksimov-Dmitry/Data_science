import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

y = [0.952,0.026,0.014,0.005,0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0 ,0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]
x = np.arange(len(y))
ax.bar(x,y, color = '0.2')
ax.set_ylabel('Вероятность очереди')
ax.grid(linestyle='-', linewidth=1)
ax.set_xlabel('длина очереди')
ax.set_xticks(x)
plt.show()
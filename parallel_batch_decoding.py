import numpy as np 
import matplotlib.pyplot as plt 

import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 22 

x = list(range(1, 11)) 
latency_measurement = [16.540902376174927, 19.05730414390564, 19.612385272979736, 20.869054794311523, 20.67660117149353, 21.932927131652832, 22.34955334663391, 22.983564138412476, 23.834572792053223, 24.18628692626953] 
x_ticks = [str(i) for i in x] 

fig, ax = plt.subplots() 

bars = ax.bar(x, latency_measurement, color = "blue", width = 0.6) 
for bar in bars: 
    yval = bar.get_height() 
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, "{:.2f}".format(yval), va='bottom', ha='center') 

ax.set_xticks(x, x_ticks) 
ax.set_xlabel("Batch Size") 
ax.set_ylabel("Latency (s)") 
ax.grid() 

plt.show() 

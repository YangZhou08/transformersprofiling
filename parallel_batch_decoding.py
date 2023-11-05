import numpy as np 
import matplotlib.pyplot as plt 

import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.fontsize'] = 22 

x = list(range(1, 21, 2)) 
# latency_measurement = [16.540902376174927, 19.05730414390564, 19.612385272979736, 20.869054794311523, 20.67660117149353, 21.932927131652832, 22.34955334663391, 22.983564138412476, 23.834572792053223, 24.18628692626953] 
latency_measurement = [9.712086200714111, 10.404566049575806, 10.668877124786377, 10.944726943969727, 11.201996803283691, 11.497334480285645, 11.744471311569214, 12.02213454246521, 12.955510139465332, 12.561285972595215] 
latency_bf16 = [9.770136833190918, 10.448176383972168, 10.71483826637268, 10.992271661758423, 11.24536418914795, 11.538933515548706, 11.78537654876709, 12.074774265289307, 12.992205142974854, 12.607782125473022] 
x_ticks = [str(i) for i in x] 

fig, ax = plt.subplots() 
bar_width = 0.6 

# bars = ax.bar([i - bar_width/2 for i in x], latency_measurement, color = "blue", width = 0.6, label = "FP32") 
bars = ax.bar([i - bar_width/2 for i in x], latency_bf16, color = "orange", width = 0.6, label = "BF16") 
# bar2 = ax.bar([i + bar_width/2 for i in x], latency_bf16, color = "orange", width = 0.6, label = "BF16") 
bar2 = ax.bar([i + bar_width/2 for i in x], latency_measurement, color = "blue", width = 0.6, label = "FP16") 
for i in range(len(bars)): 
    bar = bars[i] 
    yval = bar.get_height() 
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, "{:.2f}".format(yval), va='bottom', ha='center') 
    b2 = bar2[i] 
    yval2 = b2.get_height() 
    plt.text(b2.get_x() + b2.get_width()/2.0, yval2, "{:.2f}".format(yval2), va='bottom', ha='center') 

ax.set_xticks(x, x_ticks) 
ax.set_xlabel("Batch Size") 
ax.set_ylabel("Latency (s)") 
ax.legend() 
ax.grid() 

plt.show() 

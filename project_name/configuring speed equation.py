import numpy as np
import matplotlib.pyplot as plt
from math import radians


def vel(theta, theta_0=0, theta_dead=np.pi / 12):
        return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)
    
    
def rew(theta, theta_0=0, theta_dead=np.pi / 12):
    return vel(theta, theta_0, theta_dead) * np.cos(theta)

def line_2(theta):
    return theta/(theta + 1) * 1.64

def line_3(theta):
    return theta/(theta - 0.2) * 0.975

def line_4(theta):
    return theta/(theta - 0.8) * 0.704
    

print(rew(theta = radians(45)))


T1 = np.linspace(0 , 7*np.pi/36, 100)
T2 = np.linspace(7*np.pi/36 , 5*np.pi/8, 100)
T3 = np.linspace(5*np.pi/8 , 3*np.pi/4, 100)
T4 = np.linspace(3*np.pi/4 , np.pi, 100)

plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                  projection='polar')


#yplot = speedcurves(minV, maxV)
yplot1 = []
for i in T1:
    yplot1.append(rew(i))
    
yplot2 = []
for i in T2:
    yplot2.append(line_2(i))
    
yplot3 = []
for i in T3:
    yplot3.append(line_3(i))
    
yplot4 = []
for i in T4:
    yplot4.append(line_4(i))

ax.plot(T1, yplot1)
ax.plot(T2, yplot2)
ax.plot(T3, yplot3)
ax.plot(T4, yplot4)
ax.plot(radians(90), 0, 'go')
        
ax.set_rmin(0)
#ax.set_rmax(5)
ax.set_theta_zero_location("N")
plt.show()
# pred-prey.py
#
# example code for solving ODEs
# taken from https://www.gribblelab.org/compneuro/2_Modelling_Dynamical_Systems.html
#
import numpy as np
import pylab as plt
from scipy.integrate import odeint

def LotkaVolterra(state,t):
    x = state[0]
    y = state[1]
    alpha = 0.1
    beta =  0.1
    sigma = 0.1
    gamma = 0.1
    xd = x*(alpha - beta*y)
    yd = -y*(gamma - sigma*x)
    return [xd,yd]

t = np.arange(0,500,1)
state0 = [0.5,0.5]
state = odeint(LotkaVolterra,state0,t)
#
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(121)
ax.plot(t,state)
ax.set_ylim([0,8])
ax.set_xlabel('Time')
ax.set_ylabel('Population Size')
ax.legend(('x (prey)','y (predator)'))
ax.set_title('Lotka-Volterra equations')
# plt.savefig('fig1.png')
#
#
# animation in state-space
# fig2 = plt.figure()
ax = fig.add_subplot(122)
pb, = ax.plot(state[:,0],state[:,1],'b-',alpha=0.2)
ax.set_xlabel('x (prey population size)')
ax.set_ylabel('y (predator population size)')
p, = ax.plot(state[0:10,0],state[0:10,1],'b-')
pp, = ax.plot(state[10,0],state[10,1],'b.',markersize=10)
tt = ax.set_title("%4.2f sec" % 0.00)

# animate
step=2
for i in range(1,np.shape(state)[0]-20,step):
    p.set_xdata(state[10+i:20+i,0])
    p.set_ydata(state[10+i:20+i,1])
    pp.set_xdata(state[19+i,0])
    pp.set_ydata(state[19+i,1])
    tt.set_text("%d steps" % (i))
    plt.draw()
 #
plt.savefig('fig2.png')
#

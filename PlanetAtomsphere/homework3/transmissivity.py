import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

zlist = np.linspace(0,90,90)
def Tp(z):
    z_rad = np.radians(z)
    m = 1/np.cos(z_rad)
    return np.exp(-0.1114*m)
def TK(z):
    z_rad = np.radians(z)
    m = 1/(np.cos(z_rad)+0.50572*np.power(96.07995-z,-1.6364))
    return np.exp(-0.1114 * m)
fig, ax = plt.subplots(figsize=(8,8*0.618), constrained_layout=True)
ax.plot(zlist,[Tp(z) for z in zlist],c='black',ls='--',marker='o',lw=1,ms=5,label='Plane Parallel Atmosphere')
ax.plot(zlist, [TK(z) for z in zlist],c='red',ls='-.',marker='s',lw=1,ms=5,label='Kasten and Young (1989)')
ax.axvline(60,c='black',ls=':')
ax.set_xlabel('Zenith Distance [deg]',fontsize='xx-large')
ax.set_ylabel('Transmissivity',fontsize='xx-large')
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
ax.legend(loc='best',fontsize='xx-large')
plt.tick_params(right='on', top='on', which='both')
ax.tick_params(which='major',length=5,width=1.5,labelsize='x-large')
ax.tick_params(which='minor',length=3)
plt.show()
fig.savefig('Transmissivity.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)


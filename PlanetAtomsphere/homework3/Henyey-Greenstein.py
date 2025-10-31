import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def rayleigh(cos_theta):
    return (3/4)*(1+cos_theta**2)

def HG(cos_theta,g):
    return (1-g**2)/((1+g**2-2*g*cos_theta)**(3/2))

def TTHG(cos_theta,g):
    return (HG(cos_theta,g)+HG(cos_theta,-g))/2

theta = np.linspace(0,2*np.pi,100)
cos_theta = np.array([np.cos(th) for th in theta])
P_rayleigh = rayleigh(cos_theta)

g0 = 0.1
poptHG, pcovHG = curve_fit(HG, cos_theta, P_rayleigh, p0=[g0])
g_HG = poptHG[0]
print("HG fitting: g={:.6f}$\pm${:6f}".format(g_HG, np.sqrt(pcovHG[0][0])))
poptTTHG, pcovTTHG = curve_fit(TTHG, cos_theta, P_rayleigh, p0=[g0])
g_TTHG = poptTTHG[0]
print("TTHG fitting: g={:.6f}$\pm${:6f}".format(g_TTHG, np.sqrt(pcovTTHG[0][0])))
P_HG = HG(cos_theta, g_HG)
P_TTHG = TTHG(cos_theta, g_TTHG)
fig, ax = plt.subplots(figsize=(12,12),constrained_layout=True,subplot_kw={'projection': 'polar'})
ax.plot(theta,P_rayleigh, c='black',ls='-',marker='.',label='Rayleigh Scattering Phase Function')
ax.plot(theta,P_HG, c='red',ls='-.',marker='s',label='HG Fitting: '+\
        "g={:.6f}$\pm${:6f}".format(g_HG,np.sqrt(pcovHG[0][0])))
ax.plot(theta,P_TTHG, c='lime',ls='--',marker='o',label='TTHG Fitting: '+\
        "g={:.6f}$\pm${:6f}".format(g_TTHG,np.sqrt(pcovTTHG[0][0])))
ax.legend(loc='upper center',fontsize='xx-large')
ax.tick_params(which='major',labelsize='x-large')
plt.show()
fig.savefig('Henyey-Greenstein.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)
from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

x1 = 23.433 * np.pi / 180
x2 = 217.36720833 * np.pi / 180
x3 = -62.67083889 * np.pi / 180
y1 = -np.cos(x1)*np.cos(x2)
y2 = -np.sin(x2)
y3 = np.sin(x2)*np.sin(x3)*np.cos(x1)-np.cos(x3)*np.sin(x1)
y4 = -np.cos(x2)*np.sin(x3)
m = np.sqrt(y1**2+y2**2)
M = np.arctan(y1/y2)
n = np.sqrt(y3**2+y4**2)
N = np.arctan(y3/y4)
print(y1,y2,y3,y4)
print(m,M*180/np.pi,n,N*180/np.pi)
N = N + np.pi # considering the +/- symbols of y3 and y4

Para = 768.5*u.mas.to(u.deg)
print(Para)
mura = -3781.3100000*u.mas/u.yr
mudec = 769.766000*u.mas/u.yr

def lambdasun(t):
    tr=0.25+t/100
    sun = 280.460 + 36000.770*tr+0.0003877*tr**2
    return sun * np.pi / 180
def ra(t):
    RA = 217.36720833 + (mura*t*u.yr).to(u.deg).value + Para*m*np.cos(M+lambdasun(t))/np.cos(x3)
    return RA
def dec(t):
    DEC = -62.67083889 + (mudec*t*u.yr).to(u.deg).value + Para*n*np.cos(N+lambdasun(t))
    return DEC

tlist = np.linspace(0,2,2*365*4)
ralist = np.array([ra(t) for t in tlist])
declist = np.array([dec(t) for t in tlist])

fig, ax = plt.subplots(figsize=(9,9), layout='constrained')
ax.plot(ralist, declist, color='black', lw=2)
ax.scatter(ra(0),dec(0),s=150,color='red', marker='*', label='Proxima 2025.1.1')
ax.scatter(ra(0.25),dec(0.25),s=150,color='magenta', marker='*', label='Proxima t=0.25yr')
ax.scatter(ra(0.5),dec(0.5),s=150,color='orange', marker='*', label='Proxima t=0.5yr')
ax.scatter(ra(0.75),dec(0.75),s=150,color='lime', marker='*', label='Proxima t=0.75yr')
ax.scatter(ra(1),dec(1),s=150,color='cyan', marker='*', label='Proxima t=1yr')
ax.scatter(ra(1.25),dec(1.25),s=150,color='blue', marker='*', label='Proxima t=1.25yr')
ax.scatter(ra(1.5),dec(1.5),s=150,color='violet', marker='*', label='Proxima t=1.5yr')
ax.scatter(ra(1.75),dec(1.75),s=150,color='brown', marker='*', label='Proxima t=1.75yr')
ax.scatter(ra(2),dec(2),s=150,color='black', marker='*', label='Proxima t=2yr')
ax.set_xlabel('RA [deg]',fontsize='xx-large')
ax.set_ylabel('DEC [deg]',fontsize='xx-large')
plt.legend(loc='upper right', frameon=False,fontsize='xx-large')
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.0001))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.00005))
ax.tick_params(which='major',length=5,width=1.5,labelsize='x-large')
ax.tick_params(which='minor',length=3)
ax.set_title('Apparent motion of Proxima due to parallex and proper motion',size='xx-large',color='black')
# fig.savefig('Proxima.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)




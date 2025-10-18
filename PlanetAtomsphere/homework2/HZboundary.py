import numpy as np
from astropy import constants as const
import astropy.units as u

def SeffdTeq(Teff, L, a, e, Rs):
    if not (2600<=Teff<=7200):
        return 'Teff should be between 2600 and 7200 K'
    T = Teff - 5780
    # Runaway
    Seffsun1 = 1.0512
    a1 = 1.3242e-4
    b1 = 1.5418e-8
    c1 = -7.9895e-12
    d1 = -1.8328e-15
    # Moist
    Seffsun2 = 1.0140
    a2 = 8.1774e-5
    b2 = 1.7063e-9
    c2 = -4.3241e-12
    d2 = -6.6462e-16
    # Maximum
    Seffsun3 = 0.3438
    a3 = 5.8942e-5
    b3 = 1.6558e-9
    c3 = -3.0045e-12
    d3 = -5.2983e-16
    Seff1 = Seffsun1 + a1*T + b1*T**2 + c1*T**3 + d1*T**4
    Seff2 = Seffsun2 + a2*T + b2*T**2 + c2*T**3 + d2*T**4
    Seff3 = Seffsun3 + a3*T + b3*T**2 + c3*T**3 + d3*T**4
    d1 = L/Seff1
    d2 = L/Seff2
    d3 = L/Seff3
    return 'Runaway: Seff={:} d={:} AU \n'.format(Seff1, d1) + \
           'Moist: Seff={:} d={:} AU \n'.format(Seff2, d2) + \
           'Maximum: Seff={:} d={:} AU \n'.format(Seff3, d3) + \
           'a-c={:} AU a+c={:} AU \n'.format(a*(1-e), a*(1+e)) + \
           'Teq={:} K'.format(Teff*np.sqrt(Rs*const.R_sun.value/(2*(a*u.au).to(u.m).value)))

TOI_2285_b = SeffdTeq(3491, 0.0287, 0.1363, 0.30, 0.464)
print('Planet TOI-2285 b\n',TOI_2285_b)
K2_3_d = SeffdTeq(3844, 0.0587, 0.2014, 0.091, 0.546)
print('Planet k2-3 d\n',K2_3_d)
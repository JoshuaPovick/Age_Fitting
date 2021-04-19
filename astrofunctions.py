#######################
### Astro Functions ###
#######################

'''
Astro functions
'''

import numpy as np

#calculate absolute mag
def absmag(magnitude,distance):
    '''
    - magnitude: apparent magnitude of star
    - distance: distance to star in kpc
    Calculate the absolute magnitude of star
    '''
    absm = []
    absm.append(magnitude-5.0*np.log10(distance*1000)+5.0)
    absm = np.squeeze(np.array(absm))
    return absm

def sal(MH,aM):
    return MH + np.log(0.638*(10**(aM))+0.362)
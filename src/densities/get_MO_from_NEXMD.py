#   This function read the MO matrix from the "vhf.out" file produced by the NEXMD custom version

import numpy as np

def get_MO_from_NEXMD(fileName, timeStep):
    with open(fileName, 'r') as moFile:
        step = 0
        for line in moFile:
            step += 1
            if step == timeStep:
                values = np.fromstring(line, sep=' ')
                time = values[0]
                mo1D = values[1:]
                break
    nBf = int(np.sqrt(mo1D.size))
    return mo1D.reshape(nBf, nBf).T
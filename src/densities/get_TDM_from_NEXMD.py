#   This function reads the TDM from the NEXMD file "transition-densities.py"

import numpy as np

def get_TDM_from_NEXMD(fileName, state, timeStep, printTDM):
    tdm = []
    with open(fileName, 'r') as tdmFile:
        for line in tdmFile:
            if int(line.split()[0]) == state and float(line.split()[1]) == timeStep:
                values = np.fromstring(line, sep=' ')
                tdm = values[2:]
                break
    if tdm == []:
        print('TDM not found for state ' + str(state) + ' and timeStep ' + str(timeStep))
        return None
    if printTDM: # The complete matrix was written:
        nBf = int(np.sqrt(tdm.size))
        return tdm.reshape(nBf, nBf)
    else: # Only the diagonal part of the TDM was written:
        return tdm
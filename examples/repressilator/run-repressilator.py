import sys
from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
## from scipy import signal

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(sys.argv[1])

import cudasim
import cudasim.EulerMaruyama as EulerMaruyama
import cudasim.Gillespie as Gillespie
import cudasim.Lsoda as Lsoda

# define time structure
twidth = 0.5
times = arange(0,100,twidth)
nt = len(times)

# simulate signal
def generate_signal(n,times):
    ret = zeros([n, len(times)])

    parameters = zeros( [n,6] )
    species = zeros([n,6])

    for i in range(n):
        # parameters[i,:] = [ 1, 2, 2, 5, 1000, 1]
        parameters[i,:] = [ 1, uniform(1,3), uniform(1,3), uniform(2,10), uniform(10,1500), 1]
        species[i,:] = [0, 1, 0, 2, 0, 5]

    cudaCode = "repressilator_ode.cu"
    modelInstance = Lsoda.Lsoda(times, cudaCode)
    result = modelInstance.run(parameters, species)

    ret = result[:,0,:,5]

    return ret

def main():

    nsim = 100
    sig = generate_signal(nsim,times)

    for j in range(nsim):
         plt.plot(times, sig[j,:])
         pp.savefig()
         plt.close()
    pp.close()

main()

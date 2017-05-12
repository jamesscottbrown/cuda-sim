import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import *
from numpy.random import *
import cudasim.solvers.cuda.Lsoda as Lsoda


def generate_signal(n, t):
    parameters = zeros([n, 6])
    species = zeros([n, 6])
    for i in range(n):
        parameters[i, :] = [1, uniform(1, 3), uniform(1, 3), uniform(2, 10), uniform(10, 1500), 1]
        species[i, :] = [0, 1, 0, 2, 0, 5]

    cuda_code = "repressilator_ode.cu"
    model_instance = Lsoda.Lsoda(t, cuda_code)

    result = model_instance.run(parameters, species)
    return result[:, 0, :, 5]

if __name__ == "__main__":
    times = arange(0, 100, 0.5)
    num_simulations = 100
    results = generate_signal(num_simulations, times)

    file_name = "results.pdf"
    pp = PdfPages(file_name)

    for j in range(num_simulations):
        plt.plot(times, results[j, :])
        pp.savefig()
        plt.close()
    pp.close()

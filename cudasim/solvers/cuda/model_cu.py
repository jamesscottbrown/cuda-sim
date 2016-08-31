import numpy

import cudasim.solvers.cuda.Lsoda as Lsoda
import cudasim.solvers.cuda.EulerMaruyama as EulerMaruyama
import cudasim.solvers.cuda.Gillespie as Gillespie

class CudaModel:
    # instantiation
    def __init__(self, name, nspecies, nparameters, prior, x0prior, source, integration, fit, dt, beta, timepoints,
                 logp):
        self.nspecies = nspecies
        self.name = name

        # combine the parameters with the species
        self.kparameters = nparameters
        self.nparameters = nparameters + nspecies

        self.prior = prior
        self.prior.extend(x0prior)

        self.source = source
        self.integration = integration
        self.fit = fit
        self.cudaCode = self.name + '.cu'
        self.dt = dt
        self.beta = beta
        self.timepoints = timepoints
        self.logp = logp

        if self.integration == 'ODE':
            self.modelInstance = Lsoda.Lsoda(self.timepoints, self.cudaCode, dt=self.dt)
        elif self.integration == 'SDE':
            self.modelInstance = EulerMaruyama.EulerMaruyama(self.timepoints, self.cudaCode, beta=self.beta, dt=self.dt)
        elif self.integration == 'Gillespie':
            self.modelInstance = Gillespie.Gillespie(self.timepoints, self.cudaCode, beta=self.beta, dt=self.dt)

    def simulate(self, p, t, n, beta):
        # note that in this function t and beta are not used as they are specified at compile time

        species = numpy.zeros([n, self.nspecies])
        pp = numpy.zeros([n, self.kparameters])

        for i in range(n):
            species[i, :] = p[i][self.kparameters:self.nparameters]

            if not self.logp:
                pp[i, :] = p[i][0:self.kparameters]
            else:
                pp[i, :] = numpy.power(10, p[i][0:self.kparameters])

        result = self.modelInstance.run(pp, species)
        return result


This is a fork of [``cuda-sim``](https://sourceforge.net/projects/cuda-sim/), which is described in Zhou Y, Liepe J,
Sheng X, Stumpf MP, Barnes C. [GPU accelerated biochemical network simulation.](http://dx.doi.org/10.1093/bioinformatics/btr015), Bioinformatics. 2011 Mar 15;27(6):874-6. **It is still somewhat experimental, and there may be bugs that were introduced in the refactoring and not yet been found. In particular, compartments with volumes != 1 may be incorrectly simulated by some simulators.**

``cuda-sim`` enables the use of [CUDA-compatible Nvidia GPUs](https://en.wikipedia.org/wiki/CUDA#Supported_GPUs) to perform
many simulations in parallel, allowing rapid sweeps over different parameter values and/or initial conditions.

It supports a range of model types, simulating each with a different algorithm: a port of LSODA to simulate
[Ordinary Differential Equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation), the
[Euler–Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) to simulate
[stochastic differential equations](https://en.wikipedia.org/wiki/Stochastic_differential_equation), a modified
[Runge-Kutta method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) to simulate
 [delay differential equations](https://en.wikipedia.org/wiki/Delay_differential_equation), and the
 [Gillespie method](https://en.wikipedia.org/wiki/Gillespie_algorithm) for exact stochastic simulations of systems of
 chemical reactions.

It can generate executable code automatically from an [SBML](http://sbml.org/) (Systems Biology Modelling Language) file.

It was originally written to simulate biochemical networks, but it can be used more generally to simulate ODEs/SDEs/DDEs.
However, if your model does not correspond to a chemical reaction network, it is probably more sensible to generate a .cu
file by hand based on the examples, rather than trying to generate one from an SBML file.

# Installation

Before trying to install and use cuda-sim you must first install python and the CUDA toolkit, and several python packages
(numpy, pycuda). Model import from ABML also requires [libsbml](http://sbml.org/Software/libSBML).

The package can then be installed like any other python package (download, ``cd``, ``python setup.py install``).

For more details see [INSTALL.md](INSTALL.md)

# Changes
This repository contains modifications made by James Scott-Brown (see commits after commit [89b231](https://github.com/jamesscottbrown/cuda-sim/commit/89b231b1894c39310fc85d17218630c226f3deb8).

Broadly:

* support for constant time-delays. Note that parameters specifying delay lengths must be defined before other variables
in the SBML file.
* general code clean-up
* support for for multiple compartments, and compartments with volumes greater than 1

## Interpretation of kineticLaws and initial conditions

Adding sensible support for compartments with a volume other than 1.0 required changes that **will affect the numerical
results of the program**.

In the initial release, if all compartment volumes are 1.0, then simulation using the ODE/SDE/MJP integration method
gives numerically equivalent results (See Figures 1 and 2 of the manual). This is clearly a desirable feature to retain:
we would like to be able to run simulations using the SDE and MJP schemes, using the same param.dat and species.dat files,
and directly compare the results.

However, if compartments have a volume other than 1, concentrations and molecule counts are not numerically equal.
I made the slightly arbitrary choice to use concentrations (rather than molecule counts) in input and output formats:
when simulating as a MJP, initial conditions are converted from concentrations to molecule counts before being passed to
the kernel, and the results are converted from molecule counts to concentrations before being output. One rationale for
this is that as deterministic models are faster to simulate (and simpler to analyse), people often begin with these before
performing stochastic simulations of the same system, and so try to make this as convenient as possible.

To be consistent with general practice in chemistry and biochemistry, concentrations are treated as moles per unit volume
(rather than molecules per unit volume), so that there is also a factor of [Avogardo's constant](https://en.wikipedia.org/wiki/Avogadro_constant) (6.022E23)
in the conversion between counts and concentrations. The decision to treat them in this way has no effect on the ODE/SDE
simulations. It does affect MJP simulations (including of the original examples, despite these
using a compartment volume of 1.0), since for the same specified initial condition the number of particles is increased
by a factor of 6.022E23; if the number in the initial condition *was already the intended number of molecules* then the
simulation will use far more than intended, resulting in a massively increased simulation time and deceptively small
stochastic effects.

**If you would like to use the old behaviour** for MJP simulations (treating the initial conditions as specifying numbers
of molecules, and the rate laws as giving propensities directly as a function of molecule counts ), supply
``useMoleculeCounts=True`` as an optional argument when calling ``Parser.importSBMLCUDA()``, and do not supply the
optional argument ``speciesCompartment`` when calling ``Gillespie.Gillespie``.

### Comments
In a SBML model, each ``kineticLaw`` attribute specifies a *reaction rate* with units substance/time (*not*
concentration/time), this differs from what is generally referred to as the "rate law" by a factor that is the volume of
the compartment in which the reaction occurs: this factor should be included in the rate law (e.g. first-order degredation
of a protein may be represented by "k1 * protein * cell").

The SBML specification indicates that references to species in a ``kineticLaw`` may refer to either their concentration
or molecule count, and includes the attribute ``hasOnlySubstanceUnits``. **Cudasim ignores this**, always taking the default
value of false and interpreting each species reference in a kinetic law as referring to the corresponding concentration.

The SBML specification also includes references to units: again, these are ignored.

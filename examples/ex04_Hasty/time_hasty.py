import os
import time

import numpy as np

import cudasim.SBMLParser as Parser
import cudasim.solvers.cuda.DelaySimulator as DelaySimulator

##### parameters #####

# Location of the SBML model file
xmlModel = "hasty.sbml"

# Location of the file containing the parameters
parameterFile = "param.dat"

# Location of the file containing the initialization values for the different species
speciesFile = "species.dat"

# Length of simulation
simulationLength = 50

# Size of timesteps (only relevant for SDEs)
dt = 0.01

# Location of the folder for saving the results
resultFolder = "./results"

# Location of the temporary folder for model specific CUDA codes
# temp folder in this file's folder will be created if not specified
temp = None

##### initialization #####

# create temp folder
if(temp == None):
    temp = os.path.join(os.path.split(os.path.realpath(__file__))[0],"temp")
    try:
        os.mkdir(temp)
    except:
        pass
# create result folder
try:
    os.mkdir(os.path.realpath(resultFolder))
except:
    pass


# Type of integration
integrationType = "DDE"

# Name of the model
name = "hasty" + "_" + integrationType

# create CUDA code from SBML model
parser = Parser.importSBMLCUDA([xmlModel],[integrationType],ModelName=[name],method=None,outpath=temp)

#determining the timepoints for the output
timepoints = np.array(np.arange(0, simulationLength, dt), dtype=np.float32)

# reading in the CUDA code
cudaCode = os.path.join(temp, name + ".cu")

# reading in parameters
parameters = []
inFile = open(parameterFile,'r').read()
lines = inFile.split("\n")
for i in range(len(lines)):
    if(lines[i].strip() == ""):
        continue
    parameters.append([])
    lineParam = lines[i].strip().split(" ")
    for j in range(len(lineParam)):
        parameters[i].append(lineParam[j])

# reading in species
species = []
inFile = open(speciesFile,'r').read()
lines = inFile.split("\n")
for i in range(len(lines)):
    if(lines[i].strip() == ""):
        continue
    species.append([])
    lineSpecies = lines[i].strip().split(" ")
    for j in range(len(lineSpecies)):
        species[i].append(lineSpecies[j])


# create model
modeInstance = DelaySimulator.DelaySimulator(timepoints, cudaCode, delays, beta=1, dt=dt)
print "Model created"

numRepeats = 5
numSimulations = [1, 10, 20, 30, 40, 60, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
times = np.zeros([len(numSimulations), numRepeats])

for ind, simulations in enumerate(numSimulations):
    # extend species and parameters
    species2 = np.tile(species, (simulations, 1))
    parameters2 = np.tile(parameters, (simulations, 1))

    for repeat in range(numRepeats):
        start = time.clock()
        result = modeInstance.run(parameters2, species2)
        end = time.clock()

        times[ind, repeat] = (end-start)
        print "Num simulations: %s, repeat %s" % (simulations, repeat)

print times
np.savetxt('times', times)


# In MATLAB:
# >> times = dlmread('Desktop/times')
# >> n = [1, 10, 20, 30, 40, 60, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
# >> figure
# >> plot( n, times, 'o-')
# >> xlabel('Number of Simulations')
# >> ylabel('Time/s')


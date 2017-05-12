import argparse
import os

import numpy as np

import cudasim.SBMLParser as Parser
import cudasim.solvers.cuda.DelaySimulator as DelaySimulator


def setup(modelFileName, integrationType):

    resultFolder = "./results"
    temp = None


    # create temp folder
    if temp is None:
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

    name = "hasty" + "_" + integrationType

    (delays, speciesCompartmentList) = Parser.importSBMLCUDA([modelFileName],[integrationType],ModelName=[name],method=None,outpath=temp)
    cudaCode = os.path.join(temp, name + ".cu")
    return delays, cudaCode


def simulate(paramFileName, speciesFile, dt, simulationLength, delays, cudaCode):
    resultFolder = "./results"

    timepoints = np.array(np.arange(simulationLength, dt), dtype=np.float32)

    modeInstance = DelaySimulator.DelaySimulator(timepoints, cudaCode, delays, beta=1, dt=dt)

    # reading in species
    species = []
    inFile = open(speciesFile,'r').read()
    lines = inFile.split("\n")
    lineSpecies = lines[0].strip().split(" ")


    # reading in parameters
    parameters = []
    inFile = open(paramFileName,'r').read()
    lines = inFile.split("\n")
    for i in range(len(lines)):
        if(lines[i].strip() == ""):
            continue
        parameters.append([])
        lineParam = lines[i].strip().split(",")
        for j in range(len(lineParam)):
            parameters[i].append(lineParam[j])

    # extend initial conditions to correct size
    species = np.tile(lineSpecies, (i, 1))
    #species = []
    #for i in range(len(lines)):
    #    species.append([])
    #    for j in range(len(lineSpecies)):
    #        species[i].append(lineSpecies[j])




    result = modeInstance.run(parameters, species)

    # write output
    out = open(os.path.join(resultFolder, paramFileName+"_result.txt"),'w')
    for i in range(len(timepoints)):
        print >>out, timepoints[i],
    print >>out, ""
    for i in range(len(result)):
        for j in range(len(result[i])):
            for l in range(len(result[i][0][0])):
                for k in range(len(timepoints)):
                    print >>out, result[i][j][k][l],
                print >>out, ""
    out.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Performs a param sweep.
    Is given a list of files, each containing a one set of (comma-separated) parameters per line.
    Produces one otput file per input file.
    """)

    parser.add_argument('model', help="SBML model file to simulate")
    parser.add_argument('paramfiles', nargs='*', help="List of files containing sets of params")
    parser.add_argument('--length', help="Length of each simulation")
    parser.add_argument('--dt', help="Time-step")
    parser.add_argument('--method', help="Simulation method: ODE, DDE, MJP, SDE")
    parser.add_argument('--ic', help="Initial conditions")

    args = parser.parse_args()

    simulationLength = 500
    if args.length:
        simulationLength = float(args.length)

    dt = 0.1
    if args.dt:
        dt = float(args.dt)

    method = "DDE"
    if args.method:
        method = args.method


    speciesFile = "species.dat"
    if args.ic:
        speciesFile = args.ic

    (d, code) = setup(args.model, method)
    for fileName in args.paramfiles:
        simulate(fileName, speciesFile, dt, simulationLength, d, code)
# from CWriter import CWriter

import re
import sys
from Parser import Parser

from cudasim.writers.GillespieCUDAWriter import GillespieCUDAWriter
from cudasim.writers.DDECUDAWriter import DDECUDAWriter
from cudasim.writers.CWriter import CWriter
from cudasim.writers.GillespiePythonWriter import GillespiePythonWriter
from cudasim.writers.ODEPythonWriter import ODEPythonWriter
from cudasim.writers.SDECUDAWriter import SdeCUDAWriter
from cudasim.writers.ODECUDAWriter import OdeCUDAWriter
from cudasim.writers.SDEPythonWriter import SDEPythonWriter


def parse_and_write(source, integration_type, model_name=None, input_path="", output_path="", use_molecule_counts=False,
                    method=1):
    """
    ***** args *****
    source:
                  a list of names of SBML files to process (each name is a string)

    integrationType:
                  a list of strings.
                  The length of this tuple is determined by the number of SBML
                  files to be parsed. Each entry describes the simulation algorithm.
                  Possible algorithms are:
                  ODE         ---   for deterministic systems; solved with odeint (scipy)
                  SDE         ---   for stochastic systems; solved with sdeint (abc)
                  Gillespie   ---   for staochastic systems; solved with GillespieAlgorithm (abc)

    ***** kwargs *****
    modelName:
                  a list of strings.
                  modelName describes the names of the parsed model files.

    method:
                  an integer number.
                  Type of noise in a stochastic system.
                  (Only implemented for stochastic systems solved with sdeint.)
                  Possible options are:
                  1 --- default
                  2 --- Ornstein-Uhlenbeck
                  3 --- geometric Brownian motion

    """

    # regular expressions for detecting integration types and integration language
    c = re.compile('C', re.IGNORECASE)
    py = re.compile('Python', re.I)
    cuda = re.compile('CUDA', re.I)

    sde = re.compile('SDE', re.I)
    gil = re.compile('Gillespie', re.I)

    # check that you have appropriate lengths of integration types and sources
    # (need equal lengths)
    if not (len(source) == len(integration_type)):
        sys.exit("\nError: Number of sources is not the same as number of integrationTypes!\n")
    # check that you have model names,
    # if not the models will be named model1, model2, etc
    if model_name is None:
        model_name = []
        for x in range(len(source)):
            model_name.append("model" + repr(x + 1))
    else:
        for x in range(len(model_name)):
            if model_name[x] == "":
                model_name[x] = "model" + repr(x + 1)

    parsed_models = []
    for model in range(len(source)):

        parsed_model = Parser(source[model], model_name[model], input_path)
        parsed_models.append(parsed_model)

        writer = False
        if cuda.search(integration_type[model]):
            if sde.search(integration_type[model]):
                writer = SdeCUDAWriter(parsed_model, output_path)
            elif gil.search(integration_type[model]):
                writer = GillespieCUDAWriter(parsed_model, output_path, use_molecule_counts=use_molecule_counts)
            else:
                writer = OdeCUDAWriter(parsed_model, output_path)

        elif c.search(integration_type[model]):
                writer = CWriter(parsed_model, output_path)

        elif py.search(integration_type[model]):
            if sde.search(integration_type[model]):
                writer = SDEPythonWriter(parsed_model, output_path, method=method)
            elif gil.search(integration_type[model]):
                writer = GillespiePythonWriter(parsed_model, output_path)
            else:
                writer = ODEPythonWriter(parsed_model, output_path)

        if writer:
            writer.write()

    return parsed_models
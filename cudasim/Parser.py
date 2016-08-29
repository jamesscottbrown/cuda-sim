from numpy import *
from libsbml import *
import re
import os
from abcsysbio.relations import *
from CWriter import CWriter
from SDEPythonWriter import SDEPythonWriter
from ODEPythonWriter import ODEPythonWriter
from GillespiePythonWriter import GillespiePythonWriter
from SDECUDAWriter import SdeCUDAWriter
from ODECUDAWriter import OdeCUDAWriter
from GillespieCUDAWriter import GillespieCUDAWriter


class Parser:

    def __init__(self, sbmlFileName, modelName, integrationType, method, inputPath="", outputPath=""):

#def importSBMLCUDA(source, integrationType, ModelName=None, method=None, useMoleculeCounts=False, outpath=""):
        """
        ***** args *****
        source:
                      a list of strings.
                      Each tuple entry describes a SBML file to be parsed.

        integrationType:
                      a list of strings.
                      The length of this tuple is determined by the number of SBML
                      files to be parsed. Each entry describes the simulation algorithm.
                      Possible algorithms are:
                      ODE         ---   for deterministic systems; solved with odeint (scipy)
                      SDE         ---   for stochastic systems; solved with sdeint (abc)
                      MJP   ---   for staochastic systems; solved with GillespieAlgorithm (abc)
                      DDE         ---  for deterministic systems with delays; solved with DelaySimulator

        ***** kwargs *****
        ModelName:
                      a list of strings.
                      ModelName describes the names of the parsed model files.

        method:
                      an integer number.
                      Type of noise in a stochastic system.
                      (Only implemented for stochastic systems solved with sdeint.)
                      Possible options are:
                      1 --- default
                      2 --- Ornstein-Uhlenbeck
                      3 --- geometric Brownian motion

        """

        # regular expressions for detecting integration types
        g = re.compile('MJP')
        o = re.compile('ODE')
        s = re.compile('SDE')
        d = re.compile('DDE')

        if not (len(source) == len(integrationType)):
            print "\nError: Number of sources is not the same as number of integrationTypes!\n"
            return

        # If model names not specified, default to model1, model2, ...
        if ModelName is None:
            ModelName = []
            for x in range(0, len(source)):
                ModelName.append("model" + repr(x + 1))


        for models in range(0, len(source)):

            # if no method is specified and the integrationType is "SDE", method type defaults to 1
            if method is None:
                if s.match(integrationType[models]):
                    method = []
                    for x in range(0, len(source)):
                        method.append(1)

            parameterId = []        # parameter IDs as given in the model
            parameterId2 = []       # new parameter IDs, of the form parameter01, parameter02, ...
            listOfParameter = []    # parameter (or compartment!) objects
            speciesId = []          # species IDs as given in the model
            speciesId2 = []         # new species IDs, of the form species01, species02, ...
            species = []            # species objects, as returned by model.getListOfSpecies()

            # Get the model
            reader = SBMLReader()
            document = reader.readSBML(source[models])
            model = document.getModel()

            # get basic model properties
            numSpecies = model.getNumSpecies()
            numReactions = model.getNumReactions()
            numGlobalParameters = model.getNumParameters()

            stoichiometricMatrix = empty([numSpecies, numReactions])

            # Add compartment volumes to lists of parameters
            listOfCompartments = model.getListOfCompartments()
            numCompartments = len(listOfCompartments)

            for i in range(0, numCompartments):
                parameterId.append(listOfCompartments[i].getId())
                parameterId2.append('compartment' + repr(i + 1))
                listOfParameter.append(model.getCompartment(i))

            # Get global parameters
            for i in range(0, numGlobalParameters):
                parameterId.append(model.getParameter(i).getId())
                if (len(parameterId2) - numCompartments) < 9:
                    parameterId2.append('parameter0' + repr(i + 1))
                else:
                    parameterId2.append('parameter' + repr(i + 1))
                listOfParameter.append(model.getParameter(i))

            ###############
            # get species #
            ###############

            reactant = []
            product = []

            S1 = []
            S2 = []

            # Get a list of species
            listOfSpecies = model.getListOfSpecies()

            for k in range(0, len(listOfSpecies)):
                species.append(listOfSpecies[k])
                speciesId.append(listOfSpecies[k].getId())
                if len(speciesId2) < 9:
                    speciesId2.append('species0' + repr(k + 1))
                else:
                    speciesId2.append('species' + repr(k + 1))

                # construct temporary placeholders
                S1.append(0.0)
                S2.append(0.0)
                reactant.append(0)
                product.append(0)

            ###############################
            # analyse the model structure #
            ###############################

            numReactants = []
            numProducts = []
            kineticLaw = []
            numLocalParameters = []

            # Get the list of reactions
            listOfReactions = model.getListOfReactions()

            # For every reaction
            for i in range(0, len(listOfReactions)):

                numReactants.append(listOfReactions[i].getNumReactants())
                numProducts.append(listOfReactions[i].getNumProducts())

                kineticLaw.append(listOfReactions[i].getKineticLaw().getFormula())
                numLocalParameters.append(listOfReactions[i].getKineticLaw().getNumParameters())

                # Zero all elements of S1 and s2
                for a in range(0, len(species)):
                    S1[a] = 0.0
                    S2[a] = 0.0

                # Fill non-zero elements of S1, such that S1[k] is the number of molecules of species[k] *consumed* when the
                # reaction happens once.
                for j in range(0, numReactants[i]):
                    reactant[j] = listOfReactions[i].getReactant(j)

                    for k in range(0, len(species)):
                        if reactant[j].getSpecies() == species[k].getId():
                            S1[k] = reactant[j].getStoichiometry()

                # Fill non-zero elements of S2, such that S2[k] is the number of molecules of species[k] *produced* when the
                # reaction happens once.
                for l in range(0, numProducts[i]):
                    product[l] = listOfReactions[i].getProduct(l)

                    for k in range(0, len(species)):
                        if product[l].getSpecies() == species[k].getId():
                            S2[k] = product[l].getStoichiometry()

                # Construct the row of the stoichiometry matrix corresponding to this reaction by subtracting S1 from S2
                for m in range(0, len(species)):
                    stoichiometricMatrix[m][i] = -S1[m] + S2[m]

                for n in range(0, numLocalParameters[i]):
                    parameterId.append(listOfReactions[i].getKineticLaw().getParameter(n).getId())
                    if (len(parameterId2) - numCompartments) < 10:
                        parameterId2.append('parameter0' + repr(len(parameterId) - numCompartments))
                    else:
                        parameterId2.append('parameter' + repr(len(parameterId) - numCompartments))
                    listOfParameter.append(listOfReactions[i].getKineticLaw().getParameter(n))

                    name = listOfReactions[i].getKineticLaw().getParameter(n).getId()
                    new_name = 'parameter' + repr(len(parameterId) - numCompartments)
                    node = model.getReaction(i).getKineticLaw().getMath()
                    new_node = rename(node, name, new_name)
                    kineticLaw[i] = formulaToString(new_node)

                for n in range(0, numCompartments):
                    name = parameterId[n]
                    new_name = 'compartment' + repr(n + 1)
                    node = model.getReaction(i).getKineticLaw().getMath()
                    new_node = rename(node, name, new_name)
                    kineticLaw[i] = formulaToString(new_node)

            #####################
            # analyse functions #
            #####################

            listOfFunctions = model.getListOfFunctionDefinitions()

            FunctionArgument = []
            FunctionBody = []

            for fun in range(0, len(listOfFunctions)):
                FunctionArgument.append([])
                for funArg in range(0, listOfFunctions[fun].getNumArguments()):
                    FunctionArgument[fun].append(formulaToString(listOfFunctions[fun].getArgument(funArg)))

                FunctionBody.append(formulaToString(listOfFunctions[fun].getBody()))

            for fun in range(0, len(listOfFunctions)):
                for funArg in range(0, listOfFunctions[fun].getNumArguments()):
                    name = FunctionArgument[fun][funArg]
                    node = listOfFunctions[fun].getBody()
                    new_node = rename(node, name, "a" + repr(funArg + 1))
                    FunctionBody[fun] = formulaToString(new_node)
                    FunctionArgument[fun][funArg] = 'a' + repr(funArg + 1)

            #################
            # analyse rules #
            #################

            # Get the list of rules
            ruleFormula = []
            ruleVariable = []
            listOfRules = model.getListOfRules()
            for ru in range(0, len(listOfRules)):
                ruleFormula.append(listOfRules[ru].getFormula())
                ruleVariable.append(listOfRules[ru].getVariable())

            ##################
            # analyse events #
            ##################

            listOfEvents = model.getListOfEvents()

            EventCondition = []
            EventVariable = []
            EventFormula = []

            for eve in range(0, len(listOfEvents)):
                EventCondition.append(formulaToString(listOfEvents[eve].getTrigger().getMath()))
                listOfAssignmentRules = listOfEvents[eve].getListOfEventAssignments()
                EventVariable.append([])
                EventFormula.append([])
                for ru in range(0, len(listOfAssignmentRules)):
                    EventVariable[eve].append(listOfAssignmentRules[ru].getVariable())
                    EventFormula[eve].append(formulaToString(listOfAssignmentRules[ru].getMath()))

            ########################################################################
            # rename parameters and species in reactions, events, rules             #
            ########################################################################

            # Get paired list of function names for substitution
            (mathCuda, mathPython) = getSubstitutionMatrix(integrationType[models], o)

            NAMES = [[], []]
            NAMES[0].append(parameterId)
            NAMES[0].append(parameterId2)
            NAMES[1].append(speciesId)
            NAMES[1].append(speciesId2)

            for nam in range(0, 2):

                for i in range(0, len(NAMES[nam][0])):
                    name = NAMES[nam][0][i]
                    new_name = NAMES[nam][1][i]

                    for k in range(0, numReactions):
                        node = model.getReaction(k).getKineticLaw().getMath()
                        new_node = rename(node, name, new_name)
                        kineticLaw[k] = formulaToString(new_node)

                    for k in range(0, len(listOfRules)):
                        node = listOfRules[k].getMath()
                        new_node = rename(node, name, new_name)
                        ruleFormula[k] = formulaToString(new_node)
                        if ruleVariable[k] == name:
                            ruleVariable[k] = new_name

                    for k in range(0, len(listOfEvents)):
                        node = listOfEvents[k].getTrigger().getMath()
                        new_node = rename(node, name, new_name)
                        EventCondition[k] = formulaToString(new_node)
                        listOfAssignmentRules = listOfEvents[k].getListOfEventAssignments()
                        for cond in range(0, len(listOfAssignmentRules)):
                            node = listOfAssignmentRules[cond].getMath()
                            new_node = rename(node, name, new_name)
                            EventFormula[k][cond] = formulaToString(new_node)
                            if EventVariable[k][cond] == name:
                                EventVariable[k][cond] = new_name

            for nam in range(0, len(mathPython)):

                for k in range(0, len(kineticLaw)):
                    if re.search(mathPython[nam], kineticLaw[k]):
                        s = kineticLaw[k]
                        s = re.sub(mathPython[nam], mathCuda[nam], s)
                        kineticLaw[k] = s

                for k in range(0, len(ruleFormula)):
                    if re.search(mathPython[nam], ruleFormula[k]):
                        s = ruleFormula[k]
                        s = re.sub(mathPython[nam], mathCuda[nam], s)
                        ruleFormula[k] = s

                for k in range(0, len(EventFormula)):
                    for cond in range(0, len(listOfAssignmentRules)):
                        if re.search(mathPython[nam], EventFormula[k][cond]):
                            s = EventFormula[k][cond]
                            s = re.sub(mathPython[nam], mathCuda[nam], s)
                            EventFormula[k][cond] = s

                for k in range(0, len(EventCondition)):
                    if re.search(mathPython[nam], EventCondition[k]):
                        s = EventCondition[k]
                        s = re.sub(mathPython[nam], mathCuda[nam], s)
                        EventCondition[k] = s

                for k in range(0, len(FunctionBody)):
                    if re.search(mathPython[nam], FunctionBody[k]):
                        s = FunctionBody[k]
                        s = re.sub(mathPython[nam], mathCuda[nam], s)
                        FunctionBody[k] = s

                for fun in range(0, len(listOfFunctions)):
                    for k in range(0, len(FunctionArgument[fun])):
                        if re.search(mathPython[nam], FunctionArgument[fun][k]):
                            s = FunctionArgument[fun][k]
                            s = re.sub(mathPython[nam], mathCuda[nam], s)
                            FunctionArgument[fun][k] = s

            # Get list of delays
            delays = set()

            print "Looking for delay"
            for n in range(0, model.getNumReactions()):
                r = model.getReaction(n)
                if r.isSetKineticLaw():
                    kl = r.getKineticLaw()

                    if kl.isSetMath():
                        formula = formulaToString(kl.getMath())

                        if "delay" in formula:
                            r = re.search("delay\((\w+?), (\w+?)\)", formula).groups()
                            paramName = r[1]
                            j = int(paramName.replace("parameter", ''))

                            memoryLocation = "tex2D(param_tex," + repr(j + len(listOfCompartments) - 1) + ",tid)"
                            delays.add(memoryLocation)

            delays = list(delays)


            # Find compartment corresponding to each species
            speciesCompartmentList = []
            for i in range(0, numSpecies):

                if species[i].isSetCompartment():
                    mySpeciesCompartment = species[i].getCompartment()
                    for j in range(0, len(listOfParameter)):
                        if listOfParameter[j].getId() == mySpeciesCompartment:
                            speciesCompartmentList.append(j)


            ##########################
            # call writing functions #
            ##########################

            s = re.compile('SDE')
            if o.match(integrationType[models]):
                write_ODECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId2, listOfParameter,
                              parameterId2, ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                              ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath)
            if s.match(integrationType[models]):
                write_SDECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId2, listOfParameter,
                              parameterId2, ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                              ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath)
            if g.match(integrationType[models]):
                write_GillespieCUDA(stoichiometricMatrix, kineticLaw, species, numReactions, parameterId2, speciesId2, listOfParameter,
                                    ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                                    ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, speciesCompartmentList,
                                    useMoleculeCounts=useMoleculeCounts, outpath=outpath)
            if d.match(integrationType[models]):
                write_DDECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId2, listOfParameter,
                              parameterId2, ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                              ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, delays, numCompartments,
                              outpath)

        return (delays, speciesCompartmentList)





















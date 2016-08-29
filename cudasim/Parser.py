from numpy import *
from libsbml import *
import re
import os
from abcsysbio.relations import *
from ParsedModel import ParsedModel

class Parser:

    def __init__(self, sbmlFileName, modelName, integrationType, method, inputPath="", outputPath=""):

        # regular expressions indicating solution language
        c = re.compile('C', re.IGNORECASE)
        py = re.compile('Python', re.I)
        cuda = re.compile('CUDA', re.I)

        # regular expressions for detecting integration types
        gil = re.compile('MJP')
        ode = re.compile('ODE')
        sde = re.compile('SDE')
        dde = re.compile('DDE')

        reader = SBMLReader()
        document = reader.readSBML(inputPath + sbmlFileName)
        self.sbmlModel = document.getModel()

        self.parameterId = []

        self.listOfSpecies = []  # Used by the child
        self.speciesId = []

        self.product = []
        self.reactant = []
        self.S1 = []
        self.S2 = []

        self.listOfReactions = []  # Used by the child
        self.listOfAssignmentRules = []
        self.numLocalParameters = []  # Used by the child

        self.comp = 0
        self.parse()

        self.parsedModel = ParsedModel()


        def parse(self):
            self.getBasicModelProperties()
            self.parsedModel.stoichiometricMatrix = empty(
                [self.parsedModel.numSpecies, self.parsedModel.numReactions])
            self.getCompartmentVolume()
            self.get_delays()

        def getBasicModelProperties(self):
            self.parsedModel.numSpecies = self.sbmlModel.getNumSpecies()
            self.parsedModel.numReactions = self.sbmlModel.getNumReactions()
            self.parsedModel.numGlobalParameters = self.sbmlModel.getNumParameters()

        def getCompartmentVolume(self):
            # Add compartment volumes to lists of parameters
            listOfCompartments = self.sbmlModel.getListOfCompartments()

            for i in range(len(listOfCompartments)):
                self.comp += 1
                self.parameterId.append(listOfCompartments[i].getId())
                self.parsedModel.parameterId.append('compartment' + repr(i + 1))
                self.parsedModel.parameter.append(listOfCompartments[i].getVolume())
                self.parsedModel.listOfParameter.append(self.sbmlModel.getCompartment(i))

        def getGlobalParameters(self):
            # Differs between CUDA and Python/C
            for i in range(self.parsedModel.numGlobalParameters):
                self.parameterId.append(self.sbmlModel.getParameter(i).getId())
                self.parsedModel.parameter.append(self.sbmlModel.getParameter(i).getValue())
                self.parsedModel.listOfParameter.append(self.sbmlModel.getParameter(i))

        def getSpecies(self):
            # Differs between CUDA and Python/C
            self.listOfSpecies = self.sbmlModel.getListOfSpecies()

            for k in range(len(self.listOfSpecies)):
                self.parsedModel.species.append(self.listOfSpecies[k])
                self.speciesId.append(self.listOfSpecies[k].getId())

                self.S1.append(0.0)
                self.S2.append(0.0)

                self.reactant.append(0)
                self.product.append(0)

                # Only used by the python writer:
                self.parsedModel.initValues.append(self.getSpeciesValue(self.listOfSpecies[k]))

        def analyseModelStructure(self):
            # Differs between CUDA and Python/C
            reaction = []
            numReactants = []
            numProducts = []

            self.listOfReactions = self.sbmlModel.getListOfReactions()

            # For every reaction
            for i in range(len(self.listOfReactions)):
                numReactants.append(self.listOfReactions[i].getNumReactants())
                numProducts.append(self.listOfReactions[i].getNumProducts())

                self.parsedModel.kineticLaw.append(self.listOfReactions[i].getKineticLaw().getFormula())
                self.numLocalParameters.append(self.listOfReactions[i].getKineticLaw().getNumParameters())

                # Zero all elements of S1 and s2
                for a in range(len(self.parsedModel.species)):
                    self.S1[a] = 0.0
                    self.S2[a] = 0.0

                # Fill non-zero elements of S1, such that S1[k] is the number of molecules of species[k] *consumed* when the
                # reaction happens once.
                for j in range(numReactants[i]):
                    self.reactant[j] = self.listOfReactions[i].getReactant(j)

                    for k in range(len(self.parsedModel.species)):
                        if self.reactant[j].getSpecies() == self.parsedModel.species[k].getId():
                            self.S1[k] = self.reactant[j].getStoichiometry()

                # Fill non-zero elements of S2, such that S2[k] is the number of molecules of species[k] *produced* when the
                # reaction happens once.
                for l in range(numProducts[i]):
                    self.product[l] = self.listOfReactions[i].getProduct(l)

                    for k in range(len(self.parsedModel.species)):
                        if self.product[l].getSpecies() == self.parsedModel.species[k].getId():
                            self.S2[k] = self.product[l].getStoichiometry()

                # Construct the row of the stoichiometry matrix corresponding to this reaction by subtracting S1 from S2
                for m in range(len(self.parsedModel.species)):
                    self.parsedModel.stoichiometricMatrix[m][i] = -self.S1[m] + self.S2[m]

                for n in range(self.numLocalParameters[i]):
                    self.parsedModel.parameter.append(
                        self.listOfReactions[i].getKineticLaw().getParameter(n).getValue())
                    self.parsedModel.listOfParameter.append(self.listOfReactions[i].getKineticLaw().getParameter(n))

                for n in range(self.comp):
                    compartment_name = self.parameterId[n]
                    new_name = 'compartment' + repr(n + 1)
                    node = self.sbmlModel.getReaction(i).getKineticLaw().getMath()
                    new_node = self.rename(node, compartment_name, new_name)
                    self.parsedModel.kineticLaw[i] = formulaToString(new_node)

        def analyseFunctions(self):
            sbmlListOfFunctions = self.sbmlModel.getListOfFunctionDefinitions()

            for fun in range(len(sbmlListOfFunctions)):
                self.parsedModel.listOfFunctions.append(sbmlListOfFunctions[fun])
                self.parsedModel.functionArgument.append([])
                self.parsedModel.functionBody.append(
                    formulaToString(self.parsedModel.listOfFunctions[fun].getBody()))

                for funArg in range(self.parsedModel.listOfFunctions[fun].getNumArguments()):
                    self.parsedModel.functionArgument[fun].append(
                        formulaToString(self.parsedModel.listOfFunctions[fun].getArgument(funArg)))
                    old_name = self.parsedModel.functionArgument[fun][funArg]
                    node = self.parsedModel.listOfFunctions[fun].getBody()
                    new_node = self.rename(node, old_name, "a" + repr(funArg + 1))
                    self.parsedModel.functionBody[fun] = formulaToString(new_node)
                    self.parsedModel.functionArgument[fun][funArg] = "a" + repr(funArg + 1)

        def analyseRules(self):
            self.parsedModel.listOfRules = self.sbmlModel.getListOfRules()
            for rule in range(len(self.parsedModel.listOfRules)):
                self.parsedModel.ruleFormula.append(self.parsedModel.listOfRules[rule].getFormula())
                self.parsedModel.ruleVariable.append(self.parsedModel.listOfRules[rule].getVariable())


        def analyseEvents(self):
            self.parsedModel.listOfEvents = self.sbmlModel.getListOfEvents()
            for event in range(len(self.parsedModel.listOfEvents)):
                self.parsedModel.eventCondition.append(
                    formulaToString(self.parsedModel.listOfEvents[event].getTrigger().getMath()))
                self.listOfAssignmentRules = self.parsedModel.listOfEvents[event].getListOfEventAssignments()
                self.parsedModel.eventVariable.append([])
                self.parsedModel.eventFormula.append([])

                for rule in range(len(self.listOfAssignmentRules)):
                    self.parsedModel.eventVariable[event].append(self.listOfAssignmentRules[rule].getVariable())
                    self.parsedModel.eventFormula[event].append(
                        formulaToString(self.listOfAssignmentRules[rule].getMath()))

    def renameEverything(self):

        NAMES = [[], []]
        NAMES[0].append(self.parameterId)
        NAMES[0].append(self.parsedModel.parameterId)
        NAMES[1].append(self.speciesId)
        NAMES[1].append(self.parsedModel.speciesId)

        for nam in range(2):

            for i in range(len(NAMES[nam][0])):
                old_name = NAMES[nam][0][i]
                new_name = NAMES[nam][1][i]

                for k in range(self.parsedModel.numReactions):
                    node = self.sbmlModel.getReaction(k).getKineticLaw().getMath()
                    new_node = self.rename(node, old_name, new_name)
                    self.parsedModel.kineticLaw[k] = formulaToString(new_node)

                for k in range(len(self.parsedModel.listOfRules)):
                    node = self.parsedModel.listOfRules[k].getMath()
                    new_node = self.rename(node, old_name, new_name)
                    self.parsedModel.ruleFormula[k] = formulaToString(new_node)
                    if self.parsedModel.ruleVariable[k] == old_name:
                        self.parsedModel.ruleVariable[k] = new_name

                for k in range(len(self.parsedModel.listOfEvents)):
                    node = self.parsedModel.listOfEvents[k].getTrigger().getMath()
                    new_node = self.rename(node, old_name, new_name)
                    self.parsedModel.eventCondition[k] = formulaToString(new_node)
                    self.listOfAssignmentRules = self.parsedModel.listOfEvents[k].getListOfEventAssignments()

                    for cond in range(len(self.listOfAssignmentRules)):
                        node = self.listOfAssignmentRules[cond].getMath()
                        new_node = self.rename(node, old_name, new_name)
                        self.parsedModel.eventFormula[k][cond] = formulaToString(new_node)
                        if self.parsedModel.eventVariable[k][cond] == old_name:
                            self.parsedModel.eventVariable[k][cond] = new_name

    def rename(self, node, old_name, new_name):
        typ = node.getType()

        if typ == AST_NAME or typ == AST_NAME_TIME:
            nme = node.getName()
            if nme == old_name:
                node.setName(new_name)

        for n in range(node.getNumChildren()):
            self.rename(node.getChild(n), old_name, new_name)
        return node

    def getSpeciesValue(self, specie):
        if specie.isSetInitialAmount() and specie.isSetInitialConcentration():
            return specie.getInitialConcentration()  # The initial values are only used in ODE and SDE solvers so we take the concentration (if it was used in gillespie we would have taken the value)
        if specie.isSetInitialAmount():
            return specie.getInitialAmount()
        else:
            return specie.getInitialConcentration()

    def get_delays(self):

        delays = set()

        print "Looking for delay"
        for n in range(0, self.sbmlModel.getNumReactions()):
            r = self.sbmlModel.getReaction(n)
            if r.isSetKineticLaw():
                kl = r.getKineticLaw()

                if kl.isSetMath():
                    formula = formulaToString(kl.getMath())

                    if "delay" in formula:
                        r = re.search("delay\((\w+?), (\w+?)\)", formula).groups()
                        paramName = r[1]
                        j = int(paramName.replace("parameter", ''))

                        listOfCompartments = self.sbmlModel.getListOfCompartments()
                        memoryLocation = "tex2D(param_tex," + repr(j + len(listOfCompartments) - 1) + ",tid)"
                        delays.add(memoryLocation)

        self.parsedModel.delays = list(delays)

    def get_species_compartments(self):
        # Find compartment corresponding to each species
        self.parsedModel.speciesCompartmentList = []
        for i in range(0, self.parsedModel.numSpecies):

            if self.parsedModel.species[i].isSetCompartment():
                mySpeciesCompartment = self.parsedModel.species[i].getCompartment()
                for j in range(0, len(self.parsedModel.listOfParameter)):
                    if self.parsedModel.listOfParameter[j].getId() == mySpeciesCompartment:
                        self.parsedModel.speciesCompartmentList.append(j)



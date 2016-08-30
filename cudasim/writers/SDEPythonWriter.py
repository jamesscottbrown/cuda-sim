from libsbml import *
from cudasim.relations import *
import os
import re
from Writer import Writer


class SDEPythonWriter(Writer):
    def __init__(self, parser, outputPath=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(outputPath, self.parser.parsedModel.name + ".cu"), "w")
        self.rename()


    def rename(self):
        """
        This function renames parts of self.parser.parsedModel to meet the specific requirements of this writer.
        This behaviour replaces the previous approach of subclassing the parser to produce different results depending
        on the which writer was intended to be used.
        """

        # Remove any zero-padding from single-digit parameter names
        # This reverses any change applied by one of the CUDA writers
        for i in range(self.parser.comp-1, len(self.parser.parsedModel.parameterId)):
            old_name = self.parser.parsedModel.parameterId[i]
            num = old_name[len('parameter'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'parameter' + str(num[1:])
                self.parser.parsedModel.parameterId[i] = new_name
                self.parser.parsedModel.rename_everywhere(old_name, new_name)

        # Remove any zero-padding from single-digit species names
        # This reverses any change applied by one of the CUDA writers
        for i in range(0, len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'species' + str(num[1:])
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.parsedModel.rename_everywhere(old_name, new_name)

    def write(self, method=1):
        p = re.compile('\s')
        self.out_file.write("from math import sqrt\nfrom numpy import random\nfrom cudasim.relations import *\n\n")

        self.out_file.write("def trunc_sqrt(x):\n\tif x < 0: return 0\n\telse: return sqrt(x)\n\n")

        for i in range(len(self.parser.parsedModel.listOfFunctions)):
            self.out_file.write("def ")
            self.out_file.write(self.parser.parsedModel.listOfFunctions[i].getId())
            self.out_file.write("(")
            for j in range(self.parser.parsedModel.listOfFunctions[i].getNumArguments()):
                self.out_file.write(self.parser.parsedModel.functionArgument[i][j])
                self.out_file.write(",")
            self.out_file.write("):\n\n\toutput=")
            self.out_file.write(self.parser.parsedModel.functionBody[i])
            self.out_file.write("\n\n\treturn output\n\n")

        self.out_file.write("def modelfunction((")

        for i in range(len(self.parser.parsedModel.species)):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write(self.parser.parsedModel.parameterId[i])
                        self.out_file.write(",")

        self.out_file.write(")=(")

        for i in range(len(self.parser.parsedModel.species)):
            self.out_file.write(repr(self.parser.parsedModel.initValues[i]))
            self.out_file.write(",")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write(repr(self.parser.parsedModel.parameter[i]))
                        self.out_file.write(",")

        self.out_file.write("),dt=0,parameter=(")

        for i in range(len(self.parser.parsedModel.parameterId)):
            dontPrint = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        dontPrint = True
            if not dontPrint:
                self.out_file.write(repr(self.parser.parsedModel.parameter[i]))
                self.out_file.write(",")

                ##for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == True):
                ##    self.out_file.write(repr(self.parser.parsedModel.initValues[i]))
                ##    self.out_file.write(",")

        self.out_file.write("),time=0):\n\n")

        counter = 0
        for i in range(len(self.parser.parsedModel.parameterId)):
            dontPrint = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]: dontPrint = True
            if not dontPrint:
                self.out_file.write("\t" + self.parser.parsedModel.parameterId[i] + "=parameter[" + repr(counter) + "]\n")
                counter += 1

                ##for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == True):
                ##    self.out_file.write("\t"+self.parser.parsedModel.speciesId[i]+"=parameter["+repr(counter)+"]\n")
                ##    counter = counter+1

        self.out_file.write("\n")

        self.out_file.write("\n")

        for i in range(self.parser.parsedModel.numSpecies):
            ##if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write("\td_" + self.parser.parsedModel.speciesId[i] + "=")
            if self.parser.parsedModel.species[i].isSetCompartment():
                self.out_file.write("(")
            for k in range(self.parser.parsedModel.numReactions):
                if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
                    self.out_file.write("(")
                    self.out_file.write(repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                    self.out_file.write(")*(")
                    string = p.sub('', self.parser.parsedModel.kineticLaw[k])
                    self.out_file.write(string)
                    self.out_file.write(")+")
            self.out_file.write("0")
            if self.parser.parsedModel.species[i].isSetCompartment():
                self.out_file.write(")/")
                mySpeciesCompartment = self.parser.parsedModel.species[i].getCompartment()
                for j in range(len(self.parser.parsedModel.listOfParameter)):
                    if self.parser.parsedModel.listOfParameter[j].getId() == mySpeciesCompartment:
                        self.out_file.write(self.parser.parsedModel.parameterId[j])
                        break
            self.out_file.write("\n")

        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isRate():
                self.out_file.write("\td_")
                self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(self.parser.parsedModel.ruleFormula[i])
                self.out_file.write("\n")

                ##################################################
                # noise terms
                ##################################################

        self.out_file.write("\n")

        # check for columns of the stochiometry matrix with more than one entry
        randomVariables = ["*random.normal(0.0,sqrt(dt))"] * self.parser.parsedModel.numReactions
        for k in range(self.parser.parsedModel.numReactions):
            countEntries = 0
            for i in range(self.parser.parsedModel.numSpecies):
                if self.parser.parsedModel.stoichiometricMatrix[i][k] != 0.0:
                    countEntries += 1

            # define specific randomVariable
            if countEntries > 1:
                self.out_file.write("\trand" + repr(k) + " = random.normal(0.0,sqrt(dt))\n")
                randomVariables[k] = "*rand" + repr(k)

        if method == 1:

            for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == False):
                self.out_file.write("\tnoise_" + self.parser.parsedModel.speciesId[i] + "=")
                for k in range(self.parser.parsedModel.numReactions):
                    if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
                        self.out_file.write("(" + repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write(")*trunc_sqrt(")
                        string = p.sub('', self.parser.parsedModel.kineticLaw[k])
                        self.out_file.write(string)
                        self.out_file.write(")")
                        self.out_file.write(randomVariables[k])
                        self.out_file.write("+")
                self.out_file.write("0\n")

            for i in range(len(self.parser.parsedModel.listOfRules)):
                if self.parser.parsedModel.listOfRules[i].isRate():
                    self.out_file.write("\tnoise_")
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                    self.out_file.write("= trunc_sqrt(")
                    self.out_file.write(self.parser.parsedModel.ruleFormula[i])
                    self.out_file.write(")")
                    self.out_file.write(randomVariables[k])
                    self.out_file.write("\n")

        if method == 2:

            for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == False):
                self.out_file.write("\tnoise_" + self.parser.parsedModel.speciesId[i] + "=")
                self.out_file.write("random.normal(0.0,sqrt(dt))\n")

            for i in range(len(self.parser.parsedModel.listOfRules)):
                if self.parser.parsedModel.listOfRules[i].isRate():
                    self.out_file.write("\tnoise_")
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                    self.out_file.write("= ")
                    self.out_file.write("random.normal(0.0,sqrt(dt))\n")

        if method == 3:

            for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == False):
                self.out_file.write("\tnoise_" + self.parser.parsedModel.speciesId[i] + "=")
                for k in range(self.parser.parsedModel.numReactions):
                    if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
                        self.out_file.write("(")
                        self.out_file.write(repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write(")*(")
                        string = p.sub('', self.parser.parsedModel.kineticLaw[k])
                        self.out_file.write(string)
                        self.out_file.write(")*")
                        self.out_file.write("random.normal(0.0,sqrt(dt))")
                        self.out_file.write("+")
                self.out_file.write("0\n")

            for i in range(len(self.parser.parsedModel.listOfRules)):
                if self.parser.parsedModel.listOfRules[i].isRate():
                    self.out_file.write("\tnoise_")
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                    self.out_file.write("= (")
                    self.out_file.write(self.parser.parsedModel.ruleFormula[i])
                    self.out_file.write(" ) * random.normal(0.0,sqrt(dt))")
                    self.out_file.write("\n")

        self.out_file.write("\n\treturn((")

        for i in range(len(self.parser.parsedModel.species)):
            ##if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write("d_" + self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write("d_" + self.parser.parsedModel.parameterId[i])
                        self.out_file.write(",")

        self.out_file.write("),(")
        for i in range(self.parser.parsedModel.numSpecies):
            ##if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write("noise_" + self.parser.parsedModel.speciesId[i])
            self.out_file.write(", ")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write("noise_" + self.parser.parsedModel.parameterId[i])
                        self.out_file.write(",")

        self.out_file.write("))\n\n")

        # Write the assignment rules
        self.out_file.write("\ndef rules((")

        for i in range(len(self.parser.parsedModel.species)):
            ##if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write(self.parser.parsedModel.parameterId[i])
                        self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            dontPrint = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        dontPrint = True
            if not dontPrint:
                self.out_file.write(self.parser.parsedModel.parameterId[i])
                self.out_file.write(",")

                ##for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == True):
                ##    self.out_file.write(self.parser.parsedModel.speciesId[i])
                ##    self.out_file.write(",")

        self.out_file.write("),time):\n\n")

        # Write the events

        for i in range(len(self.parser.parsedModel.listOfEvents)):
            self.out_file.write("\tif ")
            self.out_file.write(mathMLConditionParser(self.parser.parsedModel.eventCondition[i]))
            self.out_file.write(":\n")
            listOfAssignmentRules = self.parser.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(listOfAssignmentRules)):
                self.out_file.write("\t\t")
                self.out_file.write(self.parser.parsedModel.eventVariable[i][j])
                self.out_file.write("=")
                self.out_file.write(self.parser.parsedModel.eventFormula[i][j])
                self.out_file.write("\n")
            self.out_file.write("\n")

        self.out_file.write("\n")

        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("\t")
                self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(self.parser.parsedModel.ruleFormula[i])
                self.out_file.write("\n")

        self.out_file.write("\n\treturn((")
        for i in range(self.parser.parsedModel.numSpecies):
            ##if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write(self.parser.parsedModel.parameterId[i])
                        self.out_file.write(",")
        self.out_file.write("),(")

        for i in range(len(self.parser.parsedModel.parameterId)):
            dontPrint = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        dontPrint = True
            if not dontPrint:
                self.out_file.write(self.parser.parsedModel.parameterId[i])
                self.out_file.write(",")

                ##for i in range(self.parser.parsedModel.numSpecies):
                ##if (self.parser.parsedModel.species[i].getConstant() == True):
                ##self.out_file.write(self.parser.parsedModel.speciesId[i])
                ##self.out_file.write(",")
        self.out_file.write("))\n\n")
        self.out_file.close()

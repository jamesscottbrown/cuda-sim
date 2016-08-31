from libsbml import *
from cudasim.relations import *
import os
import re
import sys
from Writer import Writer


class GillespiePythonWriter(Writer):
    def __init__(self, parser, outputPath=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(outputPath, self.parser.parsedModel.name + ".py"), "w")
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
                self.parser.rename_everywhere(old_name, new_name)

        # Remove any zero-padding from single-digit species names
        # This reverses any change applied by one of the CUDA writers
        for i in range(0, len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'species' + str(num[1:])
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

    def write(self):
        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isRate():
                print "\n Model '" + self.parser.parsedModel.name + "' contains at least one rate rule. This model can not be parsed and simmulated with the Gillespie algorithm! Please change the simmulation Type! \n"
                sys.exit()

        p = re.compile('\s')

        self.out_file.write("from cudasim.relations import *\n\n#Functions\n")

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

        self.out_file.write("\n#Gillespie Hazards\n\n")

        self.out_file.write("def Hazards((")

        for i in range(self.parser.parsedModel.numSpecies):
            # if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")

        self.out_file.write("),parameter):\n\n")

        for i in range(len(self.parser.parsedModel.parameterId)):
            self.out_file.write("\t" + self.parser.parsedModel.parameterId[i] + "=parameter[" + repr(i) + "]\n")

        counter = len(self.parser.parsedModel.parameterId)
        # for i in range(self.parser.parsedModel.numSpecies):
        # if (self.parser.parsedModel.species[i].getConstant() == True):
        #    self.out_file.write("\t"+self.parser.parsedModel.speciesId[i]+"=parameter["+repr(counter)+"]\n")
        #    counter = counter+1

        self.out_file.write("\n")

        for i in range(self.parser.parsedModel.numReactions):
            self.out_file.write("\tHazard_" + repr(i) + " = " + self.parser.parsedModel.kineticLaw[i])
            self.out_file.write("\n")

        self.out_file.write("\n\treturn(")

        for i in range(self.parser.parsedModel.numReactions):
            self.out_file.write("Hazard_" + repr(i))
            if not i == (self.parser.parsedModel.numReactions - 1):
                self.out_file.write(", ")

        self.out_file.write(")\n\n")

        self.out_file.write("#Gillespie Reactions\n\n")

        for i in range(self.parser.parsedModel.numReactions):
            self.out_file.write("def Reaction" + repr(i) + "((")
            for k in range(self.parser.parsedModel.numSpecies):
                # if (self.parser.parsedModel.species[k].getConstant() == False):
                self.out_file.write(self.parser.parsedModel.speciesId[k])
                self.out_file.write(",")

            self.out_file.write(")):\n\n")

            for k in range(self.parser.parsedModel.numSpecies):
                # if (self.parser.parsedModel.species[k].getConstant() == False):
                self.out_file.write(
                    "\t" + self.parser.parsedModel.speciesId[k] + "_new=" + self.parser.parsedModel.speciesId[k] + "+(" + str(
                        self.parser.parsedModel.stoichiometricMatrix[k][i]) + ")\n")

            self.out_file.write("\n\treturn(")
            for k in range(self.parser.parsedModel.numSpecies):
                # if (self.parser.parsedModel.species[k].getConstant() == False):
                self.out_file.write(self.parser.parsedModel.speciesId[k] + "_new")
                self.out_file.write(",")
            self.out_file.write(")\n\n")

        self.out_file.write("#Dictionary of reactions\ndef defaultfunc():\n\tpass\n\ndef Switch():\n\tswitch = {\n")
        for i in range(self.parser.parsedModel.numReactions):
            self.out_file.write("\t\t" + repr(i) + " : Reaction" + repr(i) + ",\n")

        self.out_file.write("\t\t\"default\": defaultfunc\n\t\t}\n\treturn switch\n\n")

        self.out_file.write("#Rules and Events\n")

        self.out_file.write("def rules((")
        for i in range(self.parser.parsedModel.numSpecies):
            # if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            self.out_file.write(self.parser.parsedModel.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("),t):\n\n")

        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("\t")
                self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(self.parser.parsedModel.ruleFormula[i])
                self.out_file.write("\n")

        self.out_file.write("\n\treturn((")
        for i in range(self.parser.parsedModel.numSpecies):
            # if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")

        self.out_file.write("),(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            self.out_file.write(self.parser.parsedModel.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("))\n\n")

        self.out_file.write("def events((")
        for i in range(self.parser.parsedModel.numSpecies):
            # if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            self.out_file.write(self.parser.parsedModel.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("),t):\n\n")

        for i in range(len(self.parser.parsedModel.listOfEvents)):
            self.out_file.write("\tif ")
            self.out_file.write(mathMLConditionParser(self.parser.parsedModel.eventCondition[i]))
            self.out_file.write(":\n")
            list_of_assignment_rules = self.parser.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("\t\t")
                self.out_file.write(self.parser.parsedModel.eventVariable[i][j])
                self.out_file.write("=")
                self.out_file.write(self.parser.parsedModel.eventFormula[i][j])
                self.out_file.write("\n")
            self.out_file.write("\n")

        self.out_file.write("\n\treturn((")
        for i in range(self.parser.parsedModel.numSpecies):
            # if (self.parser.parsedModel.species[i].getConstant() == False):
            self.out_file.write(self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")

        self.out_file.write("),(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            self.out_file.write(self.parser.parsedModel.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("))\n\n")

        self.out_file.close()

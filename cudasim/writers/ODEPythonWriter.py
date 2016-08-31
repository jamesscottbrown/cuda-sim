import os
from libsbml import *
from cudasim.relations import *
from Writer import Writer


class ODEPythonWriter(Writer):
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
        p = re.compile('\s')
        # Import the necessaries
        self.out_file.write("from math import *\nfrom numpy import *\nfrom cudasim.relations import *\n\n")
        # The user-defined functions used in the model must be written in the file

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

        # Write the modelfunction

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

        self.out_file.write("),time,parameter=(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            dont_print = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(repr(self.parser.parsedModel.parameter[i]))
                self.out_file.write(",")

        self.out_file.write(")):\n\n")

        counter = 0
        for i in range(len(self.parser.parsedModel.parameterId)):
            dont_print = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == self.parser.parsedModel.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write("\t" + self.parser.parsedModel.parameterId[i] + "=parameter[" + repr(counter) + "]\n")
                counter += 1

        self.out_file.write("\n")

        # write the derivatives

        for i in range(self.parser.parsedModel.numSpecies):
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
                my_species_compartment = self.parser.parsedModel.species[i].getCompartment()
                for j in range(len(self.parser.parsedModel.listOfParameter)):
                    if self.parser.parsedModel.listOfParameter[j].getId() == my_species_compartment:
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

        self.out_file.write("\n\treturn(")

        for i in range(len(self.parser.parsedModel.species)):
            self.out_file.write("d_" + self.parser.parsedModel.speciesId[i])
            self.out_file.write(",")
        for i in range(len(self.parser.parsedModel.listOfParameter)):
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        self.out_file.write("d_" + self.parser.parsedModel.parameterId[i])
                        self.out_file.write(",")

        self.out_file.write(")\n")

        # Write the rules
        self.out_file.write("\ndef rules((")

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
        self.out_file.write("),(")
        for i in range(len(self.parser.parsedModel.parameterId)):
            dont_print = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(self.parser.parsedModel.parameterId[i])
                self.out_file.write(",")

        self.out_file.write("),time):\n\n")

        # Write the events

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

        self.out_file.write("\n")

        # write the rules

        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("\t")
                self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(mathMLConditionParser(self.parser.parsedModel.ruleFormula[i]))
                self.out_file.write("\n")

        self.out_file.write("\n\treturn((")
        for i in range(self.parser.parsedModel.numSpecies):
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
            dont_print = False
            if not self.parser.parsedModel.listOfParameter[i].getConstant():
                for k in range(len(self.parser.parsedModel.listOfRules)):
                    if self.parser.parsedModel.listOfRules[k].isRate() and self.parser.parsedModel.ruleVariable[k] == \
                            self.parser.parsedModel.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(self.parser.parsedModel.parameterId[i])
                self.out_file.write(",")
        self.out_file.write("))\n\n")
        self.out_file.close()

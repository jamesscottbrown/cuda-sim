from libsbml import *
from cudasim.relations import *
import os
import sys
from Writer import Writer


class GillespiePythonWriter(Writer):
    def __init__(self, parser, output_path=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(output_path, self.parser.parsedModel.name + ".py"), "w")
        self.rename()

    def rename(self):
        """
        This function renames parts of model to meet the specific requirements of this writer.
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
        for i in range(len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'species' + str(num[1:])
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

    def write(self):

        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isRate():
                sys.exit("\n Model '" + self.parser.parsedModel.name + "' contains at least one rate rule, so " +
                         "cannot be simulated with the Gillespie algorithm! Please change the simmulation Type! \n")

        self.out_file.write("from cudasim.relations import *\n\n#Functions\n")
        
        self.write_functions()
        self.write_hazards_function()
        self.write_stoichiometry_functions()
        self.write_reaction_dictionary()
        
        self.out_file.write("#Rules and Events\n")
        
        self.write_rules_function()
        self.write_events_function()
        
        self.out_file.close()

    def write_functions(self):
        # Write one function per reaction, which will ajusst the state by adding the corresponding stoichiometric vector
        model = self.parser.parsedModel
        for i in range(len(model.listOfFunctions)):
            self.out_file.write("def ")
            self.out_file.write(model.listOfFunctions[i].getId())
            self.out_file.write("(")
            for j in range(model.listOfFunctions[i].getNumArguments()):
                self.out_file.write(model.functionArgument[i][j])
                self.out_file.write(",")
            self.out_file.write("):\n\n\toutput=")
            self.out_file.write(model.functionBody[i])
            self.out_file.write("\n\n\treturn output\n\n")

    def write_reaction_dictionary(self):
        # Writes a function that maps from reaction index to name of the corresponding stoichiometry function
        model = self.parser.parsedModel
        self.out_file.write("#Dictionary of reactions\ndef defaultfunc():\n\tpass\n\ndef Switch():\n\tswitch = {\n")
        for i in range(model.numReactions):
            self.out_file.write("\t\t" + repr(i) + " : Reaction" + repr(i) + ",\n")
        self.out_file.write("\t\t\"default\": defaultfunc\n\t\t}\n\treturn switch\n\n")

    def write_stoichiometry_functions(self):
        model = self.parser.parsedModel
        self.out_file.write("#Gillespie Reactions\n\n")
        for i in range(model.numReactions):
            self.out_file.write("def Reaction" + repr(i) + "((")
            for k in range(model.numSpecies):
                self.out_file.write(model.speciesId[k])
                self.out_file.write(",")

            self.out_file.write(")):\n\n")

            for k in range(model.numSpecies):
                self.out_file.write(
                    "\t" + model.speciesId[k] + "_new=" + model.speciesId[k] +
                    "+(" + str(model.stoichiometricMatrix[k][i]) + ")\n")

            self.out_file.write("\n\treturn(")
            for k in range(model.numSpecies):
                self.out_file.write(model.speciesId[k] + "_new")
                self.out_file.write(",")
            self.out_file.write(")\n\n")

    def write_hazards_function(self):
        model = self.parser.parsedModel

        self.out_file.write("\n#Gillespie Hazards\n\n")
        self.out_file.write("def Hazards((")
        for i in range(model.numSpecies):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),parameter):\n\n")
        for i in range(len(model.parameterId)):
            self.out_file.write("\t" + model.parameterId[i] + "=parameter[" + repr(i) + "]\n")
        self.out_file.write("\n")
        for i in range(model.numReactions):
            self.out_file.write("\tHazard_" + repr(i) + " = " + model.kineticLaw[i])
            self.out_file.write("\n")
        self.out_file.write("\n\treturn(")
        for i in range(model.numReactions):
            self.out_file.write("Hazard_" + repr(i))
            if not i == (model.numReactions - 1):
                self.out_file.write(", ")
        self.out_file.write(")\n\n")

    def write_rules_function(self):
        model = self.parser.parsedModel

        self.out_file.write("def rules((")
        for i in range(model.numSpecies):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(model.parameterId)):
            self.out_file.write(model.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("),t):\n\n")
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("\t")
                self.out_file.write(model.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(model.ruleFormula[i])
                self.out_file.write("\n")
        self.out_file.write("\n\treturn((")
        for i in range(model.numSpecies):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(model.parameterId)):
            self.out_file.write(model.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("))\n\n")

    def write_events_function(self):
        model = self.parser.parsedModel

        self.out_file.write("def events((")
        for i in range(model.numSpecies):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(model.parameterId)):
            self.out_file.write(model.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("),t):\n\n")
        for i in range(len(model.listOfEvents)):
            self.out_file.write("\tif ")
            self.out_file.write(mathml_condition_parser(model.eventCondition[i]))
            self.out_file.write(":\n")
            list_of_assignment_rules = model.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("\t\t")
                self.out_file.write(model.eventVariable[i][j])
                self.out_file.write("=")
                self.out_file.write(model.eventFormula[i][j])
                self.out_file.write("\n")
            self.out_file.write("\n")
        self.out_file.write("\n\treturn((")
        for i in range(model.numSpecies):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(model.parameterId)):
            self.out_file.write(model.parameterId[i])
            self.out_file.write(',')
        self.out_file.write("))\n\n")

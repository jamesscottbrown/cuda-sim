import os
from cudasim.relations import *
from Writer import Writer


class ODEPythonWriter(Writer):
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
        self.out_file.write("from math import *\nfrom numpy import *\nfrom cudasim.relations import *\n\n")

        # write one function per user-defined function
        self.write_functions()

        # write the function called 'modelfunction'
        self.write_modelfunction_signature()
        self.write_parameter_assignments()
        self.write_derivatives()

        # write the function called 'rules'
        self.write_rules_function_signature()
        self.write_events()
        self.write_rules_function_body()

    def write_functions(self):
        # Writes user-defined functions
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

    def write_modelfunction_signature(self):
        # Write the signature for a modelfunction, e.g.:
        # def modelfunction((species1,),time,parameter=(1.0,1.0,0.1,)):

        model = self.parser.parsedModel
        self.out_file.write("def modelfunction((")
        for i in range(len(model.species)):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        for i in range(len(model.listOfParameter)):
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        self.out_file.write(model.parameterId[i])
                        self.out_file.write(",")
        self.out_file.write("),time,parameter=(")
        for i in range(len(model.parameterId)):
            dont_print = False
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(repr(model.parameter[i]))
                self.out_file.write(",")
        self.out_file.write(")):\n\n")

    def write_parameter_assignments(self):
        # Write paramter assignment statements, e.g.
        # compartment1=parameter[0]

        model = self.parser.parsedModel
        counter = 0
        for i in range(len(model.parameterId)):
            dont_print = False
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(
                    "\t" + model.parameterId[i] + "=parameter[" + repr(counter) + "]\n")
                counter += 1
        self.out_file.write("\n")

    def write_derivatives(self):
        p = re.compile('\s')
        model = self.parser.parsedModel

        for i in range(model.numSpecies):
            self.out_file.write("\td_" + model.speciesId[i] + "=")
            if model.species[i].isSetCompartment():
                self.out_file.write("(")
            for k in range(model.numReactions):
                if not model.stoichiometricMatrix[i][k] == 0.0:
                    self.out_file.write("(")
                    self.out_file.write(repr(model.stoichiometricMatrix[i][k]))
                    self.out_file.write(")*(")
                    string = p.sub('', model.kineticLaw[k])
                    self.out_file.write(string)
                    self.out_file.write(")+")
            self.out_file.write("0")
            if model.species[i].isSetCompartment():
                self.out_file.write(")/")
                my_species_compartment = model.species[i].getCompartment()
                for j in range(len(model.listOfParameter)):
                    if model.listOfParameter[j].getId() == my_species_compartment:
                        self.out_file.write(model.parameterId[j])
                        break
            self.out_file.write("\n")
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isRate():
                self.out_file.write("\td_")
                self.out_file.write(model.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(model.ruleFormula[i])
                self.out_file.write("\n")
        self.out_file.write("\n\treturn(")
        for i in range(len(model.species)):
            self.out_file.write("d_" + model.speciesId[i])
            self.out_file.write(",")
        for i in range(len(model.listOfParameter)):
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        self.out_file.write("d_" + model.parameterId[i])
                        self.out_file.write(",")
        self.out_file.write(")\n")

    def write_rules_function_signature(self):
        model = self.parser.parsedModel
        self.out_file.write("\ndef rules((")
        for i in range(len(model.species)):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        for i in range(len(model.listOfParameter)):
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        self.out_file.write(model.parameterId[i])
                        self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(model.parameterId)):
            dont_print = False
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(model.parameterId[i])
                self.out_file.write(",")
        self.out_file.write("),time):\n\n")

    def write_events(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfEvents)):
            self.out_file.write("\tif ")
            self.out_file.write(mathMLConditionParser(model.eventCondition[i]))
            self.out_file.write(":\n")
            list_of_assignment_rules = model.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("\t\t")
                self.out_file.write(model.eventVariable[i][j])
                self.out_file.write("=")
                self.out_file.write(model.eventFormula[i][j])
                self.out_file.write("\n")
            self.out_file.write("\n")
        self.out_file.write("\n")

    def write_rules_function_body(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("\t")
                self.out_file.write(model.ruleVariable[i])
                self.out_file.write("=")
                self.out_file.write(mathMLConditionParser(model.ruleFormula[i]))
                self.out_file.write("\n")
        self.out_file.write("\n\treturn((")
        for i in range(model.numSpecies):
            self.out_file.write(model.speciesId[i])
            self.out_file.write(",")
        for i in range(len(model.listOfParameter)):
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        self.out_file.write(model.parameterId[i])
                        self.out_file.write(",")
        self.out_file.write("),(")
        for i in range(len(model.parameterId)):
            dont_print = False
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == \
                            model.parameterId[i]:
                        dont_print = True
            if not dont_print:
                self.out_file.write(model.parameterId[i])
                self.out_file.write(",")
        self.out_file.write("))\n\n")
        self.out_file.close()

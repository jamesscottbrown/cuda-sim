import os
import copy
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
        self.out_file.close()

    def write_functions(self):
        # Writes user-defined functions
        model = self.parser.parsedModel
        for i in range(len(model.listOfFunctions)):
            arg_list = ','.join(model.functionArgument[i])

            self.out_file.write("def %s (%s):\n\n" % (model.listOfFunctions[i].getId(), arg_list))
            self.out_file.write("\toutput = %s\n\n" % model.functionBody[i])
            self.out_file.write("\treturn output\n\n")

    def categorise_variables(self):
        # form a list of the species, and parameters which are set by rate rules
        model = self.parser.parsedModel
        species_list = copy.copy(model.speciesId)

        rule_params = []
        constant_params = []
        for i in range(len(model.listOfParameter)):
            is_constant = True
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == model.parameterId[i]:
                        rule_params.append(model.parameterId[i])
                        is_constant = False
            if is_constant:
                constant_params.append(model.parameterId[i])

        species_list.extend(rule_params)
        return species_list, constant_params

    def write_modelfunction_signature(self):
        # Write the signature for a modelfunction, e.g.:
        # def modelfunction((species1,),time,parameter=(1.0,1.0,0.1,)):

        (species_list, constant_params) = self.categorise_variables()
        self.out_file.write("def modelfunction((%s), time, parameter=(%s)):\n\n" %
                            (",".join(species_list), ",".join(constant_params)))

    def write_parameter_assignments(self):
        # Write paramter assignment statements, e.g.
        # compartment1=parameter[0]
        (species_list, constant_params) = self.categorise_variables()
        for i, param in enumerate(constant_params):
            self.out_file.write("\t%s = parameter[%s]\n" % (param, i))

    def write_derivatives(self):
        p = re.compile('\s')
        model = self.parser.parsedModel

        for i in range(model.numSpecies):
            self.out_file.write("\td_" + model.speciesId[i] + "=")
            if model.species[i].isSetCompartment():
                self.out_file.write("(")
            for k in range(model.numReactions):
                if not model.stoichiometricMatrix[i][k] == 0.0:
                    string = p.sub('', model.kineticLaw[k])
                    self.out_file.write("(%s)*(%s)+" % (repr(model.stoichiometricMatrix[i][k]), string))

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
                self.out_file.write("\td_%s = %s\n" % (model.ruleVariable[i], model.ruleFormula[i]))

        # print return statement
        (species_list, _) = self.categorise_variables()
        species_string = ",".join(map(lambda x: "d_%s" % x, species_list))
        self.out_file.write("\n\treturn(%s)\n" % species_string)

    def write_rules_function_signature(self):
        (species_list, constant_params) = self.categorise_variables()
        self.out_file.write("\ndef rules((%s), (%s), time):\n\n" % (",".join(species_list), ",".join(constant_params)))

    def write_events(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfEvents)):
            self.out_file.write("\tif %s:\n" % mathml_condition_parser(model.eventCondition[i]))
            list_of_assignment_rules = model.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("\t\t%s = %s\n" % (model.eventVariable[i][j], model.eventFormula[i][j]))
            self.out_file.write("\n")
        self.out_file.write("\n")

    def write_rules_function_body(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("\t%s = %s\n" %
                                    (model.ruleVariable[i], mathml_condition_parser(model.ruleFormula[i])))

        (species_list, constant_params) = self.categorise_variables()
        self.out_file.write("\n\treturn((%s),(%s))\n\n" % (",".join(species_list), ",".join(constant_params)))

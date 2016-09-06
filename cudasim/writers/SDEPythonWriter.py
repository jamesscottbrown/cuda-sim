from cudasim.relations import *
import os
import re
from Writer import Writer


class SDEPythonWriter(Writer):
    def __init__(self, parser, output_path=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(output_path, self.parser.parsedModel.name + ".py"), "w")
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
        for i in range(len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'species' + str(num[1:])
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

    def write(self, method=1):

        self.out_file.write("from math import sqrt\nfrom numpy import random\nfrom cudasim.relations import *\n\n")
        self.out_file.write("def trunc_sqrt(x):\n\tif x < 0: return 0\n\telse: return sqrt(x)\n\n")

        # write one function per rule
        self.write_functions()

        # write the 'methodfunction' function
        self.write_methodfunction_signature()
        self.write_parameter_assignments()
        self.write_reaction_rates(method)
        self.write_methodfunction_return_statement()

        # Write the 'rules' function
        self.write_rule_function_signature()
        self.write_events()
        self.write_assignment_rules()
        self.write_rulesfunction_return_statement()

        self.out_file.close()

    def write_parameter_assignments(self):
        (species_list, constant_params, _, _) = self.categorise_variables()
        for i, param in enumerate(constant_params):
            self.out_file.write("\t%s = parameter[%s]\n" % (param, i))
        self.out_file.write("\n\n")

    def write_methodfunction_signature(self):
        species_list, constant_params, species_values, constant_values = self.categorise_variables()
        self.out_file.write("def modelfunction((%s)=(%s),dt=0,parameter=(%s),time=0):\n\n" %
                            (",".join(species_list), ",".join(species_values), ",".join(constant_values)))

    def write_reaction_rates(self, method):
        model = self.parser.parsedModel
        p = re.compile('\s')

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

        ##################################################
        # noise terms
        ##################################################
        self.out_file.write("\n")
        # check for columns of the stochiometry matrix with more than one entry
        random_variables = ["*random.normal(0.0,sqrt(dt))"] * model.numReactions
        for k in range(model.numReactions):
            num_entries = 0
            for i in range(model.numSpecies):
                if model.stoichiometricMatrix[i][k] != 0.0:
                    num_entries += 1

            # define specific randomVariable
            if num_entries > 1:
                self.out_file.write("\trand" + repr(k) + " = random.normal(0.0,sqrt(dt))\n")
                random_variables[k] = "*rand" + repr(k)

        if method == 1:

            for i in range(model.numSpecies):
                self.out_file.write("\tnoise_" + model.speciesId[i] + "=")
                for k in range(model.numReactions):
                    if not model.stoichiometricMatrix[i][k] == 0.0:
                        string = p.sub('', model.kineticLaw[k])
                        self.out_file.write("(%s)*trunc_sqrt(%s)%s+" %
                                            (repr(model.stoichiometricMatrix[i][k]), string, random_variables[k]))
                self.out_file.write("0\n")

            for i in range(len(model.listOfRules)):
                if model.listOfRules[i].isRate():
                    self.out_file.write("\tnoise_%s = trunc_sqrt(%s)%s\n" %
                                        (model.ruleVariable[i], model.ruleFormula[i], random_variables[k]))

        if method == 2:

            for i in range(model.numSpecies):
                self.out_file.write("\tnoise_%s = random.normal(0.0,sqrt(dt))\n" % model.speciesId[i])

            for i in range(len(model.listOfRules)):
                if model.listOfRules[i].isRate():
                    self.out_file.write("\tnoise_%s = random.normal(0.0,sqrt(dt))\n" % model.ruleVariable[i])

        if method == 3:

            for i in range(model.numSpecies):
                self.out_file.write("\tnoise_%s =" % (model.speciesId[i]))
                for k in range(model.numReactions):
                    if not model.stoichiometricMatrix[i][k] == 0.0:
                        string = p.sub('', model.kineticLaw[k])
                        self.out_file.write("(%s)*(%s)*random.normal(0.0,sqrt(dt)) + " %
                                            (repr(model.stoichiometricMatrix[i][k]), string))
                self.out_file.write("0\n")

            for i in range(len(model.listOfRules)):
                if model.listOfRules[i].isRate():
                    self.out_file.write("\tnoise_%s = (%s) * random.normal(0.0,sqrt(dt))\n" %
                                        (model.ruleVariable[i], model.ruleFormula[i]))

    def write_methodfunction_return_statement(self):
        (species_list, constant_params, _, _) = self.categorise_variables()
        species_vars = ",".join(map(lambda x: "d_" + x, species_list))
        noise_vars = ",".join(map(lambda x: "noise_" + x, species_list))
        self.out_file.write("\n\treturn((%s),(%s))\n\n" % (species_vars, noise_vars))

    def write_functions(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfFunctions)):
            self.out_file.write("def %s(%s)\n\n:" %
                                (model.listOfFunctions[i].getId(), ",".join(model.functionArgument[i])))
            self.out_file.write("\toutput = %s\n\n" % model.functionBody[i])
            self.out_file.write("\treturn output\n\n")

    def write_rule_function_signature(self):
        (species_list, constant_params, _, _) = self.categorise_variables()
        self.out_file.write("\ndef rules((%s),(%s),time):\n\n" % (",".join(species_list), ",".join(constant_params)))

    def write_events(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfEvents)):
            self.out_file.write("\tif %s:\n" % (mathml_condition_parser(model.eventCondition[i])))
            list_of_assignment_rules = model.listOfEvents[i].getListOfEventAssignments()

            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("\t\t%s = model.eventFormula[i][j]\n" % model.eventVariable[i][j])
            self.out_file.write("\n")
        self.out_file.write("\n")

    def write_assignment_rules(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("\t%s = %s\n" % (model.ruleVariable[i], model.ruleFormula[i]))

    def write_rulesfunction_return_statement(self):
        (species_list, constant_params, _, _) = self.categorise_variables()
        self.out_file.write("\n\treturn((%s),(%s))\n\n" % (",".join(species_list), ",".join(constant_params)))

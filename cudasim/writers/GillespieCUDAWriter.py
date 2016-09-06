import os
import re

from cudasim.writers.Writer import Writer
from cudasim.cuda_helpers import rename_math_functions
from cudasim.relations import mathml_condition_parser


class GillespieCUDAWriter(Writer):
    def __init__(self, parser, output_path=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(output_path, self.parser.parsedModel.name + ".cu"), "w")
        self.rename()

    def rename(self):
        """
        This function renames parts of model to meet the specific requirements of this writer.
        This behaviour replaces the previous approach of subclassing the parser to produce different results depending
        on the which writer was intended to be used.
        """

        # Pad single-digit parameter names with a leading zero
        for i in range(self.parser.comp-1, len(self.parser.parsedModel.parameterId)):
            old_name = self.parser.parsedModel.parameterId[i]
            num = old_name[len('parameter'):]
            if len(num) < 2:
                new_name = 'parameter' + '0' + str(num)
                self.parser.parsedModel.parameterId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

        # Pad single-digit species names with a leading zero
        for i in range(len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) < 2:
                new_name = 'species' + '0' + str(num)
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

        rename_math_functions(self.parser.parsedModel, 't')

    def write(self, use_molecule_counts=False):

        p = re.compile('\s')
        model = self.parser.parsedModel
        # Open the outfile
    
        num_species = len(model.species)
    
        num_events = len(model.listOfEvents)
        num_rules = len(model.listOfRules)
        num = num_events + num_rules
    
        self.write_header(num, num_species)
        self.write_functions()
        self.write_stoichiometry_matrix()
        
        if use_molecule_counts:
            self.out_file.write("__device__ void hazards(int *y, float *h, float t, int tid){\n")
            self.out_file.write("        // Assume rate law expressed in terms of molecule counts \n")
        else:
            self.out_file.write("__device__ void hazards(int *yCounts, float *h, float t, int tid){")

            self.out_file.write("""
            // Calculate concentrations from molecule counts
            int y[NSPECIES];
            """)
            for i in range(num_species):
                volume_string = "tex2D(param_tex," + repr(model.speciesCompartmentList[i]) + ",tid)"
                self.out_file.write("y[%s] = yCounts[%s] / (6.022E23 * %s);\n" % (i, i, volume_string))

        self.write_rate_rules()
        self.write_events()
        self.write_assignment_rules()
        self.write_reaction_hazards(p, use_molecule_counts)
        
        self.out_file.write("}\n\n")

    def write_header(self, num, num_species):
        model = self.parser.parsedModel
        self.out_file.write("#define NSPECIES " + str(num_species) + "\n")
        self.out_file.write("#define NPARAM " + str(len(model.parameterId)) + "\n")
        self.out_file.write("#define NREACT " + str(model.numReactions) + "\n")
        self.out_file.write("\n")
        if num > 0:
            self.out_file.write("#define leq(a,b) a<=b\n")
            self.out_file.write("#define neq(a,b) a!=b\n")
            self.out_file.write("#define geq(a,b) a>=b\n")
            self.out_file.write("#define lt(a,b) a<b\n")
            self.out_file.write("#define gt(a,b) a>b\n")
            self.out_file.write("#define eq(a,b) a==b\n")
            self.out_file.write("#define and_(a,b) a&&b\n")
            self.out_file.write("#define or_(a,b) a||b\n")

    def write_functions(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfFunctions)):
            self.out_file.write("__device__ float " + model.listOfFunctions[i].getId() + "(")
            for j in range(model.listOfFunctions[i].getNumArguments()):
                self.out_file.write("float " + model.functionArgument[i][j])
                if j < (model.listOfFunctions[i].getNumArguments() - 1):
                    self.out_file.write(",")
            self.out_file.write("){\n    return ")
            self.out_file.write(model.functionBody[i])
            self.out_file.write(";\n}\n")
            self.out_file.write("")

    def write_stoichiometry_matrix(self):
        model = self.parser.parsedModel
        self.out_file.write("\n\n__constant__ int smatrix[]={\n")
        for i in range(len(model.stoichiometricMatrix[0])):
            for j in range(len(model.stoichiometricMatrix)):
                self.out_file.write("    " + repr(model.stoichiometricMatrix[j][i]))
                if not (i == (len(model.stoichiometricMatrix) - 1) and (j == (len(model.stoichiometricMatrix[0]) - 1))):
                    self.out_file.write(",")
            self.out_file.write("\n")
        self.out_file.write("};\n\n\n")

    def write_rate_rules(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isRate():
                self.out_file.write("    ")

                rule_variable = model.ruleVariable[i]
                if not (rule_variable in model.speciesId):
                    self.out_file.write(rule_variable)
                else:
                    string = "y[" + repr(model.speciesId.index(rule_variable)) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")

                string = model.ruleFormula[i]
                for q in range(len(model.speciesId)):
                    string = self.rep(string, model.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(len(model.parameterId)):
                    parameter_id = model.parameterId[q]
                    if not (parameter_id in model.ruleVariable):
                        flag = False
                        for r in range(len(model.eventVariable)):
                            if parameter_id in model.eventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, parameter_id, 'tex2D(param_tex,' + repr(q) + ',tid)')

                self.out_file.write(string)
                self.out_file.write(";\n")

    def write_events(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfEvents)):
            self.out_file.write("    if( ")
            self.out_file.write(mathml_condition_parser(model.EventCondition[i]))
            self.out_file.write("){\n")
            list_of_assignment_rules = model.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("        ")

                event_variable = model.eventVariable[i][j]
                if not (event_variable in model.speciesId):
                    self.out_file.write(event_variable)
                else:
                    string = "y[" + repr(model.speciesId.index(event_variable)) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")

                string = model.EventFormula[i][j]
                for q in range(len(model.speciesId)):
                    string = self.rep(string, model.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(len(model.parameterId)):
                    parameter_id = model.parameterId[q]
                    if not (parameter_id in model.ruleVariable):
                        flag = False
                        for r in range(len(model.eventVariable)):
                            if parameter_id in model.eventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, parameter_id, 'tex2D(param_tex,' + repr(q) + ',tid)')

                self.out_file.write(string)
                self.out_file.write(";\n")
            self.out_file.write("    }\n")
        self.out_file.write("\n")

    def write_assignment_rules(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("    ")

                rule_variable = model.ruleVariable[i]
                if not (rule_variable in model.speciesId):
                    self.out_file.write("float ")
                    self.out_file.write(rule_variable)
                else:
                    string = "y[" + repr(model.speciesId.index(rule_variable)) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")

                string = mathml_condition_parser(model.ruleFormula[i])
                for q in range(len(model.speciesId)):
                    string = self.rep(string, model.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(len(model.parameterId)):
                    parameter_id = model.parameterId[q]
                    if not (parameter_id in model.ruleVariable):
                        flag = False
                        for r in range(len(model.eventVariable)):
                            if parameter_id in model.eventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, parameter_id, 'tex2D(param_tex,' + repr(q) + ',tid)')
                self.out_file.write(string)
                self.out_file.write(";\n")
        self.out_file.write("\n")

    def write_reaction_hazards(self, p, use_molecule_counts):
        model = self.parser.parsedModel
        for i in range(model.numReactions):
            if use_molecule_counts:
                self.out_file.write("    h[" + repr(i) + "] = ")
            else:
                self.out_file.write("    h[" + repr(i) + "] = 6.022E23 * ")

            string = model.kineticLaw[i]
            for q in range(len(model.speciesId)):
                string = self.rep(string, model.speciesId[q], 'y[' + repr(q) + ']')
            for q in range(len(model.parameterId)):
                parameter_id = model.parameterId[q]
                if not (parameter_id in model.ruleVariable):
                    flag = False
                    for r in range(len(model.eventVariable)):
                        if parameter_id in model.eventVariable[r]:
                            flag = True
                    if not flag:
                        string = self.rep(string, parameter_id, 'tex2D(param_tex,' + repr(q) + ',tid)')

            string = p.sub('', string)
            self.out_file.write(string + ";\n")
        self.out_file.write("\n")

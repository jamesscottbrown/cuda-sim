import os
import re

from cudasim.writers.Writer import Writer
from cudasim.cuda_helpers import rename_math_functions
from cudasim.relations import mathml_condition_parser


class OdeCUDAWriter(Writer):
    def __init__(self, parser, output_path=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(output_path, self.parser.parsedModel.name + ".cu"), "w")
        self.rename()

    def rename(self):
        """
        This function renames parts of self.parser.parsedModel to meet the specific requirements of this writer.
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

        rename_math_functions(self.parser.parsedModel, 't[0]')

    def write(self):
        """
        Write the cuda file with ODE functions using the information taken by the parser
        """
    
        num_species = len(self.parser.parsedModel.species)

        self.write_header(num_species)
        self.write_functions()

        self.out_file.write("""
struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        """)

        self.write_rate_rules()
        self.write_events()
        self.write_assignment_rules()
        self.write_derivatives(num_species)
        self.out_file.write("\n    }")
        self.out_file.write("""
};


struct myJex{
__device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/)
{
    return;
}
};
""")

    def write_header(self, num_species):
        model = self.parser.parsedModel
        self.out_file.write("#define NSPECIES " + str(num_species) + "\n")
        self.out_file.write("#define NPARAM " + str(len(model.parameterId)) + "\n")
        self.out_file.write("#define NREACT " + str(model.numReactions) + "\n")
        self.out_file.write("\n")

        # An expression of the form pow((function of state), (parameter), causes a function call with signature
        # "pow(<double, float>)", an illegal call to the __host__ function std::pow from the __device__ function.
        # To avoid this, we create wrapper functions  that cast the float to a double then call pow(<double>,<double>),
        #  a __device__ function.
        self.out_file.write("float __device__ pow(double i, float j){ return pow(i, (double) j); }")
        self.out_file.write("float __device__ pow(float i, double j){ return pow((double) i, j); }")

    def write_functions(self):
        model = self.parser.parsedModel
        num_events = len(model.listOfEvents)
        num_rules = len(model.listOfRules)
        num = num_events + num_rules

        if num > 0:
            self.out_file.write("#define leq(a,b) a<=b\n")
            self.out_file.write("#define neq(a,b) a!=b\n")
            self.out_file.write("#define geq(a,b) a>=b\n")
            self.out_file.write("#define lt(a,b) a<b\n")
            self.out_file.write("#define gt(a,b) a>b\n")
            self.out_file.write("#define eq(a,b) a==b\n")
            self.out_file.write("#define and_(a,b) a&&b\n")
            self.out_file.write("#define or_(a,b) a||b\n")

        functions = model.listOfFunctions
        for i in range(len(functions)):
            args_list = ",".join(map(lambda x: "double %s" % x, model.functionArgument[i]))

            self.out_file.write("__device__ double %s(%s){\n" % (functions[i].getId(), args_list))
            self.out_file.write("    return %s; \n" % model.functionBody[i])
            self.out_file.write("}\n")

    def write_rate_rules(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isRate():
                self.out_file.write("        ")

                rule_variable = model.ruleVariable[i]
                if not (rule_variable in model.speciesId):
                    self.out_file.write(rule_variable)
                else:
                    string = "y[" + repr(model.speciesId.index(rule_variable)) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")

                string = model.ruleFormula[i]
                for q in range(len(model.speciesId)):
                    pq = re.compile(model.speciesId[q])
                    string = pq.sub('y[' + repr(q) + ']', string)
                for q in range(len(model.parameterId)):
                    parameter_id = model.parameterId[q]
                    if not (parameter_id in model.ruleVariable):
                        flag = False
                        for r in range(len(model.eventVariable)):
                            if parameter_id in model.eventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(parameter_id)
                            string = pq.sub('tex2D(param_tex,' + repr(q) + ',tid)', string)

                self.out_file.write(string)
                self.out_file.write(";\n")

    def write_events(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfEvents)):
            self.out_file.write("    if(%s){\n " % mathml_condition_parser(model.EventCondition[i]))
            list_of_assignment_rules = model.listOfEvents[i].getListOfEventAssignments()
            for j in range(len(list_of_assignment_rules)):
                self.out_file.write("        ")
                event_variable = model.eventVariable[i][j]
                if not (event_variable in model.speciesId):
                    self.out_file.write(event_variable)
                else:
                    self.out_file.write("y[%s]" % repr(model.speciesId.index(event_variable)))
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
            self.out_file.write("}\n")
        self.out_file.write("\n")

    def write_assignment_rules(self):
        model = self.parser.parsedModel
        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("    ")
                rule_variable = model.ruleVariable[i]
                if not (rule_variable in model.speciesId):
                    self.out_file.write("double %s = " % rule_variable)
                else:
                    self.out_file.write("y[%s] = " % repr(model.speciesId.index(rule_variable)))

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
        self.out_file.write("\n\n")

    def write_derivatives(self, num_species):
        p = re.compile('\s')
        model = self.parser.parsedModel

        for i in range(num_species):
            species = model.species[i]
            if not (species.getConstant() or species.getBoundaryCondition()):
                self.out_file.write("        ydot[" + repr(i) + "]=")
                if species.isSetCompartment():
                    self.out_file.write("(")

                reaction_written = False
                for k in range(model.numReactions):
                    if not model.stoichiometricMatrix[i][k] == 0.0:

                        if reaction_written and model.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reaction_written = True
                        self.out_file.write(repr(model.stoichiometricMatrix[i][k]))
                        self.out_file.write("*(")

                        string = model.kineticLaw[k]
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

                        # print string
                        self.out_file.write(string)
                        self.out_file.write(")")

                if species.isSetCompartment():
                    self.out_file.write(")/")
                    my_species_compartment = species.getCompartment()
                    for j in range(len(model.listOfParameter)):
                        if model.listOfParameter[j].getId() == my_species_compartment:
                            parameter_id = model.parameterId[j]
                            if not (parameter_id in model.ruleVariable):
                                flag = False
                                for r in range(len(model.eventVariable)):
                                    if parameter_id in model.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ";")
                                    break
                                else:
                                    self.out_file.write(parameter_id + ";")
                                    break

                else:
                    self.out_file.write(";")
                self.out_file.write("\n")

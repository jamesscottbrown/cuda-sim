import os
import re

from cudasim.writers.Writer import Writer
from cudasim.cuda_helpers import rename_math_functions


class SdeCUDAWriter(Writer):
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

        # Pad single-digit parameter names with a leading zero
        for i in range(self.parser.comp-1, len(self.parser.parsedModel.parameterId)):
            old_name = self.parser.parsedModel.parameterId[i]
            num = old_name[len('parameter'):]
            if len(num) < 2:
                new_name = 'parameter' + '0' + str(num)
                self.parser.parsedModel.parameterId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

        # Pad single-digit species names with a leading zero
        for i in range(0, len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) < 2:
                new_name = 'species' + '0' + str(num)
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

        rename_math_functions(self.parser.parsedModel, 't')

    def write(self):
        """
        Write the cuda file with ODE functions using the information taken by the parser
        """
    
        num_species = len(self.parser.parsedModel.species)
    
        p = re.compile('\s')
    
        # Write number of parameters and species
        self.out_file.write("#define NSPECIES " + str(num_species) + "\n")
        self.out_file.write("#define NPARAM " + str(len(self.parser.parsedModel.parameterId)) + "\n")
        self.out_file.write("#define NREACT " + str(self.parser.parsedModel.numReactions) + "\n")
        self.out_file.write("\n")
    
        # The user-defined functions used in the model must be written in the file
        self.out_file.write("//Code for texture memory\n")
    
        num_events = len(self.parser.parsedModel.listOfEvents)
        num_rules = len(self.parser.parsedModel.listOfRules)
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
    
        for i in range(0, len(self.parser.parsedModel.listOfFunctions)):
            self.out_file.write("__device__ float " + self.parser.parsedModel.listOfFunctions[i].getId() + "(")
            for j in range(0, self.parser.parsedModel.listOfFunctions[i].getNumArguments()):
                self.out_file.write("float " + self.parser.parsedModel.functionArgument[i][j])
                if j < (self.parser.parsedModel.listOfFunctions[i].getNumArguments() - 1):
                    self.out_file.write(",")
            self.out_file.write("){\n    return ")
            self.out_file.write(self.parser.parsedModel.functionBody[i])
            self.out_file.write(";\n}\n")
            self.out_file.write("\n")
    
        self.out_file.write("\n")
    
        self.out_file.write("__device__ void step(float *y, float t, unsigned *rngRegs, int tid){\n")
    
        num_species = len(self.parser.parsedModel.species)
    
        # write rules and events
        for i in range(0, len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isRate():
                self.out_file.write("    ")
                if not (self.parser.parsedModel.ruleVariable[i] in self.parser.parsedModel.speciesId):
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parser.parsedModel.ruleFormula[i]
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.eventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        for i in range(0, len(self.parser.parsedModel.listOfEvents)):
            self.out_file.write("    if( ")
            self.out_file.write(self.mathMLConditionParserCuda(self.parser.parsedModel.EventCondition[i]))
            self.out_file.write("){\n")
            list_of_assignment_rules = self.parser.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(0, len(list_of_assignment_rules)):
                self.out_file.write("        ")
                if not (self.parser.parsedModel.eventVariable[i][j] in self.parser.parsedModel.speciesId):
                    self.out_file.write(self.parser.parsedModel.eventVariable[i][j])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.eventVariable[i][j])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parser.parsedModel.EventFormula[i][j]
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.eventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                self.out_file.write(string)
                self.out_file.write(";\n")
            self.out_file.write("}\n")
    
        self.out_file.write("\n")
    
        for i in range(0, len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("    ")
                if not (self.parser.parsedModel.ruleVariable[i] in self.parser.parsedModel.speciesId):
                    self.out_file.write("float ")
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.mathMLConditionParserCuda(self.parser.parsedModel.ruleFormula[i])
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.eventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
                self.out_file.write(string)
                self.out_file.write(";\n")
        self.out_file.write("")
    
        # Write the derivatives
        for i in range(0, num_species):
    
            if self.parser.parsedModel.species[i].getConstant() == False and self.parser.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    float d_y" + repr(i) + "= DT * (")
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reaction_written = False
                for k in range(0, self.parser.parsedModel.numReactions):
                    if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reaction_written and self.parser.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reaction_written = True
                        self.out_file.write(repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*(")
    
                        string = self.parser.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parser.parsedModel.speciesId)):
                            string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parser.parsedModel.parameterId)):
                            if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                        string = p.sub('', string)
    
                        self.out_file.write(string)
                        self.out_file.write(")")
    
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    my_species_compartment = self.parser.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parser.parsedModel.listOfParameter)):
                        if self.parser.parsedModel.listOfParameter[j].getId() == my_species_compartment:
                            if not (self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ");")
                                    break
                                else:
                                    self.out_file.write(self.parser.parsedModel.parameterId[j] + ");")
                                    break
                else:
                    self.out_file.write(");")
                self.out_file.write("\n")
    
        self.out_file.write("\n")
    
        # check for columns of the stochiometry matrix with more than one entry
        random_variables = ["*randNormal(rngRegs,sqrt(DT))"] * self.parser.parsedModel.numReactions
        for k in range(0, self.parser.parsedModel.numReactions):
            num_entries = 0
            for i in range(0, num_species):
                if self.parser.parsedModel.stoichiometricMatrix[i][k] != 0.0:
                    num_entries += 1
    
            # define specific randomVariable
            if num_entries > 1:
                self.out_file.write("    float rand" + repr(k) + " = randNormal(rngRegs,sqrt(DT));\n")
                random_variables[k] = "*rand" + repr(k)
    
        self.out_file.write("\n")
    
        # write noise terms
        for i in range(0, num_species):
            if self.parser.parsedModel.species[i].getConstant() == False and self.parser.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    d_y" + repr(i) + " += (")
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reaction_written = False
                for k in range(0, self.parser.parsedModel.numReactions):
                    if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reaction_written and self.parser.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reaction_written = True
                        self.out_file.write(repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*sqrt(")
    
                        string = self.parser.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parser.parsedModel.speciesId)):
                            # pq = re.compile(self.parser.parsedModel.speciesId[q])
                            # string=pq.sub('y['+repr(q)+']' ,string)
                            string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parser.parsedModel.parameterId)):
                            if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                        string = p.sub('', string)
                        self.out_file.write(string)
    
                        # multiply random variable
                        self.out_file.write(")")
                        self.out_file.write(random_variables[k])
    
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    my_species_compartment = self.parser.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parser.parsedModel.listOfParameter)):
                        if self.parser.parsedModel.listOfParameter[j].getId() == my_species_compartment:
                            if not (self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ")")
                                    break
                                else:
                                    self.out_file.write(self.parser.parsedModel.parameterId[j] + ")")
                                    break
                else:
                    self.out_file.write(")")
                self.out_file.write(";\n")
    
        self.out_file.write("\n")
        # add terms
        for i in range(0, num_species):
            if self.parser.parsedModel.species[i].getConstant() == False and self.parser.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    y[" + repr(i) + "] += d_y" + repr(i) + ";\n")
    
        self.out_file.write("}\n")
    
        ################# same file
    
        p = re.compile('\s')
        # The user-defined functions used in the model must be written in the file
        self.out_file.write("//Code for shared memory\n")
    
        num_events = len(self.parser.parsedModel.listOfEvents)
        num_rules = len(self.parser.parsedModel.listOfRules)
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
    
        for i in range(0, len(self.parser.parsedModel.listOfFunctions)):
            self.out_file.write("__device__ float " + self.parser.parsedModel.listOfFunctions[i].getId() + "(")
            for j in range(0, self.parser.parsedModel.listOfFunctions[i].getNumArguments()):
                self.out_file.write("float " + self.parser.parsedModel.functionArgument[i][j])
                if j < (self.parser.parsedModel.listOfFunctions[i].getNumArguments() - 1):
                    self.out_file.write(",")
            self.out_file.write("){\n    return ")
            self.out_file.write(self.parser.parsedModel.functionBody[i])
            self.out_file.write(";\n}\n")
            self.out_file.write("\n")
    
        self.out_file.write("\n")
        self.out_file.write("__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){\n")
    
        num_species = len(self.parser.parsedModel.species)
    
        # write rules and events
        for i in range(0, len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isRate():
                self.out_file.write("    ")
                if not (self.parser.parsedModel.ruleVariable[i] in self.parser.parsedModel.speciesId):
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parser.parsedModel.ruleFormula[i]
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.eventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parser.parsedModel.parameterId[q])
                            string = pq.sub('parameter[' + repr(q) + ']', string)
    
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        for i in range(0, len(self.parser.parsedModel.listOfEvents)):
            self.out_file.write("    if( ")
            self.out_file.write(self.mathMLConditionParserCuda(self.parser.parsedModel.EventCondition[i]))
            self.out_file.write("){\n")
            list_of_assignment_rules = self.parser.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(0, len(list_of_assignment_rules)):
                self.out_file.write("        ")
                if not (self.parser.parsedModel.eventVariable[i][j] in self.parser.parsedModel.speciesId):
                    self.out_file.write(self.parser.parsedModel.eventVariable[i][j])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.eventVariable[i][j])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parser.parsedModel.EventFormula[i][j]
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.eventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parser.parsedModel.parameterId[q])
                            string = pq.sub('parameter[' + repr(q) + ']', string)
    
                self.out_file.write(string)
                self.out_file.write(";\n")
            self.out_file.write("}\n")
    
        self.out_file.write("\n")
    
        for i in range(0, len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("    ")
                if not (self.parser.parsedModel.ruleVariable[i] in self.parser.parsedModel.speciesId):
                    self.out_file.write("float ")
                    self.out_file.write(self.parser.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.mathMLConditionParserCuda(self.parser.parsedModel.ruleFormula[i])
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.eventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parser.parsedModel.parameterId[q])
                            x = "parameter[" + repr(q) + "]"
                            string = pq.sub(x, string)
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        # Write the derivatives
        for i in range(0, num_species):
            if self.parser.parsedModel.species[i].getConstant() == False and self.parser.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    float d_y" + repr(i) + "= DT * (")
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reaction_written = False
                for k in range(0, self.parser.parsedModel.numReactions):
                    if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reaction_written and self.parser.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reaction_written = True
                        self.out_file.write(repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*(")
    
                        string = self.parser.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parser.parsedModel.speciesId)):
                            string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parser.parsedModel.parameterId)):
                            if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    pq = re.compile(self.parser.parsedModel.parameterId[q])
                                    string = pq.sub('parameter[' + repr(q) + ']', string)
    
                        string = p.sub('', string)
    
                        self.out_file.write(string)
                        self.out_file.write(")")
    
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    my_species_compartment = self.parser.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parser.parsedModel.listOfParameter)):
                        if self.parser.parsedModel.listOfParameter[j].getId() == my_species_compartment:
                            if not (self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("parameter[" + repr(j) + "]" + ");")
                                    break
                                else:
                                    self.out_file.write(self.parser.parsedModel.parameterId[j] + ");")
                                    break
                else:
                    self.out_file.write(");")
                self.out_file.write("\n")
    
        self.out_file.write("\n")
    
        # check for columns of the stochiometry matrix with more than one entry
        random_variables = ["*randNormal(rngRegs,sqrt(DT))"] * self.parser.parsedModel.numReactions
        for k in range(0, self.parser.parsedModel.numReactions):
            num_entries = 0
            for i in range(0, num_species):
                if self.parser.parsedModel.stoichiometricMatrix[i][k] != 0.0:
                    num_entries += 1
    
            # define specific randomVariable
            if num_entries > 1:
                self.out_file.write("    float rand" + repr(k) + " = randNormal(rngRegs,sqrt(DT));\n")
                random_variables[k] = "*rand" + repr(k)
    
        self.out_file.write("\n")
    
        # write noise terms
        for i in range(0, num_species):
            if self.parser.parsedModel.species[i].getConstant() == False and self.parser.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    d_y" + repr(i) + "+= (")
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reaction_written = False
                for k in range(0, self.parser.parsedModel.numReactions):
                    if not self.parser.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reaction_written and self.parser.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reaction_written = True
                        self.out_file.write(repr(self.parser.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*sqrt(")
    
                        string = self.parser.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parser.parsedModel.speciesId)):
                            # pq = re.compile(self.parser.parsedModel.speciesId[q])
                            # string=pq.sub('y['+repr(q)+']' ,string)
                            string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parser.parsedModel.parameterId)):
                            if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    pq = re.compile(self.parser.parsedModel.parameterId[q])
                                    string = pq.sub('parameter[' + repr(q) + ']', string)
    
                        string = p.sub('', string)
                        self.out_file.write(string)
    
                        # multiply random variable
                        self.out_file.write(")")
                        self.out_file.write(random_variables[k])
    
                if self.parser.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    my_species_compartment = self.parser.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parser.parsedModel.listOfParameter)):
                        if self.parser.parsedModel.listOfParameter[j].getId() == my_species_compartment:
                            if not (self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parser.parsedModel.eventVariable)):
                                    if self.parser.parsedModel.parameterId[j] in self.parser.parsedModel.eventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("parameter[" + repr(j) + "]" + ")")
                                    break
                                else:
                                    self.out_file.write(self.parser.parsedModel.parameterId[j] + ")")
                                    break
                else:
                    self.out_file.write(")")
    
                self.out_file.write(";\n")
    
        self.out_file.write("\n")
        # add terms
        for i in range(0, num_species):
            if self.parser.parsedModel.species[i].getConstant() == False and self.parser.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    y[" + repr(i) + "] += d_y" + repr(i) + ";\n")
    
        self.out_file.write("}\n")

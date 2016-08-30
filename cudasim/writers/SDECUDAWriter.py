import os
import re

from cudasim.writers.Writer import Writer
from cudasim.cuda_helpers import rename_math_functions

class SdeCUDAWriter(Writer):
    def __init__(self, parsedModel, outputPath=""):
        self.parsedModel = parsedModel
        self.out_file = open(os.path.join(outputPath, self.parsedModel.name + ".cu"), "w")
        self.rename()

    def rename(self):
        """
        This function renames parts of self.parsedModel to meet the specific requirements of this writer.
        This behaviour replaces the previous approach of subclassing the parser to produce different results depending
        on the which writer was intended to be used.
        """

        # Pad single-digit parameter names with a leading zero
        for i in range(self.comp-1, len(self.parsedModel.parameterId)):
            old_name = self.parsedModel.parameterId[i]
            num = old_name[len('parameter'):]
            if len(num) < 2:
                new_name = 'parameter' + '0' + str(num)
                self.parsedModel.parameterId[i] = new_name
                self.parsedModel.rename_everywhere(old_name, new_name)

        # Pad single-digit species names with a leading zero
        for i in range(0, len(self.parsedModel.speciesId)):
            old_name = self.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) < 2:
                new_name = 'species' + '0' + str(num)
                self.parsedModel.speciesId[i] = new_name
                self.parsedModel.rename_everywhere(old_name, new_name)

        rename_math_functions(self, 't')

    def write(self, method):
        """
        Write the cuda file with ODE functions using the information taken by the parser
        """
    
        numSpecies = len(self.parsedModel.species)
    
        p = re.compile('\s')
    
        # Write number of parameters and species
        self.out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
        self.out_file.write("#define NPARAM " + str(len(self.parsedModel.parameterId)) + "\n")
        self.out_file.write("#define NREACT " + str(self.parsedModel.numReactions) + "\n")
        self.out_file.write("\n")
    
        # The user-defined functions used in the model must be written in the file
        self.out_file.write("//Code for texture memory\n")
    
        numEvents = len(self.parsedModel.listOfEvents)
        numRules = len(self.parsedModel.listOfRules)
        num = numEvents + numRules
        if num > 0:
            self.out_file.write("#define leq(a,b) a<=b\n")
            self.out_file.write("#define neq(a,b) a!=b\n")
            self.out_file.write("#define geq(a,b) a>=b\n")
            self.out_file.write("#define lt(a,b) a<b\n")
            self.out_file.write("#define gt(a,b) a>b\n")
            self.out_file.write("#define eq(a,b) a==b\n")
            self.out_file.write("#define and_(a,b) a&&b\n")
            self.out_file.write("#define or_(a,b) a||b\n")
    
        for i in range(0, len(self.parsedModel.listOfFunctions)):
            self.out_file.write("__device__ float " + self.parsedModel.listOfFunctions[i].getId() + "(")
            for j in range(0, self.parsedModel.listOfFunctions[i].getNumArguments()):
                self.out_file.write("float " + self.parsedModel.FunctionArgument[i][j])
                if j < (self.parsedModel.listOfFunctions[i].getNumArguments() - 1):
                    self.out_file.write(",")
            self.out_file.write("){\n    return ")
            self.out_file.write(self.parsedModel.FunctionBody[i])
            self.out_file.write(";\n}\n")
            self.out_file.write("\n")
    
        self.out_file.write("\n")
    
        self.out_file.write("__device__ void step(float *y, float t, unsigned *rngRegs, int tid){\n")
    
        numSpecies = len(self.parsedModel.species)
    
        # write rules and events
        for i in range(0, len(self.parsedModel.listOfRules)):
            if self.parsedModel.listOfRules[i].isRate():
                self.out_file.write("    ")
                if not (self.parsedModel.ruleVariable[i] in self.parsedModel.speciesId):
                    self.out_file.write(self.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parsedModel.ruleFormula[i]
                for q in range(0, len(self.parsedModel.speciesId)):
                    string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        for i in range(0, len(self.parsedModel.listOfEvents)):
            self.out_file.write("    if( ")
            self.out_file.write(self.mathMLConditionParserCuda(self.parsedModel.EventCondition[i]))
            self.out_file.write("){\n")
            listOfAssignmentRules = self.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(0, len(listOfAssignmentRules)):
                self.out_file.write("        ")
                if not (self.parsedModel.EventVariable[i][j] in self.parsedModel.speciesId):
                    self.out_file.write(self.parsedModel.EventVariable[i][j])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.EventVariable[i][j])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parsedModel.EventFormula[i][j]
                for q in range(0, len(self.parsedModel.speciesId)):
                    string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                self.out_file.write(string)
                self.out_file.write(";\n")
            self.out_file.write("}\n")
    
        self.out_file.write("\n")
    
        for i in range(0, len(self.parsedModel.listOfRules)):
            if self.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("    ")
                if not (self.parsedModel.ruleVariable[i] in self.parsedModel.speciesId):
                    self.out_file.write("float ")
                    self.out_file.write(self.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.mathMLConditionParserCuda(self.parsedModel.ruleFormula[i])
                for q in range(0, len(self.parsedModel.speciesId)):
                    string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
                self.out_file.write(string)
                self.out_file.write(";\n")
        self.out_file.write("")
    
        # Write the derivatives
        for i in range(0, numSpecies):
    
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    float d_y" + repr(i) + "= DT * (")
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reactionWritten = False
                for k in range(0, self.parsedModel.numReactions):
                    if not self.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reactionWritten and self.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reactionWritten = True
                        self.out_file.write(repr(self.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*(")
    
                        string = self.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parsedModel.speciesId)):
                            string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parsedModel.parameterId)):
                            if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    string = self.rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                        string = p.sub('', string)
    
                        self.out_file.write(string)
                        self.out_file.write(")")
    
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    mySpeciesCompartment = self.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parsedModel.listOfParameter)):
                        if self.parsedModel.listOfParameter[j].getId() == mySpeciesCompartment:
                            if not (self.parsedModel.parameterId[j] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[j] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ");")
                                    break
                                else:
                                    self.out_file.write(self.parsedModel.parameterId[j] + ");")
                                    break
                else:
                    self.out_file.write(");")
                self.out_file.write("\n")
    
        self.out_file.write("\n")
    
        # check for columns of the stochiometry matrix with more than one entry
        randomVariables = ["*randNormal(rngRegs,sqrt(DT))"] * self.parsedModel.numReactions
        for k in range(0, self.parsedModel.numReactions):
            countEntries = 0
            for i in range(0, numSpecies):
                if self.parsedModel.stoichiometricMatrix[i][k] != 0.0:
                    countEntries += 1
    
            # define specific randomVariable
            if countEntries > 1:
                self.out_file.write("    float rand" + repr(k) + " = randNormal(rngRegs,sqrt(DT));\n")
                randomVariables[k] = "*rand" + repr(k)
    
        self.out_file.write("\n")
    
        # write noise terms
        for i in range(0, numSpecies):
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    d_y" + repr(i) + " += (")
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reactionWritten = False
                for k in range(0, self.parsedModel.numReactions):
                    if not self.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reactionWritten and self.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reactionWritten = True
                        self.out_file.write(repr(self.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*sqrt(")
    
                        string = self.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parsedModel.speciesId)):
                            # pq = re.compile(self.parsedModel.speciesId[q])
                            # string=pq.sub('y['+repr(q)+']' ,string)
                            string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parsedModel.parameterId)):
                            if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    string = self.rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                        string = p.sub('', string)
                        self.out_file.write(string)
    
                        # multiply random variable
                        self.out_file.write(")")
                        self.out_file.write(randomVariables[k])
    
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    mySpeciesCompartment = self.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parsedModel.listOfParameter)):
                        if self.parsedModel.listOfParameter[j].getId() == mySpeciesCompartment:
                            if not (self.parsedModel.parameterId[j] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[j] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ")")
                                    break
                                else:
                                    self.out_file.write(self.parsedModel.parameterId[j] + ")")
                                    break
                else:
                    self.out_file.write(")")
                self.out_file.write(";\n")
    
        self.out_file.write("\n")
        # add terms
        for i in range(0, numSpecies):
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    y[" + repr(i) + "] += d_y" + repr(i) + ";\n")
    
        self.out_file.write("}\n")
    
        ################# same file
    
        p = re.compile('\s')
        # The user-defined functions used in the model must be written in the file
        self.out_file.write("//Code for shared memory\n")
    
        numEvents = len(self.parsedModel.listOfEvents)
        numRules = len(self.parsedModel.listOfRules)
        num = numEvents + numRules
        if num > 0:
            self.out_file.write("#define leq(a,b) a<=b\n")
            self.out_file.write("#define neq(a,b) a!=b\n")
            self.out_file.write("#define geq(a,b) a>=b\n")
            self.out_file.write("#define lt(a,b) a<b\n")
            self.out_file.write("#define gt(a,b) a>b\n")
            self.out_file.write("#define eq(a,b) a==b\n")
            self.out_file.write("#define and_(a,b) a&&b\n")
            self.out_file.write("#define or_(a,b) a||b\n")
    
        for i in range(0, len(self.parsedModel.listOfFunctions)):
            self.out_file.write("__device__ float " + self.parsedModel.listOfFunctions[i].getId() + "(")
            for j in range(0, self.parsedModel.listOfFunctions[i].getNumArguments()):
                self.out_file.write("float " + self.parsedModel.FunctionArgument[i][j])
                if j < (self.parsedModel.listOfFunctions[i].getNumArguments() - 1):
                    self.out_file.write(",")
            self.out_file.write("){\n    return ")
            self.out_file.write(self.parsedModel.FunctionBody[i])
            self.out_file.write(";\n}\n")
            self.out_file.write("\n")
    
        self.out_file.write("\n")
        self.out_file.write("__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){\n")
    
        numSpecies = len(self.parsedModel.species)
    
        # write rules and events
        for i in range(0, len(self.parsedModel.listOfRules)):
            if self.parsedModel.listOfRules[i].isRate():
                self.out_file.write("    ")
                if not (self.parsedModel.ruleVariable[i] in self.parsedModel.speciesId):
                    self.out_file.write(self.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parsedModel.ruleFormula[i]
                for q in range(0, len(self.parsedModel.speciesId)):
                    string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parsedModel.parameterId[q])
                            string = pq.sub('parameter[' + repr(q) + ']', string)
    
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        for i in range(0, len(self.parsedModel.listOfEvents)):
            self.out_file.write("    if( ")
            self.out_file.write(self.mathMLConditionParserCuda(self.parsedModel.EventCondition[i]))
            self.out_file.write("){\n")
            listOfAssignmentRules = self.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(0, len(listOfAssignmentRules)):
                self.out_file.write("        ")
                if not (self.parsedModel.EventVariable[i][j] in self.parsedModel.speciesId):
                    self.out_file.write(self.parsedModel.EventVariable[i][j])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.EventVariable[i][j])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parsedModel.EventFormula[i][j]
                for q in range(0, len(self.parsedModel.speciesId)):
                    string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parsedModel.parameterId[q])
                            string = pq.sub('parameter[' + repr(q) + ']', string)
    
                self.out_file.write(string)
                self.out_file.write(";\n")
            self.out_file.write("}\n")
    
        self.out_file.write("\n")
    
        for i in range(0, len(self.parsedModel.listOfRules)):
            if self.parsedModel.listOfRules[i].isAssignment():
                self.out_file.write("    ")
                if not (self.parsedModel.ruleVariable[i] in self.parsedModel.speciesId):
                    self.out_file.write("float ")
                    self.out_file.write(self.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.mathMLConditionParserCuda(self.parsedModel.ruleFormula[i])
                for q in range(0, len(self.parsedModel.speciesId)):
                    string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parsedModel.parameterId[q])
                            x = "parameter[" + repr(q) + "]"
                            string = pq.sub(x, string)
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        # Write the derivatives
        for i in range(0, numSpecies):
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    float d_y" + repr(i) + "= DT * (")
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reactionWritten = False
                for k in range(0, self.parsedModel.numReactions):
                    if not self.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reactionWritten and self.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reactionWritten = True
                        self.out_file.write(repr(self.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*(")
    
                        string = self.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parsedModel.speciesId)):
                            string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parsedModel.parameterId)):
                            if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    pq = re.compile(self.parsedModel.parameterId[q])
                                    string = pq.sub('parameter[' + repr(q) + ']', string)
    
                        string = p.sub('', string)
    
                        self.out_file.write(string)
                        self.out_file.write(")")
    
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    mySpeciesCompartment = self.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parsedModel.listOfParameter)):
                        if self.parsedModel.listOfParameter[j].getId() == mySpeciesCompartment:
                            if not (self.parsedModel.parameterId[j] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[j] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("parameter[" + repr(j) + "]" + ");")
                                    break
                                else:
                                    self.out_file.write(self.parsedModel.parameterId[j] + ");")
                                    break
                else:
                    self.out_file.write(");")
                self.out_file.write("\n")
    
        self.out_file.write("\n")
    
        # check for columns of the stochiometry matrix with more than one entry
        randomVariables = ["*randNormal(rngRegs,sqrt(DT))"] * self.parsedModel.numReactions
        for k in range(0, self.parsedModel.numReactions):
            countEntries = 0
            for i in range(0, numSpecies):
                if self.parsedModel.stoichiometricMatrix[i][k] != 0.0:
                    countEntries += 1
    
            # define specific randomVariable
            if countEntries > 1:
                self.out_file.write("    float rand" + repr(k) + " = randNormal(rngRegs,sqrt(DT));\n")
                randomVariables[k] = "*rand" + repr(k)
    
        self.out_file.write("\n")
    
        # write noise terms
        for i in range(0, numSpecies):
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    d_y" + repr(i) + "+= (")
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write("(")
    
                reactionWritten = False
                for k in range(0, self.parsedModel.numReactions):
                    if not self.parsedModel.stoichiometricMatrix[i][k] == 0.0:
    
                        if reactionWritten and self.parsedModel.stoichiometricMatrix[i][k] > 0.0:
                            self.out_file.write("+")
                        reactionWritten = True
                        self.out_file.write(repr(self.parsedModel.stoichiometricMatrix[i][k]))
                        self.out_file.write("*sqrt(")
    
                        string = self.parsedModel.kineticLaw[k]
                        for q in range(0, len(self.parsedModel.speciesId)):
                            # pq = re.compile(self.parsedModel.speciesId[q])
                            # string=pq.sub('y['+repr(q)+']' ,string)
                            string = self.rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                        for q in range(0, len(self.parsedModel.parameterId)):
                            if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    pq = re.compile(self.parsedModel.parameterId[q])
                                    string = pq.sub('parameter[' + repr(q) + ']', string)
    
                        string = p.sub('', string)
                        self.out_file.write(string)
    
                        # multiply random variable
                        self.out_file.write(")")
                        self.out_file.write(randomVariables[k])
    
                if self.parsedModel.species[i].isSetCompartment():
                    self.out_file.write(")/")
                    mySpeciesCompartment = self.parsedModel.species[i].getCompartment()
                    for j in range(0, len(self.parsedModel.listOfParameter)):
                        if self.parsedModel.listOfParameter[j].getId() == mySpeciesCompartment:
                            if not (self.parsedModel.parameterId[j] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[j] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    self.out_file.write("parameter[" + repr(j) + "]" + ")")
                                    break
                                else:
                                    self.out_file.write(self.parsedModel.parameterId[j] + ")")
                                    break
                else:
                    self.out_file.write(")")
    
                self.out_file.write(";\n")
    
        self.out_file.write("\n")
        # add terms
        for i in range(0, numSpecies):
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("    y[" + repr(i) + "] += d_y" + repr(i) + ";\n")
    
        self.out_file.write("}\n")
    

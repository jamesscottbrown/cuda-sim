import os
import re

from numpy import *

from cudasim.writers.Writer import Writer
from cudasim.cuda_helpers import rename_math_functions


class GillespieCUDAWriter(Writer):
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

    def write(self, useMoleculeCounts=False):

        p = re.compile('\s')
        # Open the outfile
    
        numSpecies = len(self.parser.parsedModel.species)
    
        numEvents = len(self.parser.parsedModel.listOfEvents)
        numRules = len(self.parser.parsedModel.listOfRules)
        num = numEvents + numRules
    
        # Write number of parameters and species
        self.out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
        self.out_file.write("#define NPARAM " + str(len(self.parser.parsedModel.parameterId)) + "\n")
        self.out_file.write("#define NREACT " + str(self.parser.parsedModel.numReactions) + "\n")
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
    
        for i in range(0, len(self.parser.parsedModel.listOfFunctions)):
            self.out_file.write("__device__ float " + self.parser.parsedModel.listOfFunctions[i].getId() + "(")
            for j in range(0, self.parser.parsedModel.listOfFunctions[i].getNumArguments()):
                self.out_file.write("float " + self.parser.parsedModel.FunctionArgument[i][j])
                if j < (self.parser.parsedModel.listOfFunctions[i].getNumArguments() - 1):
                    self.out_file.write(",")
            self.out_file.write("){\n    return ")
            self.out_file.write(self.parser.parsedModel.FunctionBody[i])
            self.out_file.write(";\n}\n")
            self.out_file.write("")
    
        self.out_file.write("\n\n__constant__ int smatrix[]={\n")
        for i in range(0, len(self.parser.parsedModel.stoichiometricMatrix[0])):
            for j in range(0, len(self.parser.parsedModel.stoichiometricMatrix)):
                self.out_file.write("    " + repr(self.parser.parsedModel.stoichiometricMatrix[j][i]))
                if not (i == (len(self.parser.parsedModel.stoichiometricMatrix) - 1) and (j == (len(self.parser.parsedModel.stoichiometricMatrix[0]) - 1))):
                    self.out_file.write(",")
            self.out_file.write("\n")
    
        self.out_file.write("};\n\n\n")
    
        if useMoleculeCounts:
                self.out_file.write("__device__ void hazards(int *y, float *h, float t, int tid){\n")
                self.out_file.write("        // Assume rate law expressed in terms of molecule counts \n")
        else:
            self.out_file.write("__device__ void hazards(int *yCounts, float *h, float t, int tid){")
    
            self.out_file.write("""
            // Calculate concentrations from molecule counts
            int y[NSPECIES];
            """)
    
            for i in range(0, numSpecies):
                volumeString = "tex2D(param_tex," + repr(self.parser.parsedModel.speciesCompartmentList[i]) + ",tid)"
                self.out_file.write("y[%s] = yCounts[%s] / (6.022E23 * %s);\n" % (i, i, volumeString))
    
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
                        for r in range(0, len(self.parser.parsedModel.EventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                self.out_file.write(string)
                self.out_file.write(";\n")
    
        for i in range(0, len(self.parser.parsedModel.listOfEvents)):
            self.out_file.write("    if( ")
            self.out_file.write(self.mathMLConditionParserCuda(self.parser.parsedModel.EventCondition[i]))
            self.out_file.write("){\n")
            listOfAssignmentRules = self.parser.parsedModel.listOfEvents[i].getListOfEventAssignments()
            for j in range(0, len(listOfAssignmentRules)):
                self.out_file.write("        ")
                if not (self.parser.parsedModel.EventVariable[i][j] in self.parser.parsedModel.speciesId):
                    self.out_file.write(self.parser.parsedModel.EventVariable[i][j])
                else:
                    string = "y[" + repr(self.parser.parsedModel.speciesId.index(self.parser.parsedModel.EventVariable[i][j])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")
    
                string = self.parser.parsedModel.EventFormula[i][j]
                for q in range(0, len(self.parser.parsedModel.speciesId)):
                    string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parser.parsedModel.parameterId)):
                    if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parser.parsedModel.EventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
                self.out_file.write(string)
                self.out_file.write(";\n")
            self.out_file.write("    }\n")
    
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
                        for r in range(0, len(self.parser.parsedModel.EventVariable)):
                            if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
                self.out_file.write(string)
                self.out_file.write(";\n")
        self.out_file.write("\n")
    
        for i in range(0, self.parser.parsedModel.numReactions):
    
            if useMoleculeCounts:
                self.out_file.write("    h[" + repr(i) + "] = ")
            else:
                self.out_file.write("    h[" + repr(i) + "] = 6.022E23 * ")
    
            string = self.parser.parsedModel.kineticLaw[i]
            for q in range(0, len(self.parser.parsedModel.speciesId)):
                string = self.rep(string, self.parser.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(self.parser.parsedModel.parameterId)):
                if not (self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.ruleVariable):
                    flag = False
                    for r in range(0, len(self.parser.parsedModel.EventVariable)):
                        if self.parser.parsedModel.parameterId[q] in self.parser.parsedModel.EventVariable[r]:
                            flag = True
                    if not flag:
                        string = self.rep(string, self.parser.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
    
            string = p.sub('', string)
            self.out_file.write(string + ";\n")
    
        self.out_file.write("\n")
        self.out_file.write("}\n\n")

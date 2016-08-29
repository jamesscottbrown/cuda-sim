from libsbml import *
from abcsysbio.relations import *
import os
import re
from Writer import Writer


# replace the species and parameters recursively
def rep(string, find, replace):
    ex = find + "[^0-9]"
    while re.search(ex, string) is not None:
        res = re.search(ex, string)
        string = string[0:res.start()] + replace + " " + string[res.end() - 1:]

    ex = find + "$"
    if re.search(ex, string) is not None:
        res = re.search(ex, string)
        string = string[0:res.start()] + replace + " " + string[res.end():]

    return string


class OdeCUDAWriter(Writer):
    def __init__(self, sbmlFileName, modelName="", inputPath="", outputPath=""):
        Writer.__init__(self, sbmlFileName, modelName, inputPath, outputPath)
        self.out_file = open(os.path.join(outputPath, self.parsedModel.name + ".cu"), "w")

    def mathMLConditionParserCuda(self, mathMLstring):
        """
        Replaces and and or with and_ and or_ in a MathML string.
        Returns the string with and and or replaced by and_ and or_

        ***** args *****

        mathMLstring:

                A mathMLstring

        """

        andString = re.compile("and")
        orString = re.compile("or")
        mathMLstring = andString.sub("and_", mathMLstring)
        mathMLstring = orString.sub("or_", mathMLstring)

        return mathMLstring

    def write(self):
        """
        Write the cuda file with DDE functions using the information taken by the parser
        """

        numSpecies = len(self.parsedModel.species)

        p = re.compile('\s')

        # Write number of parameters and species
        self.out_file.write("\n")
        self.out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
        self.out_file.write("#define NPARAM " + str(len(self.parsedModel.parameterId)) + "\n")
        self.out_file.write("#define NREACT " + str(self.parsedModel.numReactions) + "\n")
        self.out_file.write("#define NCOMPARTMENTS " + str(self.parsedModel.numCompartments) + "\n")
        self.out_file.write("\n")

        # The user-defined functions used in the model must be written in the file
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

        self.out_file.write("__device__  float f(float t, float *y, float yOld[NSPECIES][numDelays], int dimension){\n")
        self.out_file.write("\tint tid = blockDim.x * blockIdx.x + threadIdx.x;\n")
        self.out_file.write("\tconst float tau[numDelays] = {" + ", ".join(delays) + "};\n")
        self.out_file.write("\tfloat ydot[NSPECIES];\n")

        numSpecies = len(self.parsedModel.species)

        # write rules and events
        for i in range(0, len(self.parsedModel.listOfRules)):
            if self.parsedModel.listOfRules[i].isRate():
                self.out_file.write("        ")
                if not (self.parsedModel.ruleVariable[i] in self.parsedModel.speciesId):
                    self.out_file.write(self.parsedModel.ruleVariable[i])
                else:
                    string = "y[" + repr(self.parsedModel.speciesId.index(self.parsedModel.ruleVariable[i])) + "]"
                    self.out_file.write(string)
                self.out_file.write("=")

                string = self.parsedModel.ruleFormula[i]
                for q in range(0, len(self.parsedModel.speciesId)):
                    pq = re.compile(self.parsedModel.speciesId[q])
                    string = pq.sub('y[' + repr(q) + ']', string)
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            pq = re.compile(self.parsedModel.parameterId[q])
                            string = pq.sub('tex2D(param_tex,' + repr(q) + ',tid)', string)

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
                    string = rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

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
                    string = rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')
                for q in range(0, len(self.parsedModel.parameterId)):
                    if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                        flag = False
                        for r in range(0, len(self.parsedModel.EventVariable)):
                            if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                flag = True
                        if not flag:
                            string = rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
                self.out_file.write(string)
                self.out_file.write(";\n")
        self.out_file.write("\n\n")

        # Write the derivatives
        for i in range(0, numSpecies):
            if self.parsedModel.species[i].getConstant() == False and self.parsedModel.species[i].getBoundaryCondition() == False:
                self.out_file.write("        ydot[" + repr(i) + "]=")
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
                            string = rep(string, self.parsedModel.speciesId[q], 'y[' + repr(q) + ']')

                        for q in range(0, len(self.parsedModel.parameterId)):
                            if not (self.parsedModel.parameterId[q] in self.parsedModel.ruleVariable):
                                flag = False
                                for r in range(0, len(self.parsedModel.EventVariable)):
                                    if self.parsedModel.parameterId[q] in self.parsedModel.EventVariable[r]:
                                        flag = True
                                if not flag:
                                    string = rep(string, self.parsedModel.parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

                        string = p.sub('', string)

                        # substitute to fix delays [replace delay(y[1],...) with delay(1,...)
                        getDimension = re.compile("delay\(y\[(\d+?)\]")
                        string = getDimension.sub(r'delay(\g<1>', string)

                        # subsitute to convert from param value to delay number
                        getParamNum = re.compile("delay\((\w+?),tex2D\(param_tex,(\d+?),tid\)\)")
                        string = getParamNum.sub(r'delay(\g<1>,\g<2>)', string)

                        # print string
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
                                    self.out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ";")
                                    break
                                else:
                                    self.out_file.write(self.parsedModel.parameterId[j] + ";")
                                    break

                else:
                    self.out_file.write(";")
                self.out_file.write("\n")

        self.out_file.write("return ydot[dimension];")
        self.out_file.write("\n    }")

import os
import re

from libsbml import *
from numpy import *


# To call the parser:
#    SBMLparse.importSBML(source, integrationType, ModelName=None,method=None)
#    All arguments to function must be passed as tuples.
#    If there is only one source to parse it must still be passed as a tuple ('source.xml',)
#    with an integrationType passed as ('Gillespie',)


def rep(string, find, replace):
    # Replace the species and parameters recursively.
    #
    # This allowed the replacement of:
    # pq = re.compile(speciesId[q])
    # string=pq.sub('y['+repr(q)+']' ,string)
    # by:
    # string = rep(string, speciesId[q],'y['+repr(q)+']')

    ex = find + "[^0-9]"
    ss = string
    while re.search(ex, ss) is not None:
        res = re.search(ex, ss)
        ss = ss[0:res.start()] + replace + " " + ss[res.end() - 1:]

    ex = find + "$"
    if re.search(ex, ss) is not None:
        res = re.search(ex, ss)
        ss = ss[0:res.start()] + replace + " " + ss[res.end():]

    return ss


######################## CUDA SDE #################################

def write_SDECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId, listOfParameter, parameterId,
                  name, listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable,
                  listOfEvents, EventCondition, EventVariable, EventFormula, outpath=""):
    """
    Write the cuda file with ODE functions using the information taken by the parser
    """

    numSpecies = len(species)

    p = re.compile('\s')
    # Open the outfile
    out_file = open(os.path.join(outpath, name + ".cu"), "w")

    # Write number of parameters and species
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(len(parameterId)) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("\n")

    # The user-defined functions used in the model must be written in the file
    out_file.write("//Code for texture memory\n")

    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents + numRules
    if num > 0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0, len(listOfFunctions)):
        out_file.write("__device__ float " + listOfFunctions[i].getId() + "(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float " + FunctionArgument[i][j])
            if j < (listOfFunctions[i].getNumArguments() - 1):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")

    out_file.write("\n")

    out_file.write("__device__ void step(float *y, float t, unsigned *rngRegs, int tid){\n")

    numSpecies = len(species)

    # write rules and events
    for i in range(0, len(listOfRules)):
        if listOfRules[i].isRate():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

            out_file.write(string)
            out_file.write(";\n")

    for i in range(0, len(listOfEvents)):
        out_file.write("    if( ")
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            if not (EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y[" + repr(speciesId.index(EventVariable[i][j])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = EventFormula[i][j]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("")

    # Write the derivatives
    for i in range(0, numSpecies):

        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("    float d_y" + repr(i) + "= DT * (")
            if species[i].isSetCompartment():
                out_file.write("(")

            reactionWritten = False
            for k in range(0, numReactions):
                if not stoichiometricMatrix[i][k] == 0.0:

                    if reactionWritten and stoichiometricMatrix[i][k] > 0.0:
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")

                    string = kineticLaw[k]
                    for q in range(0, len(speciesId)):
                        string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
                    for q in range(0, len(parameterId)):
                        if not (parameterId[q] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[q] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

                    string = p.sub('', string)

                    out_file.write(string)
                    out_file.write(")")

            if species[i].isSetCompartment():
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if listOfParameter[j].getId() == mySpeciesCompartment:
                        if not (parameterId[j] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[j] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ");")
                                break
                            else:
                                out_file.write(parameterId[j] + ");")
                                break
            else:
                out_file.write(");")
            out_file.write("\n")

    out_file.write("\n")

    # check for columns of the stochiometry matrix with more than one entry
    randomVariables = ["*randNormal(rngRegs,sqrt(DT))"] * numReactions
    for k in range(0, numReactions):
        countEntries = 0
        for i in range(0, numSpecies):
            if stoichiometricMatrix[i][k] != 0.0:
                countEntries += 1

        # define specific randomVariable
        if countEntries > 1:
            out_file.write("    float rand" + repr(k) + " = randNormal(rngRegs,sqrt(DT));\n")
            randomVariables[k] = "*rand" + repr(k)

    out_file.write("\n")

    # write noise terms
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("    d_y" + repr(i) + " += (")
            if species[i].isSetCompartment():
                out_file.write("(")

            reactionWritten = False
            for k in range(0, numReactions):
                if not stoichiometricMatrix[i][k] == 0.0:

                    if reactionWritten and stoichiometricMatrix[i][k] > 0.0:
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*sqrt(")

                    string = kineticLaw[k]
                    for q in range(0, len(speciesId)):
                        # pq = re.compile(speciesId[q])
                        # string=pq.sub('y['+repr(q)+']' ,string)
                        string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
                    for q in range(0, len(parameterId)):
                        if not (parameterId[q] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[q] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

                    string = p.sub('', string)
                    out_file.write(string)

                    # multiply random variable
                    out_file.write(")")
                    out_file.write(randomVariables[k])

            if species[i].isSetCompartment():
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if listOfParameter[j].getId() == mySpeciesCompartment:
                        if not (parameterId[j] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[j] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ")")
                                break
                            else:
                                out_file.write(parameterId[j] + ")")
                                break
            else:
                out_file.write(")")
            out_file.write(";\n")

    out_file.write("\n")
    # add terms
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("    y[" + repr(i) + "] += d_y" + repr(i) + ";\n")

    out_file.write("}\n")

    ################# same file

    p = re.compile('\s')
    # The user-defined functions used in the model must be written in the file
    out_file.write("//Code for shared memory\n")

    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents + numRules
    if num > 0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0, len(listOfFunctions)):
        out_file.write("__device__ float " + listOfFunctions[i].getId() + "(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float " + FunctionArgument[i][j])
            if j < (listOfFunctions[i].getNumArguments() - 1):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")

    out_file.write("\n")
    out_file.write("__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){\n")

    numSpecies = len(species)

    # write rules and events
    for i in range(0, len(listOfRules)):
        if listOfRules[i].isRate():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        pq = re.compile(parameterId[q])
                        string = pq.sub('parameter[' + repr(q) + ']', string)

            out_file.write(string)
            out_file.write(";\n")

    for i in range(0, len(listOfEvents)):
        out_file.write("    if( ")
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            if not (EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y[" + repr(speciesId.index(EventVariable[i][j])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = EventFormula[i][j]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        pq = re.compile(parameterId[q])
                        string = pq.sub('parameter[' + repr(q) + ']', string)

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        pq = re.compile(parameterId[q])
                        x = "parameter[" + repr(q) + "]"
                        string = pq.sub(x, string)
            out_file.write(string)
            out_file.write(";\n")

    # Write the derivatives
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("    float d_y" + repr(i) + "= DT * (")
            if species[i].isSetCompartment():
                out_file.write("(")

            reactionWritten = False
            for k in range(0, numReactions):
                if not stoichiometricMatrix[i][k] == 0.0:

                    if reactionWritten and stoichiometricMatrix[i][k] > 0.0:
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")

                    string = kineticLaw[k]
                    for q in range(0, len(speciesId)):
                        string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
                    for q in range(0, len(parameterId)):
                        if not (parameterId[q] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[q] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                pq = re.compile(parameterId[q])
                                string = pq.sub('parameter[' + repr(q) + ']', string)

                    string = p.sub('', string)

                    out_file.write(string)
                    out_file.write(")")

            if species[i].isSetCompartment():
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if listOfParameter[j].getId() == mySpeciesCompartment:
                        if not (parameterId[j] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[j] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                out_file.write("parameter[" + repr(j) + "]" + ");")
                                break
                            else:
                                out_file.write(parameterId[j] + ");")
                                break
            else:
                out_file.write(");")
            out_file.write("\n")

    out_file.write("\n")

    # check for columns of the stochiometry matrix with more than one entry
    randomVariables = ["*randNormal(rngRegs,sqrt(DT))"] * numReactions
    for k in range(0, numReactions):
        countEntries = 0
        for i in range(0, numSpecies):
            if stoichiometricMatrix[i][k] != 0.0:
                countEntries += 1

        # define specific randomVariable
        if countEntries > 1:
            out_file.write("    float rand" + repr(k) + " = randNormal(rngRegs,sqrt(DT));\n")
            randomVariables[k] = "*rand" + repr(k)

    out_file.write("\n")

    # write noise terms
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("    d_y" + repr(i) + "+= (")
            if species[i].isSetCompartment():
                out_file.write("(")

            reactionWritten = False
            for k in range(0, numReactions):
                if not stoichiometricMatrix[i][k] == 0.0:

                    if reactionWritten and stoichiometricMatrix[i][k] > 0.0:
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*sqrt(")

                    string = kineticLaw[k]
                    for q in range(0, len(speciesId)):
                        # pq = re.compile(speciesId[q])
                        # string=pq.sub('y['+repr(q)+']' ,string)
                        string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
                    for q in range(0, len(parameterId)):
                        if not (parameterId[q] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[q] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                pq = re.compile(parameterId[q])
                                string = pq.sub('parameter[' + repr(q) + ']', string)

                    string = p.sub('', string)
                    out_file.write(string)

                    # multiply random variable
                    out_file.write(")")
                    out_file.write(randomVariables[k])

            if species[i].isSetCompartment():
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if listOfParameter[j].getId() == mySpeciesCompartment:
                        if not (parameterId[j] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[j] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                out_file.write("parameter[" + repr(j) + "]" + ")")
                                break
                            else:
                                out_file.write(parameterId[j] + ")")
                                break
            else:
                out_file.write(")")

            out_file.write(";\n")

    out_file.write("\n")
    # add terms
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("    y[" + repr(i) + "] += d_y" + repr(i) + ";\n")

    out_file.write("}\n")


######################## CUDA Gillespie #################################

def write_GillespieCUDA(stoichiometricMatrix, kineticLaw, numSpecies, numReactions, parameterId, speciesId, name,
                        listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable,
                        listOfEvents, EventCondition, EventVariable, EventFormula, outpath=""):
    p = re.compile('\s')
    # Open the outfile
    out_file = open(os.path.join(outpath, name + ".cu"), "w")

    # Write number of parameters and species
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(len(parameterId)) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("\n")

    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents + numRules
    if num > 0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0, len(listOfFunctions)):
        out_file.write("__device__ float " + listOfFunctions[i].getId() + "(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float " + FunctionArgument[i][j])
            if j < (listOfFunctions[i].getNumArguments() - 1):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("")

    out_file.write("\n\n__constant__ int smatrix[]={\n")
    for i in range(0, len(stoichiometricMatrix[0])):
        for j in range(0, len(stoichiometricMatrix)):
            out_file.write("    " + repr(stoichiometricMatrix[j][i]))
            if not (i == (len(stoichiometricMatrix) - 1) and (j == (len(stoichiometricMatrix[0]) - 1))):
                out_file.write(",")
        out_file.write("\n")

    out_file.write("};\n\n\n")

    out_file.write("__device__ void hazards(int *y, float *h, float t, int tid){")
    # wirte rules and events 
    for i in range(0, len(listOfRules)):
        if listOfRules[i].isRate():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

            out_file.write(string)
            out_file.write(";\n")

    for i in range(0, len(listOfEvents)):
        out_file.write("    if( ")
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            if not (EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y[" + repr(speciesId.index(EventVariable[i][j])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = EventFormula[i][j]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("    }\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("\n")

    for i in range(0, numReactions):
        out_file.write("    h[" + repr(i) + "] = ")

        string = kineticLaw[i]
        for q in range(0, len(speciesId)):
            string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
        for q in range(0, len(parameterId)):
            if not (parameterId[q] in ruleVariable):
                flag = False
                for r in range(0, len(EventVariable)):
                    if parameterId[q] in EventVariable[r]:
                        flag = True
                if not flag:
                    string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

        string = p.sub('', string)
        out_file.write(string + ";\n")

    out_file.write("\n")
    out_file.write("}\n\n")


######################## CUDA ODE #################################

def write_ODECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId, listOfParameter, parameterId,
                  name, listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable,
                  listOfEvents, EventCondition, EventVariable, EventFormula, outpath=""):
    """
    Write the cuda file with ODE functions using the information taken by the parser
    """

    numSpecies = len(species)

    p = re.compile('\s')
    # Open the outfile
    out_file = open(os.path.join(outpath, name + ".cu"), "w")

    # Write number of parameters and species
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(len(parameterId)) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("\n")

    # The user-defined functions used in the model must be written in the file
    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents + numRules
    if num > 0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0, len(listOfFunctions)):
        out_file.write("__device__ double " + listOfFunctions[i].getId() + "(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("double " + FunctionArgument[i][j])
            if j < (listOfFunctions[i].getNumArguments() - 1):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")

    out_file.write(
        "struct myFex{\n    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)\n    {\n        int tid = blockDim.x * blockIdx.x + threadIdx.x;\n")


    # write rules and events
    for i in range(0, len(listOfRules)):
        if listOfRules[i].isRate():
            out_file.write("        ")
            if not (ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0, len(speciesId)):
                pq = re.compile(speciesId[q])
                string = pq.sub('y[' + repr(q) + ']', string)
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        pq = re.compile(parameterId[q])
                        string = pq.sub('tex2D(param_tex,' + repr(q) + ',tid)', string)

            out_file.write(string)
            out_file.write(";\n")

    for i in range(0, len(listOfEvents)):
        out_file.write("    if( ")
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            if not (EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y[" + repr(speciesId.index(EventVariable[i][j])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = EventFormula[i][j]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write("double ")
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("\n\n")

    # Write the derivatives
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("        ydot[" + repr(i) + "]=")
            if species[i].isSetCompartment():
                out_file.write("(")

            reactionWritten = False
            for k in range(0, numReactions):
                if not stoichiometricMatrix[i][k] == 0.0:

                    if reactionWritten and stoichiometricMatrix[i][k] > 0.0:
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")

                    string = kineticLaw[k]
                    for q in range(0, len(speciesId)):
                        string = rep(string, speciesId[q], 'y[' + repr(q) + ']')

                    for q in range(0, len(parameterId)):
                        if not (parameterId[q] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[q] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

                    string = p.sub('', string)

                    # print string
                    out_file.write(string)
                    out_file.write(")")

            if species[i].isSetCompartment():
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if listOfParameter[j].getId() == mySpeciesCompartment:
                        if not (parameterId[j] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[j] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ";")
                                break
                            else:
                                out_file.write(parameterId[j] + ";")
                                break

            else:
                out_file.write(";")
            out_file.write("\n")

    out_file.write("\n    }")
    out_file.write(
        "\n};\n\n\n struct myJex{\n    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){\n        return; \n    }\n};")


######################## CUDA DDE #################################

def write_DDECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId, listOfParameter, parameterId,
                  name, listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable,
                  listOfEvents, EventCondition, EventVariable, EventFormula, delays, numCompartments, outpath=""):
    """
    Write the cuda file with DDE functions using the information taken by the parser
    """

    numSpecies = len(species)

    p = re.compile('\s')
    # Open the outfile
    out_file = open(os.path.join(outpath, name + ".cu"), "w")

    # Write number of parameters and species
    out_file.write("\n")
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(len(parameterId)) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("#define NCOMPARTMENTS " + str(numCompartments) + "\n")
    out_file.write("\n")

    # The user-defined functions used in the model must be written in the file
    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents + numRules
    if num > 0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0, len(listOfFunctions)):
        out_file.write("__device__ float " + listOfFunctions[i].getId() + "(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float " + FunctionArgument[i][j])
            if j < (listOfFunctions[i].getNumArguments() - 1):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")

    out_file.write("__device__  float f(float t, float *y, float yOld[NSPECIES][numDelays], int dimension){\n")
    out_file.write("\tint tid = blockDim.x * blockIdx.x + threadIdx.x;\n")
    out_file.write("\tconst float tau[numDelays] = {" + ", ".join(delays) + "};\n")
    out_file.write("\tfloat ydot[NSPECIES];\n")

    numSpecies = len(species)

    # write rules and events
    for i in range(0, len(listOfRules)):
        if listOfRules[i].isRate():
            out_file.write("        ")
            if not (ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0, len(speciesId)):
                pq = re.compile(speciesId[q])
                string = pq.sub('y[' + repr(q) + ']', string)
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        pq = re.compile(parameterId[q])
                        string = pq.sub('tex2D(param_tex,' + repr(q) + ',tid)', string)

            out_file.write(string)
            out_file.write(";\n")

    for i in range(0, len(listOfEvents)):
        out_file.write("    if( ")
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            if not (EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y[" + repr(speciesId.index(EventVariable[i][j])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = EventFormula[i][j]
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not (ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y[" + repr(speciesId.index(ruleVariable[i])) + "]"
                out_file.write(string)
            out_file.write("=")

            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0, len(speciesId)):
                string = rep(string, speciesId[q], 'y[' + repr(q) + ']')
            for q in range(0, len(parameterId)):
                if not (parameterId[q] in ruleVariable):
                    flag = False
                    for r in range(0, len(EventVariable)):
                        if parameterId[q] in EventVariable[r]:
                            flag = True
                    if not flag:
                        string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("\n\n")

    # Write the derivatives
    for i in range(0, numSpecies):
        if species[i].getConstant() == False and species[i].getBoundaryCondition() == False:
            out_file.write("        ydot[" + repr(i) + "]=")
            if species[i].isSetCompartment():
                out_file.write("(")

            reactionWritten = False
            for k in range(0, numReactions):
                if not stoichiometricMatrix[i][k] == 0.0:

                    if reactionWritten and stoichiometricMatrix[i][k] > 0.0:
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")

                    string = kineticLaw[k]
                    for q in range(0, len(speciesId)):
                        string = rep(string, speciesId[q], 'y[' + repr(q) + ']')

                    for q in range(0, len(parameterId)):
                        if not (parameterId[q] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[q] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                string = rep(string, parameterId[q], 'tex2D(param_tex,' + repr(q) + ',tid)')

                    string = p.sub('', string)

                    # substitute to fix delays [replace delay(y[1],...) with delay(1,...)
                    getDimension = re.compile("delay\(y\[(\d+?)\]")
                    string = getDimension.sub(r'delay(\g<1>', string)

                    # subsitute to convert from param value to delay number
                    getParamNum = re.compile("delay\((\w+?),tex2D\(param_tex,(\d+?),tid\)\)")
                    string = getParamNum.sub(r'delay(\g<1>,\g<2>)', string)

                    # print string
                    out_file.write(string)
                    out_file.write(")")

            if species[i].isSetCompartment():
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if listOfParameter[j].getId() == mySpeciesCompartment:
                        if not (parameterId[j] in ruleVariable):
                            flag = False
                            for r in range(0, len(EventVariable)):
                                if parameterId[j] in EventVariable[r]:
                                    flag = True
                            if not flag:
                                out_file.write("tex2D(param_tex," + repr(j) + ",tid)" + ";")
                                break
                            else:
                                out_file.write(parameterId[j] + ";")
                                break

            else:
                out_file.write(";")
            out_file.write("\n")

    out_file.write("return ydot[dimension];")
    out_file.write("\n    }")


################################################################################
# The parser for logical operations in conditions                              #
################################################################################

def mathMLConditionParserCuda(mathMLstring):
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


################################################################################
# Function to get initial amount given a species and an algorithm type         #
# Need to pass to this a libsbml species object and a type an integration type #
################################################################################

def getSpeciesValue(species, intType):
    if species.isSetInitialAmount() and species.isSetInitialConcentration():
        if intType == ODE or intType == SDE:
            return species.getInitialConcentration()
        else:  # implies intType = Gillespie
            return species.getInitialAmount()

    if species.isSetInitialAmount():
        return species.getInitialAmount()
    else:
        return species.getInitialConcentration()


##########################################
# Rename all parameters and species       #
##########################################
def rename(node, name, new_name):
    typ = node.getType()

    if typ == AST_NAME or typ == AST_NAME_TIME:
        nme = node.getName()
        if nme == name:
            node.setName(new_name)

    for n in range(0, node.getNumChildren()):
        rename(node.getChild(n), name, new_name)
    return node


##############
# The PARSER #
##############

def getSubstitutionMatrix(integrationType, o):

    mathPython = []
    mathCuda = []

    mathPython.append('log10')
    mathPython.append('acos')
    mathPython.append('asin')
    mathPython.append('atan')
    mathPython.append('time')
    mathPython.append('exp')
    mathPython.append('sqrt')
    mathPython.append('pow')
    mathPython.append('log')
    mathPython.append('sin')
    mathPython.append('cos')
    mathPython.append('ceil')
    mathPython.append('floor')
    mathPython.append('tan')

    mathCuda.append('log10')
    mathCuda.append('acos')
    mathCuda.append('asin')
    mathCuda.append('atan')

    if o.match(integrationType):
        mathCuda.append('t[0]')
    else:
        mathCuda.append('t')

    mathCuda.append('exp')
    mathCuda.append('sqrt')
    mathCuda.append('pow')
    mathCuda.append('log')
    mathCuda.append('sin')
    mathCuda.append('cos')
    mathCuda.append('ceil')
    mathCuda.append('floor')
    mathCuda.append('tan')

    return (mathCuda, mathPython)


def importSBMLCUDA(source, integrationType, ModelName=None, method=None, outpath=""):
    """
    ***** args *****
    source:
                  a list of strings.
                  Each tuple entry describes a SBML file to be parsed.

    integrationType:
                  a list of strings.
                  The length of this tuple is determined by the number of SBML 
                  files to be parsed. Each entry describes the simulation algorithm.
                  Possible algorithms are:
                  ODE         ---   for deterministic systems; solved with odeint (scipy)
                  SDE         ---   for stochastic systems; solved with sdeint (abc)
                  MJP   ---   for staochastic systems; solved with GillespieAlgorithm (abc)
                  DDE         ---  for deterministic systems with delays; solved with DelaySimulator

    ***** kwargs *****
    ModelName:
                  a list of strings.
                  ModelName describes the names of the parsed model files.

    method:
                  an integer number.
                  Type of noise in a stochastic system.
                  (Only implemented for stochastic systems solved with sdeint.)
                  Possible options are:
                  1 --- default
                  2 --- Ornstein-Uhlenbeck
                  3 --- geometric Brownian motion

    """

    # regular expressions for detecting integration types
    g = re.compile('MJP')
    o = re.compile('ODE')
    s = re.compile('SDE')
    d = re.compile('DDE')

    if not (len(source) == len(integrationType)):
        print "\nError: Number of sources is not the same as number of integrationTypes!\n"
        return

    # If model names not specified, default to model1, model2, ...
    if ModelName is None:
        ModelName = []
        for x in range(0, len(source)):
            ModelName.append("model" + repr(x + 1))


    for models in range(0, len(source)):

        # if no method is specified and the integrationType is "SDE", method type defaults to 1
        if method is None:
            if s.match(integrationType[models]):
                method = []
                for x in range(0, len(source)):
                    method.append(1)

        parameterId = []        # parameter IDs as given in the model
        parameterId2 = []       # new parameter IDs, of the form parameter01, parameter02, ...
        listOfParameter = []    # parameter (or compartment!) objects
        speciesId = []          # species IDs as given in the model
        speciesId2 = []         # new species IDs, of the form species01, species02, ...
        species = []            # species objects, as returned by model.getListOfSpecies()

        # Get the model
        reader = SBMLReader()
        document = reader.readSBML(source[models])
        model = document.getModel()

        # get basic model properties
        numSpecies = model.getNumSpecies()
        numReactions = model.getNumReactions()
        numGlobalParameters = model.getNumParameters()

        stoichiometricMatrix = empty([numSpecies, numReactions])

        # Add compartment volumes to lists of parameters
        listOfCompartments = model.getListOfCompartments()
        numCompartments = len(listOfCompartments)

        for i in range(0, numCompartments):
            if listOfCompartments[i].isSetVolume():
                parameterId.append(listOfCompartments[i].getId())
                parameterId2.append('compartment' + repr(i + 1))
                listOfParameter.append(model.getCompartment(i))

        # Get global parameters
        for i in range(0, numGlobalParameters):
            parameterId.append(model.getParameter(i).getId())
            if (len(parameterId2) - numCompartments) < 9:
                parameterId2.append('parameter0' + repr(i + 1))
            else:
                parameterId2.append('parameter' + repr(i + 1))
            listOfParameter.append(model.getParameter(i))

        ###############
        # get species #
        ###############

        reactant = []
        product = []

        S1 = []
        S2 = []

        # Get a list of species
        listOfSpecies = model.getListOfSpecies()

        for k in range(0, len(listOfSpecies)):
            species.append(listOfSpecies[k])
            speciesId.append(listOfSpecies[k].getId())
            if len(speciesId2) < 9:
                speciesId2.append('species0' + repr(k + 1))
            else:
                speciesId2.append('species' + repr(k + 1))

            # construct temporary placeholders
            S1.append(0.0)
            S2.append(0.0)
            reactant.append(0)
            product.append(0)

        ###############################
        # analyse the model structure #
        ###############################

        numReactants = []
        numProducts = []
        kineticLaw = []
        numLocalParameters = []

        # Get the list of reactions
        listOfReactions = model.getListOfReactions()

        # For every reaction
        for i in range(0, len(listOfReactions)):

            numReactants.append(listOfReactions[i].getNumReactants())
            numProducts.append(listOfReactions[i].getNumProducts())

            kineticLaw.append(listOfReactions[i].getKineticLaw().getFormula())
            numLocalParameters.append(listOfReactions[i].getKineticLaw().getNumParameters())

            # Zero all elements of S1 and s2
            for a in range(0, len(species)):
                S1[a] = 0.0
                S2[a] = 0.0

            # Fill non-zero elements of S1, such that S1[k] is the number of molecules of species[k] *consumed* when the
            # reaction happens once.
            for j in range(0, numReactants[i]):
                reactant[j] = listOfReactions[i].getReactant(j)

                for k in range(0, len(species)):
                    if reactant[j].getSpecies() == species[k].getId():
                        S1[k] = reactant[j].getStoichiometry()

            # Fill non-zero elements of S2, such that S2[k] is the number of molecules of species[k] *produced* when the
            # reaction happens once.
            for l in range(0, numProducts[i]):
                product[l] = listOfReactions[i].getProduct(l)

                for k in range(0, len(species)):
                    if product[l].getSpecies() == species[k].getId():
                        S2[k] = product[l].getStoichiometry()

            # Construct the row of the stoichiometry matrix corresponding to this reaction by subtracting S1 from S2
            for m in range(0, len(species)):
                stoichiometricMatrix[m][i] = -S1[m] + S2[m]

            for n in range(0, numLocalParameters[i]):
                parameterId.append(listOfReactions[i].getKineticLaw().getParameter(n).getId())
                if (len(parameterId2) - numCompartments) < 10:
                    parameterId2.append('parameter0' + repr(len(parameterId) - numCompartments))
                else:
                    parameterId2.append('parameter' + repr(len(parameterId) - numCompartments))
                listOfParameter.append(listOfReactions[i].getKineticLaw().getParameter(n))

                name = listOfReactions[i].getKineticLaw().getParameter(n).getId()
                new_name = 'parameter' + repr(len(parameterId) - numCompartments)
                node = model.getReaction(i).getKineticLaw().getMath()
                new_node = rename(node, name, new_name)
                kineticLaw[i] = formulaToString(new_node)

            for n in range(0, numCompartments):
                name = parameterId[n]
                new_name = 'compartment' + repr(n + 1)
                node = model.getReaction(i).getKineticLaw().getMath()
                new_node = rename(node, name, new_name)
                kineticLaw[i] = formulaToString(new_node)

        #####################
        # analyse functions #
        #####################

        listOfFunctions = model.getListOfFunctionDefinitions()

        FunctionArgument = []
        FunctionBody = []

        for fun in range(0, len(listOfFunctions)):
            FunctionArgument.append([])
            for funArg in range(0, listOfFunctions[fun].getNumArguments()):
                FunctionArgument[fun].append(formulaToString(listOfFunctions[fun].getArgument(funArg)))

            FunctionBody.append(formulaToString(listOfFunctions[fun].getBody()))

        for fun in range(0, len(listOfFunctions)):
            for funArg in range(0, listOfFunctions[fun].getNumArguments()):
                name = FunctionArgument[fun][funArg]
                node = listOfFunctions[fun].getBody()
                new_node = rename(node, name, "a" + repr(funArg + 1))
                FunctionBody[fun] = formulaToString(new_node)
                FunctionArgument[fun][funArg] = 'a' + repr(funArg + 1)

        #################
        # analyse rules #
        #################

        # Get the list of rules
        ruleFormula = []
        ruleVariable = []
        listOfRules = model.getListOfRules()
        for ru in range(0, len(listOfRules)):
            ruleFormula.append(listOfRules[ru].getFormula())
            ruleVariable.append(listOfRules[ru].getVariable())

        ##################
        # analyse events #
        ##################

        listOfEvents = model.getListOfEvents()

        EventCondition = []
        EventVariable = []
        EventFormula = []

        for eve in range(0, len(listOfEvents)):
            EventCondition.append(formulaToString(listOfEvents[eve].getTrigger().getMath()))
            listOfAssignmentRules = listOfEvents[eve].getListOfEventAssignments()
            EventVariable.append([])
            EventFormula.append([])
            for ru in range(0, len(listOfAssignmentRules)):
                EventVariable[eve].append(listOfAssignmentRules[ru].getVariable())
                EventFormula[eve].append(formulaToString(listOfAssignmentRules[ru].getMath()))

        ########################################################################
        # rename parameters and species in reactions, events, rules             #
        ########################################################################

        # Get paired list of function names for substitution
        (mathCuda, mathPython) = getSubstitutionMatrix(integrationType[models], o)

        NAMES = [[], []]
        NAMES[0].append(parameterId)
        NAMES[0].append(parameterId2)
        NAMES[1].append(speciesId)
        NAMES[1].append(speciesId2)

        for nam in range(0, 2):

            for i in range(0, len(NAMES[nam][0])):
                name = NAMES[nam][0][i]
                new_name = NAMES[nam][1][i]

                for k in range(0, numReactions):
                    node = model.getReaction(k).getKineticLaw().getMath()
                    new_node = rename(node, name, new_name)
                    kineticLaw[k] = formulaToString(new_node)

                for k in range(0, len(listOfRules)):
                    node = listOfRules[k].getMath()
                    new_node = rename(node, name, new_name)
                    ruleFormula[k] = formulaToString(new_node)
                    if ruleVariable[k] == name:
                        ruleVariable[k] = new_name

                for k in range(0, len(listOfEvents)):
                    node = listOfEvents[k].getTrigger().getMath()
                    new_node = rename(node, name, new_name)
                    EventCondition[k] = formulaToString(new_node)
                    listOfAssignmentRules = listOfEvents[k].getListOfEventAssignments()
                    for cond in range(0, len(listOfAssignmentRules)):
                        node = listOfAssignmentRules[cond].getMath()
                        new_node = rename(node, name, new_name)
                        EventFormula[k][cond] = formulaToString(new_node)
                        if EventVariable[k][cond] == name:
                            EventVariable[k][cond] = new_name

        for nam in range(0, len(mathPython)):

            for k in range(0, len(kineticLaw)):
                if re.search(mathPython[nam], kineticLaw[k]):
                    s = kineticLaw[k]
                    s = re.sub(mathPython[nam], mathCuda[nam], s)
                    kineticLaw[k] = s

            for k in range(0, len(ruleFormula)):
                if re.search(mathPython[nam], ruleFormula[k]):
                    s = ruleFormula[k]
                    s = re.sub(mathPython[nam], mathCuda[nam], s)
                    ruleFormula[k] = s

            for k in range(0, len(EventFormula)):
                for cond in range(0, len(listOfAssignmentRules)):
                    if re.search(mathPython[nam], EventFormula[k][cond]):
                        s = EventFormula[k][cond]
                        s = re.sub(mathPython[nam], mathCuda[nam], s)
                        EventFormula[k][cond] = s

            for k in range(0, len(EventCondition)):
                if re.search(mathPython[nam], EventCondition[k]):
                    s = EventCondition[k]
                    s = re.sub(mathPython[nam], mathCuda[nam], s)
                    EventCondition[k] = s

            for k in range(0, len(FunctionBody)):
                if re.search(mathPython[nam], FunctionBody[k]):
                    s = FunctionBody[k]
                    s = re.sub(mathPython[nam], mathCuda[nam], s)
                    FunctionBody[k] = s

            for fun in range(0, len(listOfFunctions)):
                for k in range(0, len(FunctionArgument[fun])):
                    if re.search(mathPython[nam], FunctionArgument[fun][k]):
                        s = FunctionArgument[fun][k]
                        s = re.sub(mathPython[nam], mathCuda[nam], s)
                        FunctionArgument[fun][k] = s

        # Get list of delays
        delays = set()

        print "Looking for delay"
        for n in range(0, model.getNumReactions()):
            r = model.getReaction(n)
            if r.isSetKineticLaw():
                kl = r.getKineticLaw()

                if kl.isSetMath():
                    formula = formulaToString(kl.getMath())

                    if "delay" in formula:
                        r = re.search("delay\((\w+?), (\w+?)\)", formula).groups()
                        paramName = r[1]
                        j = int(paramName.replace("parameter", ''))

                        memoryLocation = "tex2D(param_tex," + repr(j) + ",tid)"
                        delays.add(memoryLocation)

        delays = list(delays)


        ##########################
        # call writing functions #
        ##########################

        s = re.compile('SDE')
        if o.match(integrationType[models]):
            write_ODECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId2, listOfParameter,
                          parameterId2, ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                          ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath)
        if s.match(integrationType[models]):
            write_SDECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId2, listOfParameter,
                          parameterId2, ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                          ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath)
        if g.match(integrationType[models]):
            write_GillespieCUDA(stoichiometricMatrix, kineticLaw, numSpecies, numReactions, parameterId2, speciesId2,
                                ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                                ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula,
                                outpath)
        if d.match(integrationType[models]):
            write_DDECUDA(stoichiometricMatrix, kineticLaw, species, numReactions, speciesId2, listOfParameter,
                          parameterId2, ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules,
                          ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, delays, numCompartments,
                          outpath)

    return delays

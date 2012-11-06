from numpy import *
from libsbml import *
import re
import os


##To call the parser:
##    SBMLparse.importSBML(source, integrationType, ModelName=None,method=None)
##    All arguments to function must be passed as tuples.
##    If there is only one source to parse it must still be passed as a tuple ('source.xml',)
##    with an integrationType passed as ('Gillespie',)

## replace the species and parameters recursively
##
## replace
## pq = re.compile(speciesId[q])
## string=pq.sub('y['+repr(q)+']' ,string)
## with
## string = rep(string, speciesId[q],'y['+repr(q)+']')


def rep(str,find,replace):

    ex = find+"[^0-9]"
    ss = str;
    while re.search(ex,ss) != None:
        res = re.search(ex,ss)
        ss = ss[0:res.start()] + replace + " " + ss[res.end()-1:]

    ex = find+"$"
    if re.search(ex,ss) != None:
        res = re.search(ex,ss)
        ss = ss[0:res.start()] + replace + " " + ss[res.end():]
 
    return ss;


######################## CUDA SDE #################################

def write_SDECUDA(stoichiometricMatrix, kineticLaw, species, numSpecies, numGlobalParameters, numReactions, speciesId, listOfParameter, parameterId, parameter, InitValues, name, listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath=""):
    """
    Write the cuda file with ODE functions using the information taken by the parser
    """ 

    p=re.compile('\s')
    #Open the outfile
    out_file=open(os.path.join(outpath,name+".cu"),"w")

    #Write number of parameters and species
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(numGlobalParameters) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("\n")

    #The user-defined functions used in the model must be written in the file
    out_file.write("//Code for texture memory\n")

    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents+numRules
    if num>0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0,len(listOfFunctions)):
        out_file.write("__device__ float "+listOfFunctions[i].getId()+"(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float "+FunctionArgument[i][j])
            if(j<( listOfFunctions[i].getNumArguments()-1)):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")






    out_file.write("\n")

    out_file.write("__device__ void step(float *y, float t, unsigned *rngRegs, int tid){\n")

    numSpecies = len(species)

    #write rules and events
    for i in range(0,len(listOfRules)):
        if listOfRules[i].isRate() == True:
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)',string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')

            out_file.write(string)
            out_file.write(";\n")
            
    for i in range(0,len(listOfEvents)):
        out_file.write("    if( ")
        #print EventCondition[i]
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            #out_file.write("float ")
            if not(EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y["+repr(speciesId.index(EventVariable[i][j]))+"]"
                out_file.write(string) 
            out_file.write("=")
            
            string = EventFormula[i][j]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")
 
            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub("y["+repr(q)+"]" ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #x = "tex2D(param_tex,"+repr(q)+",tid)"
                        #string=pq.sub(x,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("")


    #Write the derivatives
    for i in range(0,numSpecies):
        
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False):
            out_file.write("    float d_y"+repr(i)+"= DT * (")
            if (species[i].isSetCompartment() == True):
                out_file.write("(")
            
            reactionWritten = False
            for k in range(0,numReactions):
                if(not stoichiometricMatrix[i][k]==0.0):

                    if(reactionWritten and stoichiometricMatrix[i][k]>0.0):
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")
                    
                    #test if reaction has a positive sign
                    #if(reactionWritten):
                    #    if(stoichiometricMatrix[i][k]>0.0):
                    #        out_file.write("+")
                    #    else:
                    #        out_file.write("-")
                    #reactionWritten = True
                    
                    #test if reaction is 1.0; then omit multiplication term
                    #if(abs(stoichiometricMatrix[i][k]) == 1.0):
                    #    out_file.write("(")
                    #else:
                    #    out_file.write(repr(abs(stoichiometricMatrix[i][k])))
                    #    out_file.write("*(")
                        
                    string = kineticLaw[k]
                    for q in range(0,len(speciesId)):
                        #pq = re.compile(speciesId[q])
                        #string=pq.sub('y['+repr(q)+']' ,string)
                        string = rep(string, speciesId[q],'y['+repr(q)+']')
                    for q in range(0,len(parameterId)):
                        if (not(parameterId[q] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[q] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                #pq = re.compile(parameterId[q])
                                #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                                string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
                                
                    string=p.sub('',string)
                    
                    out_file.write(string)
                    out_file.write(")")
                    
            if (species[i].isSetCompartment() == True):
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if (listOfParameter[j].getId() == mySpeciesCompartment):
                        if (not(parameterId[j] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[j] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                out_file.write("tex2D(param_tex,"+repr(j)+",tid)"+");")
                                break
                            else:
                                out_file.write(parameterId[j]+");")
                                break
            else:
                out_file.write(");")
            out_file.write("\n")
    
    out_file.write("\n")

    # check for columns of the stochiometry matrix with more than one entry
    randomVariables = ["*randNormal(rngRegs,sqrt(DT))"] * numReactions
    for k in range(0,numReactions):
        countEntries = 0
        for i in range(0,numSpecies):
            if(stoichiometricMatrix[i][k] != 0.0): countEntries += 1
        
        # define specific randomVariable
        if countEntries > 1:
            out_file.write("    float rand"+repr(k)+" = randNormal(rngRegs,sqrt(DT));\n")
            randomVariables[k] = "*rand" + repr(k)
    
    out_file.write("\n")
            
    #write noise terms
    for i in range(0,numSpecies):
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False):
            out_file.write("    d_y"+repr(i)+" += (")
            if (species[i].isSetCompartment() == True):
                out_file.write("(")
            
            reactionWritten = False
            for k in range(0,numReactions):
                if(not stoichiometricMatrix[i][k]==0.0):

                    if(reactionWritten and stoichiometricMatrix[i][k]>0.0):
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*sqrt(")
                    
                    #test if reaction has a positive sign
                    #if(reactionWritten):
                    #    if(stoichiometricMatrix[i][k]>0.0):
                    #        out_file.write("+")
                    #    else:
                    #        out_file.write("-")
                    #reactionWritten = True
                         
                    #test if reaction is 1.0; then omit multiplication term
                    #if(abs(stoichiometricMatrix[i][k]) == 1.0):
                    #    out_file.write("sqrtf(")
                    #else:
                    #    out_file.write(repr(abs(stoichiometricMatrix[i][k])))
                    #    out_file.write("*sqrtf(")

                    string = kineticLaw[k]
                    for q in range(0,len(speciesId)):
                        #pq = re.compile(speciesId[q])
                        #string=pq.sub('y['+repr(q)+']' ,string)
                        string = rep(string, speciesId[q],'y['+repr(q)+']')
                    for q in range(0,len(parameterId)):
                        if (not(parameterId[q] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[q] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                #pq = re.compile(parameterId[q])
                                #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                                string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
   
                    string=p.sub('',string)
                    out_file.write(string)
                    
                    #multiply random variable
                    out_file.write(")")
                    out_file.write(randomVariables[k])
                    #out_file.write("*randNormal(rngRegs,sqrt(DT))")
                    
                    
            if (species[i].isSetCompartment() == True):
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if (listOfParameter[j].getId() == mySpeciesCompartment):
                        if (not(parameterId[j] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[j] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                out_file.write("tex2D(param_tex,"+repr(j)+",tid)"+")")
                                break
                            else:
                                out_file.write(parameterId[j]+")")
                                break
            else:
                out_file.write(")")
            out_file.write(";\n")
    
    out_file.write("\n")
    #add terms
    for i in range(0,numSpecies):
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False ):
            out_file.write("    y["+repr(i)+"] += d_y"+repr(i)+";\n")
        
    out_file.write("}\n")

    
################# same file


    p=re.compile('\s')
    #The user-defined functions used in the model must be written in the file
    out_file.write("//Code for shared memory\n")

    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents+numRules
    if num>0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0,len(listOfFunctions)):
        out_file.write("__device__ float "+listOfFunctions[i].getId()+"(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float "+FunctionArgument[i][j])
            if(j<( listOfFunctions[i].getNumArguments()-1)):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")






    out_file.write("\n")
    out_file.write("__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){\n")

    numSpecies = len(species)

    #write rules and events
    for i in range(0,len(listOfRules)):
        if listOfRules[i].isRate() == True:
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        pq = re.compile(parameterId[q])
                        string=pq.sub('parameter['+repr(q)+']' ,string)

            out_file.write(string)
            out_file.write(";\n")
            
    for i in range(0,len(listOfEvents)):
        out_file.write("    if( ")
        #print EventCondition[i]
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            #out_file.write("float ")
            if not(EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y["+repr(speciesId.index(EventVariable[i][j]))+"]"
                out_file.write(string) 
            out_file.write("=")
            
            string = EventFormula[i][j]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        pq = re.compile(parameterId[q])
                        string=pq.sub('parameter['+repr(q)+']' ,string)

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")
 
            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub("y["+repr(q)+"]" ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        pq = re.compile(parameterId[q])
                        x = "parameter["+repr(q)+"]"
                        string=pq.sub(x,string)
            out_file.write(string)
            out_file.write(";\n")
    #out_file.write("\n\n")


    #Write the derivatives
    for i in range(0,numSpecies):
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False):
            out_file.write("    float d_y"+repr(i)+"= DT * (")
            if (species[i].isSetCompartment() == True):
                out_file.write("(")
            
            reactionWritten = False
            for k in range(0,numReactions):
                if(not stoichiometricMatrix[i][k]==0.0):

                    if(reactionWritten and stoichiometricMatrix[i][k]>0.0):
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")
                    
                    #test if reaction has a positive sign
                    #if(reactionWritten):
                    #    if(stoichiometricMatrix[i][k]>0.0):
                    #        out_file.write("+")
                    #    else:
                    #        out_file.write("-")
                    #reactionWritten = True
                    
                    #test if reaction is 1.0; then omit multiplication term
                    #if(abs(stoichiometricMatrix[i][k]) == 1.0):
                    #    out_file.write("(")
                    #else:
                    #    out_file.write(repr(abs(stoichiometricMatrix[i][k])))
                    #    out_file.write("*(")

                    string = kineticLaw[k]
                    for q in range(0,len(speciesId)):
                        #pq = re.compile(speciesId[q])
                        #string=pq.sub('y['+repr(q)+']' ,string)
                        string = rep(string, speciesId[q],'y['+repr(q)+']')
                    for q in range(0,len(parameterId)):
                        if (not(parameterId[q] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[q] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                pq = re.compile(parameterId[q])
                                string=pq.sub('parameter['+repr(q)+']' ,string)
   
                    string=p.sub('',string)
                    
                    out_file.write(string)
                    out_file.write(")")
                    
            if (species[i].isSetCompartment() == True):
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if (listOfParameter[j].getId() == mySpeciesCompartment):
                        if (not(parameterId[j] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[j] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                out_file.write("parameter["+repr(j)+"]"+");")
                                break
                            else:
                                out_file.write(parameterId[j]+");")                                
                                break
            else:
                out_file.write(");")
            out_file.write("\n")

    out_file.write("\n")

    # check for columns of the stochiometry matrix with more than one entry
    randomVariables = ["*randNormal(rngRegs,sqrt(DT))"] * numReactions
    for k in range(0,numReactions):
        countEntries = 0
        for i in range(0,numSpecies):
            if(stoichiometricMatrix[i][k] != 0.0): countEntries += 1
        
        # define specific randomVariable
        if countEntries > 1:
            out_file.write("    float rand"+repr(k)+" = randNormal(rngRegs,sqrt(DT));\n")
            randomVariables[k] = "*rand" + repr(k)
    
    out_file.write("\n")

    #write noise terms
    for i in range(0,numSpecies):
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False):
            out_file.write("    d_y"+repr(i)+"+= (")
            if (species[i].isSetCompartment() == True):
                out_file.write("(")
                
            reactionWritten = False
            for k in range(0,numReactions):
                if(not stoichiometricMatrix[i][k]==0.0):

                    if(reactionWritten and stoichiometricMatrix[i][k]>0.0):
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*sqrt(")
                    
                    #test if reaction has a positive sign
                    #if(reactionWritten):
                    #    if(stoichiometricMatrix[i][k]>0.0):
                    #        out_file.write("+")
                    #    else:
                    #        out_file.write("-")
                    #reactionWritten = True
                         
                    #test if reaction is 1.0; then omit multiplication term
                    #if(abs(stoichiometricMatrix[i][k]) == 1.0):
                    #    out_file.write("sqrtf(")
                    #else:
                    #    out_file.write(repr(abs(stoichiometricMatrix[i][k])))
                    #    out_file.write("*sqrtf(")

                    string = kineticLaw[k]
                    for q in range(0,len(speciesId)):
                        #pq = re.compile(speciesId[q])
                        #string=pq.sub('y['+repr(q)+']' ,string)
                        string = rep(string, speciesId[q],'y['+repr(q)+']')
                    for q in range(0,len(parameterId)):
                        if (not(parameterId[q] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[q] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                pq = re.compile(parameterId[q])
                                string=pq.sub('parameter['+repr(q)+']' ,string)
   
                    string=p.sub('',string)
                    out_file.write(string)
                    
                    #multiply random variable
                    out_file.write(")")
                    out_file.write(randomVariables[k])
                    #out_file.write("*randNormal(rngRegs,sqrt(DT))")
                    
            if (species[i].isSetCompartment() == True):
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if (listOfParameter[j].getId() == mySpeciesCompartment):
                        if (not(parameterId[j] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[j] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                out_file.write("parameter["+repr(j)+"]"+")")
                                break
                            else:
                                out_file.write(parameterId[j]+")")                                
                                break
            else:
                out_file.write(")")

            out_file.write(";\n")

    out_file.write("\n")
    #add terms
    for i in range(0,numSpecies):
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False):
            out_file.write("    y["+repr(i)+"] += d_y"+repr(i)+";\n")
        
    out_file.write("}\n")

    








######################## CUDA Gillespie #################################

def write_GillespieCUDA(stoichiometricMatrix, kineticLaw, numSpecies, numGlobalParameters, numReactions, species, parameterId, InitValues, speciesId,name, listOfFunctions,FunctionArgument,FunctionBody, listOfRules, ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath=""):
    p=re.compile('\s')
    #Open the outfile
    out_file=open(os.path.join(outpath,name+".cu"),"w")

    #Write number of parameters and species
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(numGlobalParameters) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("\n")

    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents+numRules
    if num>0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")
    

    for i in range(0,len(listOfFunctions)):
        out_file.write("__device__ float "+listOfFunctions[i].getId()+"(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float "+FunctionArgument[i][j])
            if(j<( listOfFunctions[i].getNumArguments()-1)):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("")

    out_file.write("\n\n__constant__ int smatrix[]={\n")
    for i in range(0,len(stoichiometricMatrix[0])):
        for j in range(0,len(stoichiometricMatrix)):
            out_file.write("    "+repr(stoichiometricMatrix[j][i]))
            if (not(i==(len(stoichiometricMatrix)-1) and (j==(len(stoichiometricMatrix[0])-1)))):
                out_file.write(",")
        out_file.write("\n")


    out_file.write("};\n\n\n")

    
    #stoichiometry function moved to Gillespie.py
    #out_file.write("__device__ void stoichiometry(int *y, int r, int tid){\n")
    #out_file.write("    for(int i=0; i<"+repr(len(species))+"; i++){\n        y[i]+=smatrix[r*"+repr(len(species))+"+ i];\n    }\n}\n\n\n")
    
    out_file.write("__device__ void hazards(int *y, float *h, float t, int tid){")
    # wirte rules and events 
    for i in range(0,len(listOfRules)):
        if listOfRules[i].isRate() == True:
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')

            out_file.write(string)
            out_file.write(";\n")
            
    for i in range(0,len(listOfEvents)):
        out_file.write("    if( ")
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            #out_file.write("float ")
            if not(EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y["+repr(speciesId.index(EventVariable[i][j]))+"]"
                out_file.write(string)           
            out_file.write("=")
            
            string = EventFormula[i][j]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("    }\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")
 
            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub("y["+repr(q)+"]" ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #x = "tex2D(param_tex,"+repr(q)+",tid)"
                        #string=pq.sub(x,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("\n")



    for i in range(0,numReactions):
        out_file.write("    h["+repr(i)+"] = ")

        string = kineticLaw[i]
        for q in range(0,len(speciesId)):
            #pq = re.compile(speciesId[q])
            #string=pq.sub('y['+repr(q)+']' ,string)
            string = rep(string, speciesId[q],'y['+repr(q)+']')
        for q in range(0,len(parameterId)):
            if (not(parameterId[q] in ruleVariable)):
                flag = False
                for r in range(0,len(EventVariable)):
                    if (parameterId[q] in EventVariable[r]):
                        flag = True
                if flag==False:
                    #pq = re.compile(parameterId[q])
                    #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                    string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
   
        string=p.sub('',string)
        out_file.write(string+";\n")
        
    out_file.write("\n")
    out_file.write("}\n\n")
    



######################## CUDA ODE #################################

def write_ODECUDA(stoichiometricMatrix, kineticLaw, species, numSpecies, numGlobalParameters, numReactions, speciesId, listOfParameter, parameterId,parameter,InitValues,name, listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath=""):
    """
    Write the cuda file with ODE functions using the information taken by the parser
    """ 
    p=re.compile('\s')
    #Open the outfile
    out_file=open(os.path.join(outpath,name+".cu"),"w")

    #Write number of parameters and species
    out_file.write("#define NSPECIES " + str(numSpecies) + "\n")
    out_file.write("#define NPARAM " + str(numGlobalParameters) + "\n")
    out_file.write("#define NREACT " + str(numReactions) + "\n")
    out_file.write("\n")

    #The user-defined functions used in the model must be written in the file
    numEvents = len(listOfEvents)
    numRules = len(listOfRules)
    num = numEvents+numRules
    if num>0:
        out_file.write("#define leq(a,b) a<=b\n")
        out_file.write("#define neq(a,b) a!=b\n")
        out_file.write("#define geq(a,b) a>=b\n")
        out_file.write("#define lt(a,b) a<b\n")
        out_file.write("#define gt(a,b) a>b\n")
        out_file.write("#define eq(a,b) a==b\n")
        out_file.write("#define and_(a,b) a&&b\n")
        out_file.write("#define or_(a,b) a||b\n")

    for i in range(0,len(listOfFunctions)):
        out_file.write("__device__ float "+listOfFunctions[i].getId()+"(")
        for j in range(0, listOfFunctions[i].getNumArguments()):
            out_file.write("float "+FunctionArgument[i][j])
            if(j<( listOfFunctions[i].getNumArguments()-1)):
                out_file.write(",")
        out_file.write("){\n    return ")
        out_file.write(FunctionBody[i])
        out_file.write(";\n}\n")
        out_file.write("\n")






    out_file.write("struct myFex{\n    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)\n    {\n        int tid = blockDim.x * blockIdx.x + threadIdx.x;\n")

    numSpecies = len(species)

    #write rules and events
    for i in range(0,len(listOfRules)):
        if listOfRules[i].isRate() == True:
            out_file.write("        ")
            if not(ruleVariable[i] in speciesId):
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")

            string = ruleFormula[i]
            for q in range(0,len(speciesId)):
                pq = re.compile(speciesId[q])
                string=pq.sub('y['+repr(q)+']' ,string)
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        pq = re.compile(parameterId[q])
                        string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)

            out_file.write(string)
            out_file.write(";\n")
            
    for i in range(0,len(listOfEvents)):
        out_file.write("    if( ")
        #print EventCondition[i]
        out_file.write(mathMLConditionParserCuda(EventCondition[i]))
        out_file.write("){\n")
        listOfAssignmentRules = listOfEvents[i].getListOfEventAssignments()
        for j in range(0, len(listOfAssignmentRules)):
            out_file.write("        ")
            #out_file.write("float ")
            if not(EventVariable[i][j] in speciesId):
                out_file.write(EventVariable[i][j])
            else:
                string = "y["+repr(speciesId.index(EventVariable[i][j]))+"]"
                out_file.write(string) 
            out_file.write("=")
            
            string = EventFormula[i][j]
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub('y['+repr(q)+']' ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')

            out_file.write(string)
            out_file.write(";\n")
        out_file.write("}\n")

    out_file.write("\n")

    for i in range(0, len(listOfRules)):
        if listOfRules[i].isAssignment():
            out_file.write("    ")
            if not(ruleVariable[i] in speciesId):
                out_file.write("float ")
                out_file.write(ruleVariable[i])
            else:
                string = "y["+repr(speciesId.index(ruleVariable[i]))+"]"
                out_file.write(string)
            out_file.write("=")
 
            string = mathMLConditionParserCuda(ruleFormula[i])
            for q in range(0,len(speciesId)):
                #pq = re.compile(speciesId[q])
                #string=pq.sub("y["+repr(q)+"]" ,string)
                string = rep(string, speciesId[q],'y['+repr(q)+']')
            for q in range(0,len(parameterId)):
                if (not(parameterId[q] in ruleVariable)):
                    flag = False
                    for r in range(0,len(EventVariable)):
                        if (parameterId[q] in EventVariable[r]):
                            flag = True
                    if flag==False:
                        #pq = re.compile(parameterId[q])
                        #x = "tex2D(param_tex,"+repr(q)+",tid)"
                        #string=pq.sub(x,string)
                        string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
            out_file.write(string)
            out_file.write(";\n")
    out_file.write("\n\n")

    #Write the derivatives
    for i in range(0,numSpecies):
        if (species[i].getConstant() == False and species[i].getBoundaryCondition() == False):
            out_file.write("        ydot["+repr(i)+"]=")
            if (species[i].isSetCompartment() == True):
                out_file.write("(")
                
            reactionWritten = False
            for k in range(0,numReactions):
                if(not stoichiometricMatrix[i][k]==0.0):

                    if(reactionWritten and stoichiometricMatrix[i][k]>0.0):
                        out_file.write("+")
                    reactionWritten = True
                    out_file.write(repr(stoichiometricMatrix[i][k]))
                    out_file.write("*(")
                    
                    #test if reaction has a positive sign
                    #if(reactionWritten):
                    #    if(stoichiometricMatrix[i][k]>0.0):
                    #        out_file.write("+")
                    #    else:
                    #        out_file.write("-")
                    #reactionWritten = True
                    
                    #test if reaction is 1.0; then omit multiplication term
                    #if(abs(stoichiometricMatrix[i][k]) == 1.0):
                    #    out_file.write("(")
                    #else:
                    #    out_file.write(repr(abs(stoichiometricMatrix[i][k])))
                    #    out_file.write("*(")

                    string = kineticLaw[k]
                    for q in range(0,len(speciesId)):
                        #pq = re.compile(speciesId[q]+'[^0-9]')
                        #pq = re.compile(speciesId[q]+'[^0-9]')
                        #pq = re.compile(speciesId[q])
                        #ret=pq.sub('y['+repr(q)+']' ,string)

                        string = rep(string, speciesId[q],'y['+repr(q)+']')
                        
                        ##if ret != string:
                        #if q == 5: 
                        #    print speciesId[q], "|", 'y['+repr(q)+']', "\n\t", string, "\n\t", ret
                        #string = ret;
                    for q in range(0,len(parameterId)):
                        if (not(parameterId[q] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[q] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                #pq = re.compile(parameterId[q])
                                #string=pq.sub('tex2D(param_tex,'+repr(q)+',tid)' ,string)
                                string = rep(string, parameterId[q],'tex2D(param_tex,'+repr(q)+',tid)')
   
                    string=p.sub('',string)

                    ##print string
                    out_file.write(string)
                    out_file.write(")")
                    
                    
            if (species[i].isSetCompartment() == True):
                out_file.write(")/")
                mySpeciesCompartment = species[i].getCompartment()
                for j in range(0, len(listOfParameter)):
                    if (listOfParameter[j].getId() == mySpeciesCompartment):
                        if (not(parameterId[j] in ruleVariable)):
                            flag = False
                            for r in range(0,len(EventVariable)):
                                if (parameterId[j] in EventVariable[r]):
                                    flag = True
                            if flag==False:
                                out_file.write("tex2D(param_tex,"+repr(j)+",tid)"+";")
                                break
                            else:
                                out_file.write(parameterId[j]+";")                                
                                break

            else:
                out_file.write(";")
            out_file.write("\n")


    out_file.write("\n    }")
    out_file.write("\n};\n\n\n struct myJex{\n    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){\n        return; \n    }\n};") 



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
        if intType==ODE or intType==SDE:
            return species.getInitialConcentration()
        else: #implies intType = Gillespie
            return species.getInitialAmount()

    if species.isSetInitialAmount():
        return species.getInitialAmount()
    else:
        return species.getInitialConcentration()

##########################################
#Rename all parameters and species       #
##########################################
def rename(node,name,new_name):
    typ = node.getType()
    
    if (typ==AST_NAME or typ==AST_NAME_TIME):
        nme = node.getName()
        if nme == name:
            node.setName(new_name)

    for n in range(0,node.getNumChildren()):
        rename(node.getChild(n),name,new_name)
    return node


##############
# The PARSER #
##############

def importSBMLCUDA(source,integrationType,ModelName=None,method=None,outpath=""):

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

 
    #regular expressions for detecting integration types
    g=re.compile('MJP')
    o=re.compile('ODE')
    s=re.compile('SDE')
    
    #output general properties
    #output = []

    #check that you have appropriate lengths of integration types and sources
    #(need equal lengths)
    if(not(len(source)==len(integrationType))):
        print "\nError: Number of sources is not the same as number of integrationTypes!\n"
    #check that you have model names,
    #if not the models will be named model1, model2, etc
    else:        
        if(ModelName==None):
            ModelName=[]
            for x in range(0,len(source)):
                ModelName.append("model"+repr(x+1))

        #if no method is specified and the integrationType is "SDE"
        #the method type defaults to 1
        for models in range(0,len(source)):
            intType = integrationType[models]
            if method==None:
                if s.match(integrationType[models]):
                    method=[]
                    for x in range(0, len(source)):
                        method.append(1)

        #All the below should happen for each model
        #Arguments to pass to the writing functions:
            #species IDs
            #species concentrations (initial values from model)
            #reactions in the form of kinetic law list
            #stoichiometric matrix
            #parameters
            #values of parameters
            #name of output file
            #list of functions if they need to be defined at the top of the written .py file

                        
            #I think that we can pass parameters directly to the writing functions, non?
            parameterId=[]
            parameterId2=[]
            parameter=[]
            listOfParameter=[]
            #Likewise species?
            speciesId=[]
            speciesId2=[]
            species=[]

##    r=re.compile('.mod')
##    if(r.search(source)):
##        old_source=source
##        source=r.sub(".xml",old_source)
##        call='python mod2sbml.py '+old_source+' > '+ source
##        os.system(call)

            #Get the model
            reader=SBMLReader()
            document=reader.readSBML(source[models])
            model=document.getModel()
            
            #get basic model properties
            numSpeciesTypes=model.getNumSpeciesTypes()
            numSpecies=model.getNumSpecies()
            numReactions=model.getNumReactions()
            numGlobalParameters=model.getNumParameters()
            numFunctions=model.getNumFunctionDefinitions()
            
            stoichiometricMatrix=empty([numSpecies, numReactions])
            
            #output.append((numReactions,numGlobalParameters+1,numSpecies))


 
#################################################################################################
# get compartment volume/size - if it is set, pass as parameter with corresponding Id and value #
#################################################################################################

            listOfCompartments = model.getListOfCompartments()  
            comp=0
            for i in range(0, len(listOfCompartments)):
            #    listOfCompartments[i].setId('compartment'+repr(i+1))
                if listOfCompartments[i].isSetVolume():
                    comp=comp+1
                    parameterId.append(listOfCompartments[i].getId())
                    parameterId2.append('compartment'+repr(i+1))
                    parameter.append(listOfCompartments[i].getVolume())
                    listOfParameter.append(model.getCompartment(i))


#########################
# get global parameters #
#########################

            for i in range(0,numGlobalParameters):
                parameterId.append(model.getParameter(i).getId())
                if ((len(parameterId2)-comp)<9):
                    parameterId2.append('parameter0'+repr(i+1))
                else:
                    parameterId2.append('parameter'+repr(i+1))
                parameter.append(model.getParameter(i).getValue())
                listOfParameter.append(model.getParameter(i))


###############
# get species #
###############

            #Empty matrix to hold reactants
            reactant=[]
            #Empty matrix to hold products
            product=[]
            #Empty matrix to hold Species Ids
            #Empty matrix to hold the InitValues used going forward
            InitValues=[]
            S1 = []
            S2 = []

            #Get a list of species
            listOfSpecies = model.getListOfSpecies()
            #Make the matrices long enough
            for k in range(0, len(listOfSpecies)):
                species.append(listOfSpecies[k])
                speciesId.append(listOfSpecies[k].getId())
                if (len(speciesId2)<9):
                    speciesId2.append('species0'+repr(k+1))
                else:
                    speciesId2.append('species'+repr(k+1))
                #get the initial value
                #Need to fix this part
                #So that it will take getInitialConcentration
                #or getInitialValue as appropriate
                InitValues.append(getSpeciesValue(listOfSpecies[k],intType))
                #I'm not really sure what this part is doing
                #Hopefully it will become more clear later
                S1.append(0.0)
                S2.append(0.0)
                #placeholder in reactant matrix for this species
                reactant.append(0)
                #placeholder in product matrix for this species
                product.append(0)

###############################
# analyse the model structure #
###############################

            reaction=[]
            numReactants=[]
            numProducts=[]
            kineticLaw=[]
            numLocalParameters=[]

            #Get the list of reactions
            listOfReactions = model.getListOfReactions()

            #For every reaction    
            for i in range(0, len(listOfReactions)):
                #What does this part do?
                for a in range(0, len(species)):
                    #what do S1 and S2 represent?
                    #S1 is something to do with stoichimetry of reactants
                    #At the moment S1 and S2 are as long as len(species)
                    S1[a]=0.0
                    #S2 is something to do with stoichiometry of products
                    S2[a]=0.0
        
                numReactants.append(listOfReactions[i].getNumReactants())
                numProducts.append(listOfReactions[i].getNumProducts())
                
                kineticLaw.append(listOfReactions[i].getKineticLaw().getFormula())
                numLocalParameters.append(listOfReactions[i].getKineticLaw().getNumParameters())

                for j in range(0, numReactants[i]):
                    reactant[j]=listOfReactions[i].getReactant(j)
        
                    for k in range(0,len(species)):
                        if (reactant[j].getSpecies()==species[k].getId()):
                            S1[k]=reactant[j].getStoichiometry()

                    
                for l in range(0,numProducts[i]):
                    product[l]=listOfReactions[i].getProduct(l)
        
                    for k in range(0,len(species)):
                        if (product[l].getSpecies()==species[k].getId()):
                            S2[k]=product[l].getStoichiometry()

                for m in range(0, len(species)):
                    stoichiometricMatrix[m][i]=-S1[m]+S2[m]

                for n in range(0,numLocalParameters[i]):
                    parameterId.append(listOfReactions[i].getKineticLaw().getParameter(n).getId())
                    if ((len(parameterId2)-comp)<10):
                        parameterId2.append('parameter0'+repr(len(parameterId)-comp))
                    else:
                        parameterId2.append('parameter'+repr(len(parameterId)-comp))
                    parameter.append(listOfReactions[i].getKineticLaw().getParameter(n).getValue())
                    listOfParameter.append(listOfReactions[i].getKineticLaw().getParameter(n))

                    name=listOfReactions[i].getKineticLaw().getParameter(n).getId()
                    new_name='parameter'+repr(len(parameterId)-comp)
                    node=model.getReaction(i).getKineticLaw().getMath()
                    new_node=rename(node,name,new_name)
                    kineticLaw[i]=formulaToString(new_node)

                for n in range(0,comp):
                    
                    name=parameterId[n]
                    new_name='compartment'+repr(n+1)
                    node=model.getReaction(i).getKineticLaw().getMath()
                    new_node=rename(node,name,new_name)
                    kineticLaw[i]=formulaToString(new_node)

#####################
# analyse functions #
#####################

            #Get the list of functions

            listOfFunctions = model.getListOfFunctionDefinitions()


            FunctionArgument=[]
            FunctionBody=[]
                
            for fun in range(0,len(listOfFunctions)):
                FunctionArgument.append([])
                for funArg in range(0, listOfFunctions[fun].getNumArguments()):
                    FunctionArgument[fun].append(formulaToString(listOfFunctions[fun].getArgument(funArg)))

                FunctionBody.append(formulaToString(listOfFunctions[fun].getBody()))

            for fun in range(0, len(listOfFunctions)):
                for funArg in range(0,listOfFunctions[fun].getNumArguments()):
                    name=FunctionArgument[fun][funArg]
                    node=listOfFunctions[fun].getBody()
                    new_node=rename(node,name,"a"+repr(funArg+1))
                    FunctionBody[fun]=formulaToString(new_node)
                    FunctionArgument[fun][funArg]='a'+repr(funArg+1)
                   
        
#################
# analyse rules #
#################

            #Get the list of rules
            ruleFormula=[]
            ruleVariable=[]
            listOfRules = model.getListOfRules()
            for ru in range(0,len(listOfRules)):
                ruleFormula.append(listOfRules[ru].getFormula())
                ruleVariable.append(listOfRules[ru].getVariable())


##################
# analyse events #
##################
   
            listOfEvents = model.getListOfEvents()

            EventCondition=[]
            EventVariable=[]
            EventFormula=[]
           # listOfAssignmentRules=[]

            for eve in range(0,len(listOfEvents)):
                EventCondition.append(formulaToString(listOfEvents[eve].getTrigger().getMath()))
                listOfAssignmentRules=listOfEvents[eve].getListOfEventAssignments()
                EventVariable.append([])
                EventFormula.append([])
                for ru in range(0, len(listOfAssignmentRules)):
                   EventVariable[eve].append(listOfAssignmentRules[ru].getVariable())
                   EventFormula[eve].append(formulaToString(listOfAssignmentRules[ru].getMath()))



########################################################################
#rename math expressions from python to cuda                           #
########################################################################


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


            mathCuda.append('__log10f')
            mathCuda.append('acosf')
            mathCuda.append('asinf')
            mathCuda.append('atanf')

            if o.match(integrationType[models]):
                mathCuda.append('t[0]')
            if g.match(integrationType[models]):
                mathCuda.append('t')
            s=re.compile('SDE')
            if s.match(integrationType[models]):
                mathCuda.append('t')

            mathCuda.append('__expf')
            mathCuda.append('sqrtf')
            mathCuda.append('__powf')
            mathCuda.append('__logf')
            mathCuda.append('__sinf')
            mathCuda.append('__cosf')
            mathCuda.append('ceilf')
            mathCuda.append('floorf')
            mathCuda.append('__tanf')


        
########################################################################
#rename parameters and species in reactions, events, rules             #
########################################################################

            NAMES=[[],[]]
            NAMES[0].append(parameterId)
            NAMES[0].append(parameterId2)
            NAMES[1].append(speciesId)
            NAMES[1].append(speciesId2)

            for nam in range(0,2):
                
                for i in range(0, len(NAMES[nam][0])):
                    name=NAMES[nam][0][i]
                    new_name=NAMES[nam][1][i]
             
                    for k in range(0, numReactions):
                        node=model.getReaction(k).getKineticLaw().getMath()
                        new_node=rename(node,name,new_name)
                        kineticLaw[k]=formulaToString(new_node)
                        
                    for k in range(0,len(listOfRules)):
                        node=listOfRules[k].getMath()
                        new_node=rename(node,name,new_name)
                        ruleFormula[k]=formulaToString(new_node)
                        if ruleVariable[k]==name: ruleVariable[k]=new_name


                    for k in range(0,len(listOfEvents)):
                        node=listOfEvents[k].getTrigger().getMath()
                        new_node=rename(node,name,new_name)
                        EventCondition[k]=formulaToString(new_node)
                        listOfAssignmentRules=listOfEvents[k].getListOfEventAssignments()
                        for cond in range(0, len(listOfAssignmentRules)):
                            node=listOfAssignmentRules[cond].getMath()
                            new_node=rename(node,name,new_name)
                            EventFormula[k][cond]=formulaToString(new_node)
                            if EventVariable[k][cond]==name: EventVariable[k][cond]=new_name

               

            for nam in range(0,len(mathPython)):
 
                        
                for k in range(0,len(kineticLaw)):
                    if re.search(mathPython[nam],kineticLaw[k]):
                        s = kineticLaw[k]
                        s = re.sub(mathPython[nam],mathCuda[nam],s)
                        kineticLaw[k]=s

                for k in range(0,len(ruleFormula)):
                    if re.search(mathPython[nam],ruleFormula[k]):
                        s = ruleFormula[k]
                        s = re.sub(mathPython[nam],mathCuda[nam],s)
                        ruleFormula[k]=s

                for k in range(0,len(EventFormula)):
                    for cond in range(0, len(listOfAssignmentRules)):
                        if re.search(mathPython[nam],EventFormula[k][cond]):
                            s = EventFormula[k][cond]
                            s = re.sub(mathPython[nam],mathCuda[nam],s)
                            EventFormula[k][cond]=s

                for k in range(0,len(EventCondition)):
                    if re.search(mathPython[nam],EventCondition[k]):
                        s = EventCondition[k]
                        s = re.sub(mathPython[nam],mathCuda[nam],s)
                        EventCondition[k]=s

                for k in range(0,len(FunctionBody)):
                    if re.search(mathPython[nam],FunctionBody[k]):
                        s = FunctionBody[k]
                        s = re.sub(mathPython[nam],mathCuda[nam],s)
                        FunctionBody[k]=s

                for fun in range(0, len(listOfFunctions)):
                    for k in range(0,len(FunctionArgument[fun])):
                        if re.search(mathPython[nam],FunctionArgument[fun][k]):
                            s = FunctionArgument[fun][k]
                            s = re.sub(mathPython[nam],mathCuda[nam],s)
                            FunctionArgument[fun][k]=s

          
##########################
# call writing functions #
##########################

            s=re.compile('SDE')
            if o.match(integrationType[models]):
                write_ODECUDA(stoichiometricMatrix, kineticLaw, species, numSpecies, numGlobalParameters+1, numReactions, speciesId2, listOfParameter, parameterId2, parameter, InitValues, ModelName[models], listOfFunctions,FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable,EventFormula, outpath)
            if s.match(integrationType[models]):
                write_SDECUDA(stoichiometricMatrix, kineticLaw, species, numSpecies, numGlobalParameters+1, numReactions, speciesId2,listOfParameter, parameterId2, parameter,InitValues,ModelName[models], listOfFunctions, FunctionArgument, FunctionBody, listOfRules, ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath)
            if g.match(integrationType[models]):
                  write_GillespieCUDA(stoichiometricMatrix, kineticLaw, numSpecies, numGlobalParameters+1, numReactions, species, parameterId2, InitValues, speciesId2,ModelName[models], listOfFunctions,FunctionArgument,FunctionBody, listOfRules, ruleFormula, ruleVariable, listOfEvents, EventCondition, EventVariable, EventFormula, outpath)
    
    # output is:
    # (numReactions,numGlobalParameters,numSpecies)
    # return output

import re


def list_math_function_names(time_symbol):
    mathPython = []
    mathPython.append('log10')
    mathPython.append('acos')
    mathPython.append('asin')
    mathPython.append('atan')
    mathPython.append('exp')
    mathPython.append('sqrt')
    mathPython.append('pow')
    mathPython.append('log')
    mathPython.append('sin')
    mathPython.append('cos')
    mathPython.append('ceil')
    mathPython.append('floor')
    mathPython.append('tan')
    mathPython.append('time')

    mathCuda = []
    mathCuda.append('log10')
    mathCuda.append('acos')
    mathCuda.append('asin')
    mathCuda.append('atan')
    mathCuda.append('exp')
    mathCuda.append('sqrt')
    mathCuda.append('mpow')
    mathCuda.append('log')
    mathCuda.append('sin')
    mathCuda.append('cos')
    mathCuda.append('ceil')
    mathCuda.append('floor')
    mathCuda.append('tan')
    mathCuda.append(time_symbol)

    return (mathCuda, mathPython)


def rename_math_functions(model, time_symbol):

    (mathCuda, mathPython) = list_math_function_names(time_symbol)

    for nam in range(len(mathPython)):
        for k in range(len(model.kineticLaw)):
            if re.search(mathPython[nam], model.kineticLaw[k]):
                s = model.kineticLaw[k]
                s = re.sub(mathPython[nam], mathCuda[nam], s)
                model.kineticLaw[k] = s
    
        for k in range(len(model.ruleFormula)):
            if re.search(mathPython[nam], model.ruleFormula[k]):
                s = model.ruleFormula[k]
                s = re.sub(mathPython[nam], mathCuda[nam], s)
                model.ruleFormula[k] = s
    
        for k in range(len(model.eventFormula)):
            for cond in range(len(self.listOfAssignmentRules)):
                if re.search(mathPython[nam], model.eventFormula[k][cond]):
                    s = model.eventFormula[k][cond]
                    s = re.sub(mathPython[nam], mathCuda[nam], s)
                    model.eventFormula[k][cond] = s
    
        for k in range(len(model.eventCondition)):
            if re.search(mathPython[nam], model.eventCondition[k]):
                s = model.eventCondition[k]
                s = re.sub(mathPython[nam], mathCuda[nam], s)
                model.eventCondition[k] = s
    
        for k in range(len(model.functionBody)):
            if re.search(mathPython[nam], model.functionBody[k]):
                s = model.functionBody[k]
                s = re.sub(mathPython[nam], mathCuda[nam], s)
                model.functionBody[k] = s
    
        for fun in range(len(model.listOfFunctions)):
            for k in range(len(model.functionArgument[fun])):
                if re.search(mathPython[nam], model.functionArgument[fun][k]):
                    s = model.functionArgument[fun][k]
                    s = re.sub(mathPython[nam], mathCuda[nam], s)
                    model.functionArgument[fun][k] = s

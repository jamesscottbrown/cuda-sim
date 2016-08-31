import re


def rename_math_functions(model, time_symbol):

    math_python = ['log10', 'acos', 'asin', 'atan', 'exp', 'sqrt', 'pow', 'log', 'sin', 'cos', 'ceil', 'floor', 'tan',
                   'time']

    math_cuda = ['log10', 'acos', 'asin', 'atan', 'exp', 'sqrt', 'mpow', 'log', 'sin', 'cos', 'ceil', 'floor', 'tan',
                 time_symbol]

    for nam in range(len(math_python)):
        for k in range(len(model.kineticLaw)):
            if re.search(math_python[nam], model.kineticLaw[k]):
                s = model.kineticLaw[k]
                s = re.sub(math_python[nam], math_cuda[nam], s)
                model.kineticLaw[k] = s
    
        for k in range(len(model.ruleFormula)):
            if re.search(math_python[nam], model.ruleFormula[k]):
                s = model.ruleFormula[k]
                s = re.sub(math_python[nam], math_cuda[nam], s)
                model.ruleFormula[k] = s
    
        for k in range(len(model.eventFormula)):
            for cond in range(len(model.listOfAssignmentRules)):
                if re.search(math_python[nam], model.eventFormula[k][cond]):
                    s = model.eventFormula[k][cond]
                    s = re.sub(math_python[nam], math_cuda[nam], s)
                    model.eventFormula[k][cond] = s
    
        for k in range(len(model.eventCondition)):
            if re.search(math_python[nam], model.eventCondition[k]):
                s = model.eventCondition[k]
                s = re.sub(math_python[nam], math_cuda[nam], s)
                model.eventCondition[k] = s
    
        for k in range(len(model.functionBody)):
            if re.search(math_python[nam], model.functionBody[k]):
                s = model.functionBody[k]
                s = re.sub(math_python[nam], math_cuda[nam], s)
                model.functionBody[k] = s
    
        for fun in range(len(model.listOfFunctions)):
            for k in range(len(model.functionArgument[fun])):
                if re.search(math_python[nam], model.functionArgument[fun][k]):
                    s = model.functionArgument[fun][k]
                    s = re.sub(math_python[nam], math_cuda[nam], s)
                    model.functionArgument[fun][k] = s

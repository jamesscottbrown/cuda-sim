from cudasim.ParsedModel import ParsedModel
import re
import copy

class Writer:

    def __init__(self):
        pass

    # replace the species and parameters recursively
    @staticmethod
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

    def categorise_variables(self):
        # form a list of the species, and parameters which are set by rate rules
        model = self.parser.parsedModel

        rule_params = []
        rule_values = []
        constant_params = []
        constant_values = []

        for i in range(len(model.listOfParameter)):
            is_constant = True
            if not model.listOfParameter[i].getConstant():
                for k in range(len(model.listOfRules)):
                    if model.listOfRules[k].isRate() and model.ruleVariable[k] == model.parameterId[i]:
                        rule_params.append(model.parameterId[i])
                        rule_values.append(str(model.parameter[i]))
                        is_constant = False
            if is_constant:
                constant_params.append(model.parameterId[i])
                constant_values.append(str(model.parameter[i]))

        species_list = copy.copy(model.speciesId)
        species_list.extend(rule_params)

        species_values = map(lambda x: str(x), model.initValues)
        species_values.extend(rule_values)

        return species_list, constant_params, species_values, constant_values

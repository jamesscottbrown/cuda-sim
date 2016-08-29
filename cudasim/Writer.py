from ParsedModel import ParsedModel
import re

class Writer:
    def __init__(self, sbmlFileName, modelName="", inputPath="", outputPath=""):
        self.parsedModel = ParsedModel()

        if modelName == "":
            self.parsedModel.name = "unnamedModel"
        else:
            self.parsedModel.name = modelName

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

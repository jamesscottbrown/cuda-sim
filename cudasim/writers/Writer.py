from cudasim.ParsedModel import ParsedModel
import re


class Writer:

    def __init__(self):
        pass

    # replace the species and parameters recursively
    def rep(self, string, find, replace):
        ex = find + "[^0-9]"
        while re.search(ex, string) is not None:
            res = re.search(ex, string)
            string = string[0:res.start()] + replace + " " + string[res.end() - 1:]

        ex = find + "$"
        if re.search(ex, string) is not None:
            res = re.search(ex, string)
            string = string[0:res.start()] + replace + " " + string[res.end():]

        return string

    def mathMLConditionParserCuda(self, mathml_string):
        """
        Replaces and and or with and_ and or_ in a MathML string.
        Returns the string with and and or replaced by and_ and or_

        ***** args *****

        mathMLstring:

                A mathMLstring

        """

        and_string = re.compile("and")
        or_string = re.compile("or")
        mathml_string = and_string.sub("and_", mathml_string)
        mathml_string = or_string.sub("or_", mathml_string)

        return mathml_string

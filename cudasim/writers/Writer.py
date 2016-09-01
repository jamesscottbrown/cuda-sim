from cudasim.ParsedModel import ParsedModel
import re


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

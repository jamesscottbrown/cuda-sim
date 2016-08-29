import os
import re

from libsbml import *
from numpy import *

# TODO: edit this file to maintain the old interface
# By creating a wrapper function:
#  def importSBMLCUDA(source, integrationType, ModelName=None, method=None, useMoleculeCounts=False, outpath=""):


# To call the parser:
#    SBMLparse.importSBML(source, integrationType, ModelName=None,method=None)
#    All arguments to function must be passed as tuples.
#    If there is only one source to parse it must still be passed as a tuple ('source.xml',)
#    with an integrationType passed as ('Gillespie',)


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


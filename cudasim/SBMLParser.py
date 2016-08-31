# TODO: edit this file to maintain the old interface
# By creating a wrapper function:
#  def importSBMLCUDA(source, integrationType, ModelName=None, method=None, useMoleculeCounts=False, outpath=""):
# which would call ParseAndWrite()

# To call the parser:
#    SBMLparse.importSBML(source, integrationType, ModelName=None,method=None)
#    All arguments to function must be passed as tuples.
#    If there is only one source to parse it must still be passed as a tuple ('source.xml',)
#    with an integrationType passed as ('Gillespie',)

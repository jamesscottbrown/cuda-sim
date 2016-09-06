from cudasim.ParseAndWrite import parse_and_write


def importSBMLCUDA(source, integrationType, ModelName=None, method=None, outpath=""):
    parse_and_write(source, integrationType, ModelName, input_path="", output_path="", method="")


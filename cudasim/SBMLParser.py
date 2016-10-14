from cudasim.ParseAndWrite import parse_and_write


def importSBMLCUDA(source, integrationType, ModelName=None, method=None, outpath=""):
    integration_types = {'SDE': 'CUDA_SDE', 'ODE': 'CUDA_ODE', 'MJP': 'CUDA_Gillespie'}
    new_integration_type = map(lambda x: integration_types[x], integrationType)

    parse_and_write(source, new_integration_type, ModelName, input_path="", output_path=outpath,  method=method)

import numpy as np
import pycuda.driver as driver
from pycuda.compiler import SourceModule
import pycuda.autoinit

import Simulator as sim

class DelaySimulator(sim.Simulator):
    _param_tex = None
    _putIntoShared = False

    def __init__(self, timepoints, stepCode, delays, beta=1, dt=0.01, dump=False):
        self._delays = delays

        maxDelay = 10
        self._histTimeSteps = int(1 + maxDelay / (dt / 2))  # TODO: calculate this correctly

        sim.Simulator.__init__(self, timepoints, stepCode, beta=beta, dt=dt, dump=dump)

    def _compile(self, step_code):
        # TODO: determine if shared memory is enough to fit parameters ???

        delay_macro = """
        #define HISTLENGTH """ + str(self._histTimeSteps) + """
        #define delay(dimension, delayNum) yOld[dimension][delayNum - NCOMPARTMENTS]
        """

        general_parameters_source = """
        const int NRESULTS = """ + str(self._resultNumber) + """;
        const int BETA = """ + str(self._beta) + """;
        const float DT = """ + str(self._dt) + """f;
        const float TWOPI = 6.283185f;

        const int NSPECIES = """ + str(self._speciesNumber) + """;
        const int numDelays = """ + str(len(self._delays)) + """;

        //timepoints
        __device__ const float timepoints[""" + str(self._resultNumber) + "]={"

        for i in range(self._resultNumber):
            general_parameters_source += str(self._timepoints[i]) + 'f'
            if i != self._resultNumber - 1:
                general_parameters_source += ','

        general_parameters_source += '};'

        tMax = self._timepoints[-1]

        if not self._putIntoShared:
            general_parameters_source += """
        //parameter texture
        texture<float, 2, cudaReadModeElementType> param_tex;
        """

        solver_source = """
            __global__ void myDDEsolver(float *ics, float  *Y){
""" + "\tint tid = blockDim.x * blockIdx.x + threadIdx.x;\n" + "\tconst float tau[numDelays] = {" + ", ".join(
            self._delays) + "};\n" + """

                float maxDelay = 0;
                for (int i=0; i < numDelays; i++ ){
                    if (tau[i] > maxDelay ){
                        maxDelay = tau[i];
                    }
                }

                float tMax = """ + str(tMax) + """;


                // populate initial conditions
                float yHist[HISTLENGTH][NSPECIES];
                float yOld[NSPECIES];

                for (int j=0; j<NSPECIES; j++){
                    for (int i=0; i < HISTLENGTH; i++){
                        yHist[i][j] = ics[NSPECIES*tid + j];
                    }
                    yOld[j] = ics[NSPECIES*tid + j];
                }


                long nStart = 1 + maxDelay / (DT/2);

                // Set Y at t=0
                for (int n = 0; n < NSPECIES; n++){
                     Y[NSPECIES*tid*NRESULTS + 0 * NSPECIES + n] = yHist[ (HISTLENGTH-1)][ n ];
                }

                int kRs[numDelays];
                for (int i=0; i<numDelays; i++){
                    kRs[i] = tau[i] / DT;
                }

                float gam[5][NSPECIES][numDelays];
                float g[5][NSPECIES]; // so indexes stay 1-4, ignoring 0
                """

        array_declaration = "float kOld[%s][5][%s];\n" % (int(tMax / self._dt), self._speciesNumber)
        solver_source += array_declaration

        solver_source += """
                int lastSavedTimepoint = 1;
                for (int n = 0; n < (tMax / DT); n++){ // n is timestep

                    float t = n * DT;

                    for (int delayNum =0; delayNum < numDelays; delayNum ++) {

                        // loop over each variable
                        for (int dimension = 0; dimension < NSPECIES; dimension++) {

                            int kR = kRs[delayNum];
                            int s = maxDelay - tau[delayNum];

                            if ((t / DT) < (kR)) {
                                gam[1][dimension][delayNum] = yHist[(int) ( s + (t) / (DT / 2.0))][dimension];
                                gam[2][dimension][delayNum] = yHist[(int) ( s + (t + (DT / 2.0)) / (DT / 2.0))][dimension];
                                gam[3][dimension][delayNum] = yHist[(int) ( s + (t + (DT / 2.0)) / (DT / 2.0))][dimension];
                                gam[4][dimension][delayNum] = yHist[(int) ( s + (t + DT) / DT)][dimension];
                            } else {
                                gam[1][dimension][delayNum] = kOld[n - kR][1][dimension];
                                gam[2][dimension][delayNum] = kOld[n - kR][2][dimension];
                                gam[3][dimension][delayNum] = kOld[n - kR][3][dimension];
                                gam[4][dimension][delayNum] = kOld[n - kR][4][dimension];
                            }
                        }
                    }

                    for (int dimension = 0; dimension < NSPECIES; dimension++) {
                        g[1][dimension] = yOld[dimension];
                    }

                    for (int dimension = 0; dimension < NSPECIES; dimension++) {
                        g[2][dimension] = yOld[dimension] + (DT / 2) * f(t + DT / 2, (float *) &g[1], gam[1], dimension);
                    }
                    for (int dimension = 0; dimension < NSPECIES; dimension++) {
                        g[3][dimension] = yOld[dimension] + (DT / 2) * f(t + DT / 2, (float *) &g[2],  gam[2], dimension);
                    }
                    for (int dimension = 0; dimension < NSPECIES; dimension++) {
                        g[4][dimension] = yOld[dimension] + DT * f(t + DT, (float *) &g[3], gam[3], dimension);
                    }
                    for (int dimension = 0; dimension < NSPECIES; dimension++) {
                        float a = f(t, (float *) &g[1], gam[1], dimension);
                        float b = f(t + DT / 2, (float *) &g[2], gam[2], dimension);
                        float c = f(t + DT / 2, (float *) &g[3],  gam[3], dimension);
                        float d = f(t + DT, (float *) &g[4], gam[4], dimension);

                        yOld[dimension] = yOld[dimension] + (DT / 6) * (a + 2 * b + 2 * c + d);

                        for (int d = 0; d <= 5; d++) {
                            kOld[n][d][dimension] = g[d][dimension];
                        }

                    }

                     if ( (t + DT) >= timepoints[lastSavedTimepoint]){
                        for (int dimension = 0; dimension < NSPECIES; dimension++) {
                            Y[NSPECIES*tid*NRESULTS + lastSavedTimepoint * NSPECIES + dimension] = yOld[dimension];
                        }
                        lastSavedTimepoint++;
                    }


                }
            }
            """

        # actual compiling step compile
        completeCode = general_parameters_source + delay_macro + step_code + solver_source

        if self._dump:
            of = open("full_delay_code.cu", "w")
            print >> of, completeCode

        module = SourceModule(completeCode)

        if not self._putIntoShared:
            self._param_tex = module.get_texref("param_tex")

        return module, module.get_function('myDDEsolver')

    def _runSimulation(self, parameters, initValues, blocks, threads):
        print "In runSimulation"
        totalThreads = blocks * threads
        experiments = len(parameters)

        # simulation specific parameters
        param = np.zeros((totalThreads / self._beta + 1, self._parameterNumber), dtype=np.float32)
        try:
            for i in range(experiments):
                for j in range(self._parameterNumber):
                    param[i][j] = parameters[i][j]
        except IndexError:
            pass

        if not self._putIntoShared:
            # parameter texture
            ary = sim.create_2D_array(param)
            sim.copy2D_host_to_array(ary, param, self._parameterNumber * 4, totalThreads / self._beta + 1)
            self._param_tex.set_array(ary)
            sharedMemoryParameters = 0
        else:
            # parameter shared Mem
            sharedMemoryParameters = self._parameterNumber * (threads / self._beta + 2) * 4

        sharedTot = sharedMemoryParameters

        if self._putIntoShared:
            parametersInput = np.zeros(self._parameterNumber * totalThreads / self._beta, dtype=np.float32)
        speciesInput = np.zeros(self._speciesNumber * totalThreads, dtype=np.float32)
        result = np.zeros(self._speciesNumber * totalThreads * self._resultNumber, dtype=np.float32)

        # non coalesced
        try:
            for i in range(len(initValues)):
                for j in range(self._speciesNumber):
                    speciesInput[i * self._speciesNumber + j] = initValues[i][j]
        except IndexError:
            pass
        if self._putIntoShared:
            try:
                for i in range(experiments):
                    for j in range(self._parameterNumber):
                        parametersInput[i * self._parameterNumber + j] = parameters[i][j]
            except IndexError:
                pass

        species_gpu = driver.mem_alloc(speciesInput.nbytes)
        if self._putIntoShared:
            parameters_gpu = driver.mem_alloc(parametersInput.nbytes)
        result_gpu = driver.mem_alloc(result.nbytes)

        driver.memcpy_htod(species_gpu, speciesInput)
        if self._putIntoShared:
            driver.memcpy_htod(parameters_gpu, parametersInput)
        driver.memcpy_htod(result_gpu, result)

        # run code
        if self._putIntoShared:
            print "Putting in shared"
            self._compiledRunMethod(species_gpu, parameters_gpu, result_gpu, block=(threads, 1, 1),
                                    grid=(blocks, 1), shared=sharedTot)
        else:
            print "About to run", threads, blocks
            print "Number of threads is", totalThreads

            print "Species inputs: ", speciesInput
            self._compiledRunMethod(species_gpu, result_gpu, block=(threads, 1, 1), grid=(blocks, 1),
                                    shared=sharedTot)

        # fetch from GPU memory
        driver.memcpy_dtoh(result, result_gpu)

        print "Result:", type(result), result

        # reshape result
        result = result[0:experiments * self._beta * self._resultNumber * self._speciesNumber]
        result.shape = (experiments, self._beta, self._resultNumber, self._speciesNumber)

        return result

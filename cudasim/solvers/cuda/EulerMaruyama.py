import os
import string

import numpy as np
import pycuda
import pycuda.driver as driver
from pycuda.compiler import SourceModule
import cudasim
import cudasim.solvers.cuda.Simulator as sim


class EulerMaruyama(sim.Simulator):
    _param_tex = None
    _putIntoShared = False

    # general RNG parameters (fixed)
    _warp_size = 32
    _state_words = 32

    def _compile(self, step_code):

        # determine if shared memory is enough to fit parameters 
        # max_threads = self._maxThreadsPerMP
        # since max_threads is set to 64, maximally 512 threads can be on one MP
        max_threads = 512

        total_shared_memory = driver.Device(self._device).max_shared_memory_per_block

        # 32 words/warp for RNG * maximum number of warps/block * 4 (bytes/word)
        free_shared_memory = total_shared_memory - max_threads / self._warp_size * self._state_words * 4

        # assuming maximum number of threads (should be per MP)
        max_parameters = free_shared_memory / max_threads / 4

        # remove the shared part
        # if self._parameterNumber <= self._beta * max_parameters:
        #    self._putIntoShared = True

        # print "cuda-sim: Euler-Maruyama : Using shared memory code: " + str(self._putIntoShared)
        step_code = self._modify_step_code(step_code, self._putIntoShared)

        general_parameters_source = """
    const int NRESULTS = """ + str(self._resultNumber) + """;
    
    const int BETA = """ + str(self._beta) + """;
    const float DT = """ + str(self._dt) + """f;
    const float TWOPI = 6.283185f;
    
    //timepoints
    __device__ const float timepoints[""" + str(self._resultNumber) + "]={"

        for i in range(self._resultNumber):
            general_parameters_source += str(self._timepoints[i]) + 'f'
            if i != self._resultNumber - 1:
                general_parameters_source += ','

        general_parameters_source += '};'

        if not self._putIntoShared:
            general_parameters_source += """
    //parameter texture
    texture<float, 2, cudaReadModeElementType> param_tex;"""

        rng_source = """
    
    //gernerate normal distributed random numbers
    //DO NOT EVER try to call this after intra-warp divergence!!
    __device__ float randNormal(unsigned *regs, float deviation){
        
        float uniformRdn1 = WarpStandard_Generate(regs)/4294967295.0f;
        float uniformRdn2 = WarpStandard_Generate(regs)/4294967295.0f;
        
        // Box-Muller transformation to N(mean=0,var=DT)
        float returnval = 0.0f;
        
        returnval =  deviation * sqrtf( -2.0f*__logf(uniformRdn1) ) * __cosf(TWOPI*uniformRdn2);
        
        return returnval;
    }
    
    // shared memory allocation for RNG
    extern __shared__ unsigned WarpStandard_shmem[];
    
    """

        neg_eq_zero_source = """
    __device__ float negEqZero(float number){
        if(number < 0)
            return 0;
        else
            return number;
    }
    """

        if self._putIntoShared:
            sde_src_rest = """
    __global__ void sdeMain(float *species, float *parameters, unsigned *seed, float *result){"""
        else:
            sde_src_rest = """
    __global__ void sdeMain(float *species, unsigned *seed, float *result){"""

        sde_src_rest += """
        //initialize RNG
        unsigned rngRegs[WarpStandard_REG_COUNT];
        WarpStandard_LoadState(seed, rngRegs);
        
        //set threadID for saving species update
        int tid = blockIdx.x*blockDim.x + threadIdx.x;
        
        //initialize species
        float y[NSPECIES];
        
        for(int i = 0; i<NSPECIES; i++){
            //non-coalesced
            y[i] = species[NSPECIES*tid + i];
        }
        
        //initialize parameters
        
        """

        if self._putIntoShared:
            sde_src_rest += """
        //offset for RNG (= number of threads)
        // beta = # of multiple use of the same parameters
        // blockDim.x is the shared memory occupied by the RNG
        
        int shMemIndex = blockDim.x + (tid/BETA - blockIdx.x *blockDim.x/BETA)*NPARAM;
        float *parameter = (float*)&WarpStandard_shmem[shMemIndex];
        
        for(int i = 0; i<NPARAM; i++){
            //non-coalesced
            parameter[i] = parameters[NPARAM*(tid/BETA) + i];
        }"""

        else:
            sde_src_rest += """
        int texMemIndex = tid/BETA;"""

        sde_src_rest += """
        float t = 0;
        
        //repeat number of dt dts
        for(int i = 0; i<NRESULTS;){
            //write results every outputEvery steps
            if(t >= timepoints[i]){
                for(int j = 0; j<NSPECIES; j++){
                    //non-coalesced
                    result[NSPECIES*tid*NRESULTS + i * NSPECIES + j] = y[j];
                    }
                i++;
            }
            t += DT;
            """

        if self._putIntoShared:
            sde_src_rest += "step(parameter, y, t, rngRegs);"
        else:
            sde_src_rest += "step(y, t, rngRegs, texMemIndex);"

        sde_src_rest += """
            //set values to zero if they are negative"""

        for i in range(self._speciesNumber):
            sde_src_rest += """
            y[""" + str(i) + """] = negEqZero(y[""" + str(i) + """]);"""
        sde_src_rest += """
            }
        }"""

        # external rng
        cuda = os.path.join(os.path.split(os.path.realpath(cudasim.__file__))[0], 'WarpStandard.cu')
        f = open(cuda, 'r')
        rng_ext = f.read()
        f.close()

        # actual compiling step compile
        complete_code = general_parameters_source + rng_ext + rng_source + neg_eq_zero_source + step_code + sde_src_rest

        if self._dump:
            of = open("full_sde_code.cu", "w")
            print >> of, complete_code

        print "DEFAULT_NVCC_FLAGS:", pycuda.compiler.DEFAULT_NVCC_FLAGS
        module = SourceModule(complete_code)

        if not self._putIntoShared:
            self._param_tex = module.get_texref("param_tex")

        return module, module.get_function('sdeMain')

    def _run_simulation(self, parameters, init_values, blocks, threads):

        total_threads = blocks * threads
        experiments = len(parameters)

        # simulation specific parameters
        param = np.zeros((total_threads / self._beta + 1, self._parameterNumber), dtype=np.float32)
        try:
            for i in range(experiments):
                for j in range(self._parameterNumber):
                    param[i][j] = parameters[i][j]
        except IndexError:
            pass

        if not self._putIntoShared:
            # parameter texture
            ary = sim.create_2d_array(param)
            sim.copy_2d_host_to_array(ary, param, self._parameterNumber * 4, total_threads / self._beta + 1)
            self._param_tex.set_array(ary)
            shared_memory_parameters = 0
        else:
            # parameter shared Mem
            shared_memory_parameters = self._parameterNumber * (threads / self._beta + 2) * 4

        shared_memory_per_block_for_rng = threads / self._warp_size * self._state_words * 4
        shared_tot = shared_memory_per_block_for_rng + shared_memory_parameters

        if self._putIntoShared:
            parameters_input = np.zeros(self._parameterNumber * total_threads / self._beta, dtype=np.float32)
        species_input = np.zeros(self._speciesNumber * total_threads, dtype=np.float32)
        result = np.zeros(self._speciesNumber * total_threads * self._resultNumber, dtype=np.float32)

        # non coalesced
        try:
            for i in range(len(init_values)):
                for j in range(self._speciesNumber):
                    species_input[i * self._speciesNumber + j] = init_values[i][j]
        except IndexError:
            pass
        if self._putIntoShared:
            try:
                for i in range(experiments):
                    for j in range(self._parameterNumber):
                        parameters_input[i * self._parameterNumber + j] = parameters[i][j]
            except IndexError:
                pass

        # set seeds using python rng
        seeds = np.zeros(total_threads / self._warp_size * self._state_words, dtype=np.uint32)
        for i in range(len(seeds)):
            seeds[i] = np.uint32(4294967296 * np.random.uniform(0, 1))
            # seeds[i] =  np.random.random_integers(0,4294967295)

        species_gpu = driver.mem_alloc(species_input.nbytes)
        if self._putIntoShared:
            parameters_gpu = driver.mem_alloc(parameters_input.nbytes)
        seeds_gpu = driver.mem_alloc(seeds.nbytes)
        result_gpu = driver.mem_alloc(result.nbytes)

        driver.memcpy_htod(species_gpu, species_input)
        if self._putIntoShared:
            driver.memcpy_htod(parameters_gpu, parameters_input)
        driver.memcpy_htod(seeds_gpu, seeds)
        driver.memcpy_htod(result_gpu, result)

        # run code
        if self._putIntoShared:
            self._compiledRunMethod(species_gpu, parameters_gpu, seeds_gpu, result_gpu, block=(threads, 1, 1),
                                    grid=(blocks, 1), shared=shared_tot)
        else:
            self._compiledRunMethod(species_gpu, seeds_gpu, result_gpu, block=(threads, 1, 1), grid=(blocks, 1),
                                    shared=shared_tot)

        # fetch from GPU memory
        driver.memcpy_dtoh(result, result_gpu)

        # reshape result
        result = result[0:experiments * self._beta * self._resultNumber * self._speciesNumber]
        result.shape = (experiments, self._beta, self._resultNumber, self._speciesNumber)

        return result

    # comments out the part of the step code that is not used (either parameters in texture or shared memory)
    def _modify_step_code(self, step_code, put_into_shared):
        lines = str(step_code).split("\n")

        comment = False

        for i in range(len(lines)):
            if lines[i].find("//Code for shared memory") != -1:
                if put_into_shared:
                    comment = False
                else:
                    comment = True
            elif lines[i].find("//Code for texture memory") != -1:
                if put_into_shared:
                    comment = True
                else:
                    comment = False
            else:
                # comment out
                if comment:
                    lines[i] = "// " + lines[i]

        return string.join(lines, "\n")

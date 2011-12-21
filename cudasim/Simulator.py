import numpy as np
import time, math,os

import pycuda.tools as tools
import pycuda.driver as driver

class Simulator:

    # constant private variables
    _MAXBLOCKSPERDEVICE = 500
    _MAXTHREADSPERBLOCK = 64

    # private variables
    _compiledRunMethod = None
    _completeCode = None
    
    _beta = None
    _timepoints = None
    _dt = None
    
    _stepCode = None
    _runtimeCompile = False
    
    _parameterNumber = None
    _speciesNumber = None
    _hazardNumber = None
    _resultNumber = None
    
    #_seedValue = None
    
    # device used
    # ToDo enable more than default device to be used
    _device = None
    _maxThreadsPerMP = None
    _maxBlocksPerMP = None

    _dump = False
    
    def __init__(self, timepoints, stepCode, beta=1, dt=0.01):
        # only beta which are factors of _MAXBLOCKSPERDEVICE are accepted, 
        # else _MAXBLOCKSPERDEVICE is reduced to the next smallest acceptable value
        if(self._MAXBLOCKSPERDEVICE%int(beta) != 0):
            self._MAXBLOCKSPERDEVICE -= self._MAXBLOCKSPERDEVICE%int(beta)
            print "Beta must be a factor of", self._MAXBLOCKSPERDEVICE
            print "_MAXBLOCKSPERDEVICE reduced to", self._MAXBLOCKSPERDEVICE
            
        # read stepCode from file
        stepCode = open(stepCode,'r').read()
        
        self._beta = int(beta)
        self._timepoints = np.array(timepoints,dtype=np.float32)
        self._dt = float(dt)
        
        device = os.getenv("CUDA_DEVICE")
        if(device==None):
            self._device = 0
        else:
            self._device = int(device)
        ## print "cuda-sim: Using device", self._device
        
        
        self._getKernelParams(stepCode)
        
        self._resultNumber = len(timepoints)
            
        compability = driver.Device(self._device).compute_capability()
        self._maxThreadsPerMP =  getMaxThreadsPerMP(compability)
        self._maxBlocksPerMP = getMaxBlocksPerMP(compability)
        
        if(not self._runtimeCompile):
            self._completeCode, self._compiledRunMethod = self._compile(stepCode)
        else:
            self._stepCode = stepCode
            
    
    ############ private methods ############
    
    # method for generating seeds for random number generators
    #def _seed(self):
    #    return self._seedValue
    
    # method for extracting the number of species, variables and reactions from CUDA kernel
    def _getKernelParams(self, stepCode):
        lines = str(stepCode).split("\n")
        for i in lines:
            if(i.find("NSPECIES") != -1):
                self._speciesNumber = int(i.split("NSPECIES")[1])
            elif(i.find("NPARAM") != -1):
                self._parameterNumber = int(i.split("NPARAM")[1])
            elif(i.find("NREACT") != -1):
                self._hazardNumber = int(i.split("NREACT")[1])
    
    
    # method for calculating optimal number of blocks and threads per block
    def _getOptimalGPUParam(self, parameters, compiledRunMethod = None):
        if compiledRunMethod == None:
            compiledRunMethod = self._compiledRunMethod
        
        # general parameters
        maxThreadsPerBlock = driver.Device(self._device).max_threads_per_block
        warp_size = 32
        
        # calculate number of threads per block; assuming that registers are the limiting factor
        #maxThreads = min(driver.Device(self._device).max_registers_per_block/compiledRunMethod.num_regs,maxThreadsPerBlock)
        
        # assume smaller blocksize creates less overhead; ignore occupancy..
        maxThreads = min(driver.Device(self._device).max_registers_per_block/compiledRunMethod.num_regs, self._MAXTHREADSPERBLOCK)
        
        maxWarps = maxThreads / warp_size
        # warp granularity up to compability 2.0 is 2. Therefore if maxWarps is uneven only maxWarps-1 warps
        # can be run
        #if(maxWarps % 2 == 1):
            #maxWarps -= 1
        
        # maximum number of threads per block
        threads = maxWarps * warp_size
        
        # assign number of blocks
        if (len(parameters)*self._beta%threads == 0):
            blocks = len(parameters)*self._beta/threads
        else:
            blocks = len(parameters)*self._beta/threads + 1
        
        return blocks, threads
    
    # ABSTRACT
    # method for compiling, given some C code for every simulation step
    def _compile(self, step_code):
        return None
    
    # ABSTRACT
    # method for compiling at runtime, given some C code for every simulation step
    def _compileAtRuntime(self, step_code, parameters):
        return None
    
    
    # ABSTRACT
    # method for running simulation
    def _runSimulation(self, parameters, initValues, blocks, threads):
        return None
    
    
    
    ############ public methods ############
    
    # specify GPU specific variables and _runSimulation()
    def run(self, parameters, initValues, timing=True, info=False):
        
        #check parameters and initValues for compability with pre-defined parameterNumber and spieciesNumber
        if(len(parameters[0]) != self._parameterNumber):
            print "Error: Number of parameters specified (" + str(self._parameterNumber) + ") and given in parameter array (" + str(len(parameters[0])) + ") differ from each other!"
            exit()
        elif(len(initValues[0]) != self._speciesNumber):
            print "Error: Number of species specified (" +  str(self._speciesNumber) + ") and given in species array (" + str(len(initValues[0])) + ") differ from each other!"
            exit()
        elif(len(parameters) != len(initValues)):
            print "Error: Number of sets of parameters (" + str(len(parameters)) + ") and species (" + str(len(initValues)) + ") do not match!"
            exit()
        
        if(self._compiledRunMethod == None and self._runtimeCompile):
            #compile to determine blocks and threads
            self._completeCode, self._compiledRunMethod = self._compileAtRuntime(self._stepCode, parameters)
        
        blocks, threads = self._getOptimalGPUParam(parameters)
        if info==True:
            print "cuda-sim: threads/blocks:", threads, blocks

        # real runtime compile
        
        #self._seedValue = seed
        #np.random.seed(self._seedValue)

        # make multiples of initValues
        initNew = np.zeros((len(initValues)*self._beta,self._speciesNumber))
        for i in range(len(initValues)):
            for j in range(self._beta):
                for k in range(self._speciesNumber):
                    initNew[i*self._beta + j][k] = initValues[i][k]
        initValues = initNew
        
        if info==True:
            print "cuda-sim: kernel mem local / shared / registers : ", self._compiledRunMethod.local_size_bytes, self._compiledRunMethod.shared_size_bytes, self._compiledRunMethod.num_regs
            occ = tools.OccupancyRecord( tools.DeviceData(), threads=threads, shared_mem=self._compiledRunMethod.shared_size_bytes, registers=self._compiledRunMethod.num_regs )
            print "cuda-sim: threadblocks per mp / limit / occupancy :", occ.tb_per_mp, occ.limited_by, occ.occupancy


        if timing:
            start = time.time()
        
        # number of device calls
        runs = int(math.ceil(blocks / float(self._MAXBLOCKSPERDEVICE)))
        for i in range(runs):
            # for last device call calculate number of remaining threads to run
            if(i==runs-1):
                runblocks = int(blocks % self._MAXBLOCKSPERDEVICE)
                if(runblocks == 0):
                    runblocks = self._MAXBLOCKSPERDEVICE
            else:
                runblocks = int(self._MAXBLOCKSPERDEVICE)

            if info==True:
                print "cuda-sim: Run", runblocks, "blocks."

            minIndex = self._MAXBLOCKSPERDEVICE*i*threads
            maxIndex = minIndex + threads*runblocks
            runParameters = parameters[minIndex/self._beta:maxIndex/self._beta]
            runInitValues = initValues[minIndex:maxIndex]
            
            #first run store return Value
            if(i==0):
                returnValue = self._runSimulation(runParameters, runInitValues, runblocks, threads)
            else:
                returnValue = np.append(returnValue,self._runSimulation(runParameters, runInitValues, runblocks, threads),axis=0)
        
        if timing:
            print "cuda-sim: GPU blocks / threads / running time:", threads, blocks, round((time.time()-start),4), "s"

        if info:
            print ""
        
        return returnValue
    
# static non-class methods
def copy2D_host_to_device(dev, host, src_pitch, dst_pitch, width, height ):
    copy = driver.Memcpy2D()
    copy.set_src_host(host)
    copy.set_dst_device(dev)
    copy.src_pitch = src_pitch
    copy.dst_pitch = dst_pitch
    copy.width_in_bytes = width
    copy.height = height
    copy(aligned=True)

def copy2D_device_to_host(host, dev, src_pitch, dst_pitch, width, height ):
    copy = driver.Memcpy2D()
    copy.set_src_device(dev)
    copy.set_dst_host(host)
    copy.src_pitch = src_pitch
    copy.dst_pitch = dst_pitch
    copy.width_in_bytes = width
    copy.height = height
    copy(aligned=True)

# Create a 2D GPU array (for assignment
# to texture) from a numpy 2D array
def create_2D_array( mat ):
    descr = driver.ArrayDescriptor()
    descr.width = mat.shape[1]
    descr.height = mat.shape[0]
    descr.format = driver.dtype_to_array_format( mat.dtype )
    descr.num_channels = 1
    descr.flags = 0
    ary = driver.Array(descr)
    return ary

# Copy 2D host numpy array to 2D
# GPU array object
def copy2D_host_to_array(arr, host, width, height ):
    copy = driver.Memcpy2D()
    copy.set_src_host(host)
    copy.set_dst_array(arr)
    copy.height = height
    copy.width_in_bytes = copy.src_pitch = width
    copy.height = height
    copy(aligned=True)
    
# Determine thread granularity
#def getRegisterGranularity(compabilityTuple):
    #if(compabilityTuple[0] == 1):
        #if(compabilityTuple[1] == 0 or compabilityTuple[1] == 1):
            #return 256
        #elif(compabilityTuple[1] == 2 or compabilityTuple[1] == 3):
            #return 512
    #elif(compabilityTuple[0] == 2)
        #if(compabilityTuple[1] == 0):
            #return 
    #return 512
    
# Determine maximum number of threads per MP
def getMaxThreadsPerMP(compabilityTuple):
    if(compabilityTuple[0] == 1):
        if(compabilityTuple[1] == 0 or compabilityTuple[1] == 1):
            return 768
        elif(compabilityTuple[1] == 2 or compabilityTuple[1] == 3):
            return 1024
    elif(compabilityTuple[0] == 2):
        if(compabilityTuple[1] == 0):
            return 1536
    return 768

# Determine maximum number of blocks per MP
def getMaxBlocksPerMP(compabilityTuple):
    return 8
    
    
    
    
    

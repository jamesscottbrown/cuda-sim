import copy
import math
import time

import numpy as np
# import threading
import multiprocessing

import pycuda.tools as tools
import pycuda.driver as driver


# class Simulator_mg(threading.Thread):
class Simulator_mg(multiprocessing.Process):
    # constant private variables
    _MAXBLOCKSPERDEVICE = 500
    _MAXTHREADSPERBLOCK = 64

    # compiler variables
    _compiledRunMethod = None  # this is a pointer to the __global__ kernel function
    _completeCode = None  # this is a pointer to the complete compiled cuda kernel
    _stepCode = None  # this contains the code generated for this particular model
    _runtimeCompile = False  # this flag is set to true for Lsoda

    # hardware variables
    _context = None
    _device = None

    def __init__(self, timepoints, stepCodeFile, parameters, initValues, output_cpu, card=-1, beta=1, dt=0.01,
                 timing=True, info=False, dump=False):
        # threading.Thread.__init__(self)
        multiprocessing.Process.__init__(self)

        self._card = card

        self._timepoints = np.array(timepoints, dtype=np.float32)
        self._resultNumber = len(timepoints)
        self._beta = int(beta)

        self._parameters = copy.deepcopy(parameters)
        self._initValues = copy.deepcopy(initValues)

        self.output_cpu = output_cpu

        self._dt = float(dt)
        self._dump = dump
        self._info = info
        self._timing = timing

        # read stepCode from file
        self._stepCode = open(stepCodeFile, 'r').read()

        # Do some error checking
        self._getKernelParams()
        self._check_consistency()

        # only beta which are factors of _MAXBLOCKSPERDEVICE are accepted, 
        # else _MAXBLOCKSPERDEVICE is reduced to the next smallest acceptable value
        if self._MAXBLOCKSPERDEVICE % int(beta) != 0:
            self._MAXBLOCKSPERDEVICE -= self._MAXBLOCKSPERDEVICE % int(beta)
            print "Beta must be a factor of", self._MAXBLOCKSPERDEVICE
            print "_MAXBLOCKSPERDEVICE reduced to", self._MAXBLOCKSPERDEVICE

    ############ private methods ############

    # method for extracting the number of species, variables and reactions from CUDA kernel
    def _getKernelParams(self):
        lines = str(self._stepCode).split("\n")
        for i in lines:
            if i.find("NSPECIES") != -1:
                self._speciesNumber = int(i.split("NSPECIES")[1])
            elif i.find("NPARAM") != -1:
                self._parameterNumber = int(i.split("NPARAM")[1])
            elif i.find("NREACT") != -1:
                self._hazardNumber = int(i.split("NREACT")[1])

    # check parameters and initValues for compability with pre-defined parameterNumber and spieciesNumber
    def _check_consistency(self):
        if len(self._parameters[0]) != self._parameterNumber:
            print "Error: Number of parameters specified (" + str(
                self._parameterNumber) + ") and given in parameter array (" + str(
                len(self._parameters[0])) + ") differ from each other!"
            exit()
        elif len(self._initValues[0]) != self._speciesNumber:
            print "Error: Number of species specified (" + str(
                self._speciesNumber) + ") and given in species array (" + str(
                len(self._initValues[0])) + ") differ from each other!"
            exit()
        elif len(self._parameters) != len(self._initValues):
            print "Error: Number of sets of parameters (" + str(len(self._parameters)) + ") and species (" + str(
                len(self._initValues)) + ") do not match!"
            exit()

    # method for calculating optimal number of blocks and threads per block
    def _getOptimalGPUParam(self, compiledRunMethod=None):
        if compiledRunMethod is None:
            compiledRunMethod = self._compiledRunMethod

        # general parameters
        maxThreadsPerBlock = self._context.get_device().max_threads_per_block
        warp_size = 32

        # calculate number of threads per block; assuming that registers are the limiting factor
        # maxThreads = min(driver.Device(self._device).max_registers_per_block/compiledRunMethod.num_regs,maxThreadsPerBlock)

        # assume smaller blocksize creates less overhead; ignore occupancy..
        maxThreads = min(self._context.get_device().max_registers_per_block / compiledRunMethod.num_regs,
                         self._MAXTHREADSPERBLOCK)

        maxWarps = maxThreads / warp_size
        # warp granularity up to compability 2.0 is 2. Therefore if maxWarps is uneven only maxWarps-1 warps
        # can be run
        # if(maxWarps % 2 == 1):
        # maxWarps -= 1

        # maximum number of threads per block
        threads = maxWarps * warp_size

        # assign number of blocks : len(self._parameters) is the number of threads
        if len(self._parameters) * self._beta % threads == 0:
            blocks = len(self._parameters) * self._beta / threads
        else:
            blocks = len(self._parameters) * self._beta / threads + 1

        return blocks, threads

    # ABSTRACT
    # method for compiling, given some C code for every simulation step
    def _compile(self, step_code):
        return None

    # ABSTRACT
    # method for running simulation
    def _runSimulation(self, parameters, initValues, blocks, threads):
        return None

    # ABSTRACT
    # method for object deletion

    ############ public methods ############

    def join(self):
        # self._context.detach()
        # threading.Thread.join(self)
        multiprocessing.Process.join(self)

    # specify GPU specific variables and _runSimulation()
    def run(self):

        # obtain a CUDA context
        driver.init()
        if self._card < 0:
            self._context = tools.make_default_context()
        else:
            self._context = driver.Device(self._card).make_context()

        if self._info:
            print "cuda-sim: running on device ", self._card, self._context.get_device().name(), self._context.get_device().pci_bus_id()

        # hack for SDE code
        self._device = 0

        # compile code
        self._completeCode, self._compiledRunMethod = self._compile(self._stepCode)

        blocks, threads = self._getOptimalGPUParam()
        if self._info:
            print "cuda-sim: threads/blocks:", threads, blocks

        # make multiples of initValues incase beta > 1
        initNew = np.zeros((len(self._initValues) * self._beta, self._speciesNumber))
        for i in range(len(self._initValues)):
            for j in range(self._beta):
                for k in range(self._speciesNumber):
                    initNew[i * self._beta + j][k] = self._initValues[i][k]
        self._initValues = copy.deepcopy(initNew)

        if self._info:
            print "cuda-sim: kernel mem local / shared / registers : ", self._compiledRunMethod.local_size_bytes, self._compiledRunMethod.shared_size_bytes, self._compiledRunMethod.num_regs
            occ = tools.OccupancyRecord(tools.DeviceData(), threads=threads,
                                        shared_mem=self._compiledRunMethod.shared_size_bytes,
                                        registers=self._compiledRunMethod.num_regs)
            print "cuda-sim: threadblocks per mp / limit / occupancy :", occ.tb_per_mp, occ.limited_by, occ.occupancy

        if self._timing:
            start = time.time()

        # number of device calls
        runs = int(math.ceil(blocks / float(self._MAXBLOCKSPERDEVICE)))
        for i in range(runs):
            # for last device call calculate number of remaining threads to run
            if i == runs - 1:
                runblocks = int(blocks % self._MAXBLOCKSPERDEVICE)
                if runblocks == 0:
                    runblocks = self._MAXBLOCKSPERDEVICE
            else:
                runblocks = int(self._MAXBLOCKSPERDEVICE)

            if self._info:
                print "cuda-sim: Run", runblocks, "blocks."

            minIndex = self._MAXBLOCKSPERDEVICE * i * threads
            maxIndex = minIndex + threads * runblocks
            runParameters = self._parameters[minIndex / self._beta:maxIndex / self._beta]
            runInitValues = self._initValues[minIndex:maxIndex]

            # first run store return Value
            if i == 0:
                self._returnValue = self._runSimulation(runParameters, runInitValues, runblocks, threads)
            else:
                self._returnValue = np.append(self._returnValue,
                                              self._runSimulation(runParameters, runInitValues, runblocks, threads),
                                              axis=0)

        self.output_cpu.put([self._card, self._returnValue])
        self.output_cpu.close()

        # if self._timing:
        #    print "cuda-sim: GPU blocks / threads / running time:", threads, blocks, round((time.time()-start),4), "s"

        if self._info:
            print ""

        # return the context
        self._context.pop()
        del self._context

        return self._returnValue


# static non-class methods
def copy2D_host_to_device(dev, host, src_pitch, dst_pitch, width, height):
    c = driver.Memcpy2D()
    c.set_src_host(host)
    c.set_dst_device(dev)
    c.src_pitch = src_pitch
    c.dst_pitch = dst_pitch
    c.width_in_bytes = width
    c.height = height
    c(aligned=True)


def copy2D_device_to_host(host, dev, src_pitch, dst_pitch, width, height):
    c = driver.Memcpy2D()
    c.set_src_device(dev)
    c.set_dst_host(host)
    c.src_pitch = src_pitch
    c.dst_pitch = dst_pitch
    c.width_in_bytes = width
    c.height = height
    c(aligned=True)


# Create a 2D GPU array (for assignment
# to texture) from a numpy 2D array
def create_2D_array(mat):
    descr = driver.ArrayDescriptor()
    descr.width = mat.shape[1]
    descr.height = mat.shape[0]
    descr.format = driver.dtype_to_array_format(mat.dtype)
    descr.num_channels = 1
    descr.flags = 0
    ary = driver.Array(descr)
    return ary


# Copy 2D host numpy array to 2D
# GPU array object
def copy2D_host_to_array(arr, host, width, height):
    copy = driver.Memcpy2D()
    copy.set_src_host(host)
    copy.set_dst_array(arr)
    copy.height = height
    copy.width_in_bytes = copy.src_pitch = width
    copy.height = height
    copy(aligned=True)


# Determine maximum number of threads per MP
def getMaxThreadsPerMP(compabilityTuple):
    if compabilityTuple[0] == 1:
        if compabilityTuple[1] == 0 or compabilityTuple[1] == 1:
            return 768
        elif compabilityTuple[1] == 2 or compabilityTuple[1] == 3:
            return 1024
    elif compabilityTuple[0] == 2:
        if compabilityTuple[1] == 0:
            return 1536
    return 768


# Determine maximum number of blocks per MP
def getMaxBlocksPerMP(compabilityTuple):
    return 8

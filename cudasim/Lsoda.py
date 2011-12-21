# cuda utilities

import Simulator as sim

import os, time

import pycuda.driver as driver
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
from struct import unpack

class Lsoda(sim.Simulator):

    
    _param_tex = None
    
    _step_code = None
    _runtimeCompile = True
    
    _lsoda_source_ = """
    
    extern "C"{
    
    __device__ myFex myfex;
    __device__ myJex myjex;
    
    __global__ void init_common(){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    cuLsodaCommonBlockInit( &(common[tid]) );
    }
    
    __global__ void cuLsoda(int *neq, double *y, double *t, double *tout, int *itol, 
                double *rtol, double *atol, int *itask, int *istate, int *iopt, 
                            double *rwork, int *lrw, int *iwork, int *liw, int *jt)
    {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    dlsoda_(myfex, neq+tid, y+tid*NSPECIES, t+tid, tout+tid, itol+tid, rtol+tid, atol+tid, itask+tid, 
        istate+tid, iopt+tid, rwork+tid*RSIZE, lrw+tid, iwork+tid*ISIZE, liw+tid, myjex, jt+tid, &(common[tid]) );
    }
    }
    
    """
    
    
    def _compileAtRuntime(self, step_code, parameters):
        # set beta to 1 - deterministic!!
        self._beta = 1
        
        fc = open( os.path.join(os.path.split(os.path.realpath(__file__))[0],'cuLsoda_all.cu'),'r')
        _sourceFromFile_ = fc.read()
        
        # Options?
        options = []
        
        # Write Code?
        write = False
    
        isize = 20 + self._speciesNumber
        rsize = 22 + self._speciesNumber * max(16, self._speciesNumber + 9)
    
        _isize_ = "#define ISIZE " + repr( 20 + self._speciesNumber ) + "\n"
        _rsize_ = "#define RSIZE " + repr( 22 + self._speciesNumber * max(16, self._speciesNumber + 9) ) + "\n"
    
        _textures_ = "texture<float, 2, cudaReadModeElementType> param_tex;\n"
        _common_block_ = "__device__ struct cuLsodaCommonBlock common[" + repr(1*1) + "];\n"
        _code_ =  _isize_ + _rsize_ + _textures_ + step_code + _sourceFromFile_ + _common_block_ + self._lsoda_source_
        
        
        # dummy compile to determine optimal blockSize and gridSize
        compiled = pycuda.compiler.SourceModule( _code_, nvcc="nvcc", options=options, no_extern_c=True )
        
        blocks, threads = self._getOptimalGPUParam(parameters, compiled.get_function("cuLsoda"))
        blocks = self._MAXBLOCKSPERDEVICE
        
        # real compile
        _common_block_ = "__device__ struct cuLsodaCommonBlock common[" + repr(blocks*threads) + "];\n"
        _code_ =  _isize_ + _rsize_ + _textures_ + step_code + _sourceFromFile_ + _common_block_ + self._lsoda_source_
        
        compiled = pycuda.compiler.SourceModule( _code_, nvcc="nvcc", options=options, no_extern_c=True)
        
        self._param_tex = compiled.get_texref("param_tex")
        
        lsoda_Kernel = compiled.get_function("cuLsoda")
        return (compiled, lsoda_Kernel)
        
            
    def _runSimulation(self, parameters, initValues, blocks, threads, in_atol=1e-12,in_rtol=1e-6 ):
        
        totalThreads = threads * blocks
        experiments = len(parameters)
        
        neqn = self._speciesNumber
        
        # compile 
        timer = time.time()
        ## print "Init Common..",
        init_common_Kernel = self._completeCode.get_function("init_common")
        init_common_Kernel( block=(threads,1,1), grid=(blocks,1) )
        ## print "finished in", round(time.time()-timer,4), "s"
        
        start_time = time.time()
        # output array
        ret_xt = np.zeros( [totalThreads, 1, self._resultNumber, self._speciesNumber] )
    
        # calculate sizes of work spaces
        isize = 20 + self._speciesNumber
        rsize = 22 + self._speciesNumber * max(16, self._speciesNumber + 9)
            
        # local variables
        t      = np.zeros( [totalThreads], dtype=np.float64)
        jt     = np.zeros( [totalThreads], dtype=np.int32)
        neq    = np.zeros( [totalThreads], dtype=np.int32)
        itol   = np.zeros( [totalThreads], dtype=np.int32)
        iopt   = np.zeros( [totalThreads], dtype=np.int32)
        rtol   = np.zeros( [totalThreads], dtype=np.float64)
        iout   = np.zeros( [totalThreads], dtype=np.int32)
        tout   = np.zeros( [totalThreads], dtype=np.float64)
        itask  = np.zeros( [totalThreads], dtype=np.int32)
        istate = np.zeros( [totalThreads], dtype=np.int32)
        atol   = np.zeros( [totalThreads], dtype=np.float64)
    
        liw    = np.zeros( [totalThreads], dtype=np.int32)
        lrw    = np.zeros( [totalThreads], dtype=np.int32)
        iwork  = np.zeros( [isize*totalThreads], dtype=np.int32)
        rwork  = np.zeros( [rsize*totalThreads], dtype=np.float64)
        y      = np.zeros( [self._speciesNumber*totalThreads], dtype=np.float64)
        
        
        for i in range(totalThreads):
            neq[i] = neqn
            #t[i] = self._timepoints[0]
            t[i] = 0
            itol[i] = 1
            itask[i] = 1
            istate[i] = 1
            iopt[i] = 0
            jt[i] = 2
            atol[i] = in_atol
            rtol[i] = in_rtol
    
            liw[i] = isize
            lrw[i] = rsize
    
            try:
            # initial conditions
                for j in range(self._speciesNumber):
                    # loop over species
                    y[i*self._speciesNumber + j] = initValues[i][j]
                    ret_xt[i, 0, 0, j] = initValues[i][j]
            except IndexError:
                pass
    
        # allocate on device
        d_t      = driver.mem_alloc(t.size      * t.dtype.itemsize)
        d_jt     = driver.mem_alloc(jt.size     * jt.dtype.itemsize)
        d_neq    = driver.mem_alloc(neq.size    * neq.dtype.itemsize)
        d_liw    = driver.mem_alloc(liw.size    * liw.dtype.itemsize)
        d_lrw    = driver.mem_alloc(lrw.size    * lrw.dtype.itemsize)
        d_itol   = driver.mem_alloc(itol.size   * itol.dtype.itemsize)
        d_iopt   = driver.mem_alloc(iopt.size   * iopt.dtype.itemsize)
        d_rtol   = driver.mem_alloc(rtol.size   * rtol.dtype.itemsize)
        d_iout   = driver.mem_alloc(iout.size   * iout.dtype.itemsize)
        d_tout   = driver.mem_alloc(tout.size   * tout.dtype.itemsize)
        d_itask  = driver.mem_alloc(itask.size  * itask.dtype.itemsize)
        d_istate = driver.mem_alloc(istate.size * istate.dtype.itemsize)
        d_y      = driver.mem_alloc(y.size      * y.dtype.itemsize)
        d_atol   = driver.mem_alloc(atol.size   * atol.dtype.itemsize)
        d_iwork  = driver.mem_alloc(iwork.size  * iwork.dtype.itemsize)
        d_rwork  = driver.mem_alloc(rwork.size  * rwork.dtype.itemsize)
    
        # copy to device
        driver.memcpy_htod(d_t, t)
        driver.memcpy_htod(d_jt, jt)
        driver.memcpy_htod(d_neq, neq)
        driver.memcpy_htod(d_liw, liw)
        driver.memcpy_htod(d_lrw, lrw)
        driver.memcpy_htod(d_itol, itol)
        driver.memcpy_htod(d_iopt, iopt)
        driver.memcpy_htod(d_rtol, rtol)
        driver.memcpy_htod(d_iout, iout)
        driver.memcpy_htod(d_tout, tout)
        driver.memcpy_htod(d_itask, itask)
        driver.memcpy_htod(d_istate, istate)
        driver.memcpy_htod(d_y, y)
        driver.memcpy_htod(d_atol, atol)
        driver.memcpy_htod(d_iwork, iwork)
        driver.memcpy_htod(d_rwork, rwork)
    
        param = np.zeros((totalThreads,self._parameterNumber),dtype=np.float32)
        try:
            for i in range(len(parameters)):
                for j in range(self._parameterNumber):
                    param[i][j] = parameters[i][j]
        except IndexError:
            pass
    
        # parameter texture
        ary = sim.create_2D_array( param )
        sim.copy2D_host_to_array(ary, param, self._parameterNumber*4, totalThreads )
        self._param_tex.set_array(ary)
    
        if self._dt <= 0:
            start_time = time.time()
            #for i in range(1,self._resultNumber):
            for i in range(0,self._resultNumber):    
    
                for j in range(totalThreads):
                    tout[j] = self._timepoints[i]; 
                driver.memcpy_htod( d_tout, tout ) 
    
                self._compiledRunMethod( d_neq, d_y, d_t, d_tout, d_itol, d_rtol, d_atol, d_itask, d_istate,
                            d_iopt, d_rwork, d_lrw, d_iwork, d_liw, d_jt, block=(threads,1,1), grid=(blocks,1) );
    
                driver.memcpy_dtoh(t, d_t)
                driver.memcpy_dtoh(y, d_y)
                driver.memcpy_dtoh(istate, d_istate)
    
                for j in range(totalThreads):
                    for k in range(self._speciesNumber):
                        ret_xt[j, 0, i, k] = y[j*self._speciesNumber + k]
    
            # end of loop over time points
    
        else:
            tt = self._timepoints[0]
            
            start_time = time.time()
            #for i in range(1,self._resultNumber):
            for i in range(0,self._resultNumber):  
                while 1:
                    
                    next_time = min(tt+self._dt, self._timepoints[i])
                    
                    for j in range(totalThreads):
                        tout[j] = next_time; 
                    driver.memcpy_htod( d_tout, tout ) 
    
                    self._compiledRunMethod( d_neq, d_y, d_t, d_tout, d_itol, d_rtol, d_atol, d_itask, d_istate,
                                d_iopt, d_rwork, d_lrw, d_iwork, d_liw, d_jt, block=(threads,1,1), grid=(blocks,1) );
    
                    driver.memcpy_dtoh(t, d_t)
                    driver.memcpy_dtoh(y, d_y)
                    driver.memcpy_dtoh(istate, d_istate)
    
                    if np.abs(next_time - self._timepoints[i]) < 1e-5:
                        tt = next_time
                        break
    
                    tt = next_time
                    
    
                for j in range(totalThreads):
                    for k in range(self._speciesNumber):
                        ret_xt[j, 0, i, k] = y[j*self._speciesNumber + k]
    
            # end of loop over time points
        
        return ret_xt[0:experiments]


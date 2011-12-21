import os, time

import Simulator as sim
import pycuda.tools as tools
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
from struct import unpack

# total device global memory usage
# MT RNG
# MT  : 32768 * sizeof(mt_struct_stripped) = 32768 x 16 = 524288 bytes
# MTS : 32768 * sizeof(MersenneTwisterState) = 32768 x 96 = 3145728 bytes
# values
# t : nt x 4
# x : nt x pitch (usually 64)
# tot = 3670016 + nt x 68 
# nt = 2048 mem = 3,801,088 = 3.8MB

class Gillespie(sim.Simulator):

    _pvxp = None
    _param_tex = None
    
    _rng_test_source_ ="""
    
    __global__ void TestMersenneTwisters(float *z, int *r) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    int nrand = r[tid];
    
    for(int i=0; i<nrand; ++i){
        z[ i + tid*nrand ] = MersenneTwisterGenerate(&(MTS[tid]), tid)/ 4294967295.0f;
    }
    }
    
    """
    
    def _compile(self, application_code):
        
        mt_cu =  os.path.join(os.path.split(os.path.realpath(__file__))[0],'MersenneTwister.cu')
        
        _gillespieHeader = """
        const int NRESULTS = """ + str(self._resultNumber) + """;
    
        const int BETA = """ + str(self._beta) + """;
        const float DT = """ + str(self._dt) + """f;
        texture<float, 2, cudaReadModeElementType> param_tex;
        """
        
        _gillespieSource="""
        __device__ const float timepoints[""" + str(self._resultNumber) +"]={"
        
        for i in range(self._resultNumber):
            _gillespieSource += str(self._timepoints[i]) + 'f'
            if(i !=  self._resultNumber-1):
                _gillespieSource += ','
        _gillespieSource += '};'
        
        #update species
        _gillespieSource += """
        __device__ void stoichiometry(int *y, int r, int tid){
            for(int i=0; i<NSPECIES; i++){
                y[i]+=smatrix[r*NSPECIES+ i];
            }
        }"""
        
        _gillespieSource += """
        __constant__ int vxp;
        
        __device__ int sample(int nh, float* h, float u){
            int i = 0;
            for(; i < nh - 1 && u > h[i] ; i++){
                u -= h[i];
            }
            
            return i;
        }
        
        
        
        __global__ void Gillespie_one_step(int* vx, int* result){
        
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int texMemIndex = tid/BETA;
        float t = 0.0f;
        float u = 0.0f;
        
        //initialize hazards 
        float h0;
        float h[NREACT];
        for(int i=0; i<NREACT; i++){
            h[i] = 0;
        }
        
        //initialize species
        int y[NSPECIES];
        int k = 0;
        for(int i=0; i<NSPECIES; i++){
            y[i] = ((int*)( (char*) vx + tid * vxp))[i];
        }
        
        //calculate initial hazards
        hazards( y, h, t, texMemIndex );
        h0 = 0;
        for(int i=0; i<NREACT; i++) 
            h0 += h[i];
        
        //calculate next hazard time
        if(h0 > 0){
            u = MersenneTwisterGenerate(&(MTS[tid]), tid)/ 4294967295.0f;
            t += -__logf(u)/h0;
        }
        else t += DT;
        
        for(int i=0; i<NRESULTS;){

            if(t>=timepoints[i]){
                  
                //save current status
                for(int j=0; j<NSPECIES; j++){
                    //non coalesced 
                    result[tid*NRESULTS*NSPECIES + i*NSPECIES + j] = y[j];
                }
                
                //update counter
                i++;
            }
            else{

                if( h0 > 0 ){
                    //update species for current time
                    u = MersenneTwisterGenerate(&(MTS[tid]), tid)/ 4294967295.0f;
                
                    // sample reaction
                    k = sample(NREACT, h, h0*u);
                    // update stochiometry
                    stoichiometry( y ,k, texMemIndex );     
                }

                //calculate hazards for current state
                hazards( y, h, t, texMemIndex );
                h0 = 0;
                for(int j=0; j<NREACT; j++) 
                    h0 += h[j];

                if(h0 > 0){
                    u = MersenneTwisterGenerate(&(MTS[tid]), tid)/ 4294967295.0f;
                    t += -__logf(u)/h0;
                }
                else t += DT;
            }  
        }
        return;
    }
        
    """

        _source_ = _gillespieHeader + application_code + _gillespieSource

        if self._dump == True:
            of = open("full_mjp_code.cu","w")
            print >>of, _source_
        
        m = self._compile_mt_code( _source_, mt_cu)

        self._pvxp = int(m.get_global("vxp")[0])
        self._param_tex = m.get_texref("param_tex")
        
        return (m, m.get_function("Gillespie_one_step"))
            
    def _runSimulation(self, parameters, initValues, blocks, threads):
        totalThreads = blocks * threads
        experiments = len(parameters)
        
        mt_data =  os.path.join(os.path.split(os.path.realpath(__file__))[0],'MersenneTwister.dat')
        
        # initialize Mersenne Twister
        timer = time.time()
        ## print "Init MT..",
        self._initialise_twisters( mt_data, self._completeCode, threads, blocks)
        ## print "finished in", round(time.time()-timer,4), "s"
        
        param = np.zeros((totalThreads/self._beta + 1,self._parameterNumber),dtype=np.float32)
        try:
            for i in range(len(parameters)):
                for j in range(self._parameterNumber):
                    param[i][j] = parameters[i][j]
        except IndexError:
            pass
                
        # parameter texture
        ary = sim.create_2D_array( param )
        sim.copy2D_host_to_array(ary, param, self._parameterNumber*4, totalThreads/self._beta + 1 )
        self._param_tex.set_array(ary)

    
        # 2D species arrays
        x = np.zeros( [totalThreads,self._speciesNumber], dtype=np.int32)
        d_x , p_x = cuda.mem_alloc_pitch(width= self._speciesNumber*4, height=totalThreads, access_size=4 )
        #print '#### 2D arrays:', time.time() - start_time
    
        cuda.memcpy_htod( self._pvxp, np.array([p_x], dtype=np.int32) )
        
        # initialize species
        speciesInput = np.zeros((totalThreads,self._speciesNumber),dtype=np.int32)
        try:
            for i in range(len(initValues)):
                for j in range(self._speciesNumber):
                    speciesInput[i][j] = initValues[i][j]
        except IndexError:
            pass
        sim.copy2D_host_to_device(d_x, speciesInput,  self._speciesNumber*4, p_x,  self._speciesNumber*4, totalThreads )
        
        # output array
        result = np.zeros(totalThreads*self._resultNumber*self._speciesNumber,dtype=np.int32)
        d_result = cuda.mem_alloc(result.nbytes)
        
        # run code
        timer = time.time()
        ## print "Actual run..",
        self._compiledRunMethod(d_x, d_result, block=(threads,1,1), grid=(blocks,1) );
        ## print "finished in", round(time.time()-timer,4), "s"
        
        # fetch from GPU memory
        cuda.memcpy_dtoh(result, d_result)
        result = result[0:experiments*self._beta*self._resultNumber*self._speciesNumber]
        result.shape = (experiments,self._beta,self._resultNumber,self._speciesNumber)
        
        return result


    def _compile_mt_code(self, code, mt_cu, options=[]):
        
        f = open(mt_cu,'r')
        _code_ = f.read() + code
    
        opts = pycuda.driver.jit_option()
        compiled = pycuda.compiler.SourceModule( _code_, nvcc="nvcc", options=options)
    
        return compiled

    def _initialise_twisters(self, mt_data, mod, blockSize, gridSize):
            
        pMT = int(mod.get_global("MT")[0])
        MT_RNG_COUNT = 32768
        
        f = open(mt_data, 'rb')
        s = f.read(16*MT_RNG_COUNT)
        # the file should contain 32768 x 4 integers
        tup = np.array(unpack('131072i',s)).astype(np.uint32)
        
        for i in range(MT_RNG_COUNT):
            tup[i*4 + 3] = np.uint32( 4294967296*np.random.uniform(0,1) )
            #tup[i*4 + 3] = np.uint32( np.random.random_integers(0,4294967295) )


        # Copy the offline MT parameters over to GPU
        cuda.memcpy_htod( pMT, tup )
        
        InitialiseAllMersenneTwisters = mod.get_function("InitialiseAllMersenneTwisters")
        InitialiseAllMersenneTwisters( block=(512,1,1), grid=(64,1) )
    
    #def run_mt_test( mods, blockSize, gridSize, nrandom ):
        #nr = np.array([nrandom for i in range(blockSize*gridSize)], dtype=np.int32)
        #d_nr = cuda.mem_alloc(nr.size*nr.dtype.itemsize)
        #cuda.memcpy_htod(d_nr, nr)
    
        #r = np.zeros( blockSize*gridSize*nrandom, np.float32)
        #d_r = cuda.mem_alloc(r.size*r.dtype.itemsize)
        #cuda.memcpy_htod(d_r, r)
    
        #TestMersenneTwisters = mods[0].get_function("TestMersenneTwisters")
        #TestMersenneTwisters( d_r, d_nr, block=(blockSize,1,1), grid=(gridSize,1) )
    
        #cuda.memcpy_dtoh(r, d_r)
    
        #return r
    




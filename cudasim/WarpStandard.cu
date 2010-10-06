// This code was devloped by David Barrie Thomas at Imperial College
// http://www.doc.ic.ac.uk/~dt10/research/rngs-gpu-uniform.html

    // shared memory allocation for RNG
    extern __shared__ unsigned WarpStandard_shmem[];

    // RNG 
    // Public constants
    const unsigned WarpStandard_K=32;
    const unsigned WarpStandard_REG_COUNT=3;
    const unsigned WarpStandard_STATE_WORDS=32;
    
    // Private constants
    const char *WarpStandard_name="WarpRNG[CorrelatedU32Rng;k=32;g=16;rs=0;w=32;n=1024;hash=deac2e12ec6e615]";
    const char *WarpStandard_post_processing="addtaps";
    const unsigned WarpStandard_N=1024;
    const unsigned WarpStandard_W=32;
    const unsigned WarpStandard_G=16;
    const unsigned WarpStandard_SR=0;
    __device__ const unsigned WarpStandard_Q[2][32]={
        {29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,18,15,27,13,10,16,8,17,25,12,19,30,7,6,4,21},
        {5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,23,4,30,12,25,3,21,26,27,31,18,22,16,29,1}
    };
    const unsigned WarpStandard_Z0=2;
    __device__ const unsigned WarpStandard_Z1[32]={
        0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1};
    const unsigned WarpStandard_SHMEM_WORDS=32;
    const unsigned WarpStandard_GMEM_WORDS=0;
    
    // Public functions
    __device__ void WarpStandard_LoadState(const unsigned *seed, unsigned *regs)
    {
        unsigned offset=threadIdx.x % 32;  unsigned base=threadIdx.x-offset;
        // setup constants
        regs[0]=WarpStandard_Z1[offset];
        regs[1]=base + WarpStandard_Q[0][offset];
        regs[2]=base + WarpStandard_Q[1][offset];
        // Setup state
        unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
        WarpStandard_shmem[threadIdx.x]=seed[stateOff];
    }

    __device__ void WarpStandard_SaveState(const unsigned *regs, unsigned *seed)
    {
        unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
        seed[stateOff] = WarpStandard_shmem[threadIdx.x];
    }

    __device__ unsigned WarpStandard_Generate(unsigned *regs)
    {
    #if __DEVICE_EMULATION__
        __syncthreads();
    #endif
        unsigned t0=WarpStandard_shmem[regs[1]], t1=WarpStandard_shmem[regs[2]];
        unsigned res=(t0<<WarpStandard_Z0) ^ (t1>>regs[0]);
    #if __DEVICE_EMULATION__
        __syncthreads();
    #endif
        WarpStandard_shmem[threadIdx.x]=res;
        return t0+t1;
    };

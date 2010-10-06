#define NSPECIES 1
#define NPARAM 3
#define NREACT 2

__device__ float function_1(float a1){
    return a1;
}

struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;



        ydot[0]=(1.0*(tex2D(param_tex,0,tid)*function_1(tex2D(param_tex,1,tid)))-1.0*(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[0]))/tex2D(param_tex,0,tid);

    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};
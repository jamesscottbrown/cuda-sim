#define NSPECIES 3
#define NPARAM 9
#define NREACT 6

struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;



        ydot[0]=((tex2D(param_tex,4,tid)*tex2D(param_tex,8,tid))-(tex2D(param_tex,5,tid)*y[0])-(tex2D(param_tex,6,tid)*y[0]*y[2]/(y[0]+tex2D(param_tex,7,tid)*tex2D(param_tex,8,tid))))/tex2D(param_tex,0,tid);
        ydot[1]=((tex2D(param_tex,1,tid)*y[0])-(tex2D(param_tex,2,tid)*y[1]))/tex2D(param_tex,0,tid);
        ydot[2]=((tex2D(param_tex,2,tid)*y[1])-(tex2D(param_tex,3,tid)*y[2]))/tex2D(param_tex,0,tid);

    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};
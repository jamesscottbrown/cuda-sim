#define NSPECIES 1
#define NPARAM 3
#define NREACT 2

__device__ float function_1(float a1){
    return a1;
}


__constant__ int smatrix[]={
    1.0,
    -1.0,
};


__device__ void hazards(int *y, float *h, float t, int tid){

    h[0] = tex2D(param_tex,0,tid)*function_1(tex2D(param_tex,1,tid));
    h[1] = tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[0];

}


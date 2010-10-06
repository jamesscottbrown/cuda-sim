#define NSPECIES 3
#define NPARAM 9
#define NREACT 6



__constant__ int smatrix[]={
    1.0,    0.0,    0.0,
    -1.0,    0.0,    0.0,
    -1.0,    0.0,    0.0,
    0.0,    1.0,    0.0,
    0.0,    -1.0,    1.0,
    0.0,    0.0,    -1.0,
};


__device__ void hazards(int *y, float *h, float t, int tid){

    h[0] = tex2D(param_tex,4,tid)*tex2D(param_tex,8,tid);
    h[1] = tex2D(param_tex,5,tid)*y[0];
    h[2] = tex2D(param_tex,6,tid)*y[0]*y[2]/(y[0]+tex2D(param_tex,7,tid)*tex2D(param_tex,8,tid));
    h[3] = tex2D(param_tex,1,tid)*y[0];
    h[4] = tex2D(param_tex,2,tid)*y[1];
    h[5] = tex2D(param_tex,3,tid)*y[2];

}


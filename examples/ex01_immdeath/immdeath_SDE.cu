#define NSPECIES 1
#define NPARAM 3
#define NREACT 2

//Code for texture memory
__device__ float function_1(float a1){
    return a1;
}


__device__ void step(float *y, float t, unsigned *rngRegs, int tid){

    float d_y0= DT * (((tex2D(param_tex,0,tid)*function_1(tex2D(param_tex,1,tid)))-(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[0]))/tex2D(param_tex,0,tid));


    d_y0 += ((sqrtf(tex2D(param_tex,0,tid)*function_1(tex2D(param_tex,1,tid)))*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT))-sqrtf(tex2D(param_tex,0,tid)*tex2D(param_tex,2,tid)*y[0])*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT)))/tex2D(param_tex,0,tid));

    y[0] += d_y0;
}
//Code for shared memory
__device__ float function_1(float a1){
    return a1;
}


__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){

    float d_y0= DT * (((parameter[0]*function_1(parameter[1]))-(parameter[0]*parameter[2]*y[0]))/parameter[0]);


    d_y0+= ((sqrtf(parameter[0]*function_1(parameter[1]))*randNormal(rngRegs,sqrt(DT))-sqrtf(parameter[0]*parameter[2]*y[0])*randNormal(rngRegs,sqrt(DT)))/parameter[0]);

    y[0] += d_y0;
}

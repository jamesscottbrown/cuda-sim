#define NSPECIES 3
#define NPARAM 9
#define NREACT 6

//Code for texture memory

__device__ void step(float *y, float t, unsigned *rngRegs, int tid){

    float d_y0= DT * (((tex2D(param_tex,4,tid)*tex2D(param_tex,8,tid))-(tex2D(param_tex,5,tid)*y[0])-(tex2D(param_tex,6,tid)*y[0]*y[2]/(y[0]+tex2D(param_tex,7,tid)*tex2D(param_tex,8,tid))))/tex2D(param_tex,0,tid));
    float d_y1= DT * (((tex2D(param_tex,1,tid)*y[0])-(tex2D(param_tex,2,tid)*y[1]))/tex2D(param_tex,0,tid));
    float d_y2= DT * (((tex2D(param_tex,2,tid)*y[1])-(tex2D(param_tex,3,tid)*y[2]))/tex2D(param_tex,0,tid));

    float rand4 = randNormal(rngRegs,WarpStandard_shmem,sqrt(DT));

    d_y0 += ((sqrtf(tex2D(param_tex,4,tid)*tex2D(param_tex,8,tid))*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT))-sqrtf(tex2D(param_tex,5,tid)*y[0])*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT))-sqrtf(tex2D(param_tex,6,tid)*y[0]*y[2]/(y[0]+tex2D(param_tex,7,tid)*tex2D(param_tex,8,tid)))*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT)))/tex2D(param_tex,0,tid));
    d_y1 += ((sqrtf(tex2D(param_tex,1,tid)*y[0])*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT))-sqrtf(tex2D(param_tex,2,tid)*y[1])*rand4)/tex2D(param_tex,0,tid));
    d_y2 += ((sqrtf(tex2D(param_tex,2,tid)*y[1])*rand4-sqrtf(tex2D(param_tex,3,tid)*y[2])*randNormal(rngRegs,WarpStandard_shmem,sqrt(DT)))/tex2D(param_tex,0,tid));

    y[0] += d_y0;
    y[1] += d_y1;
    y[2] += d_y2;
}
//Code for shared memory

__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){

    float d_y0= DT * (((parameter[4]*parameter[8])-(parameter[5]*y[0])-(parameter[6]*y[0]*y[2]/(y[0]+parameter[7]*parameter[8])))/parameter[0]);
    float d_y1= DT * (((parameter[1]*y[0])-(parameter[2]*y[1]))/parameter[0]);
    float d_y2= DT * (((parameter[2]*y[1])-(parameter[3]*y[2]))/parameter[0]);

    float rand4 = randNormal(rngRegs,sqrt(DT));

    d_y0+= ((sqrtf(parameter[4]*parameter[8])*randNormal(rngRegs,sqrt(DT))-sqrtf(parameter[5]*y[0])*randNormal(rngRegs,sqrt(DT))-sqrtf(parameter[6]*y[0]*y[2]/(y[0]+parameter[7]*parameter[8]))*randNormal(rngRegs,sqrt(DT)))/parameter[0]);
    d_y1+= ((sqrtf(parameter[1]*y[0])*randNormal(rngRegs,sqrt(DT))-sqrtf(parameter[2]*y[1])*rand4)/parameter[0]);
    d_y2+= ((sqrtf(parameter[2]*y[1])*rand4-sqrtf(parameter[3]*y[2])*randNormal(rngRegs,sqrt(DT)))/parameter[0]);

    y[0] += d_y0;
    y[1] += d_y1;
    y[2] += d_y2;
}

#define NSPECIES 3
#define NPARAM 9
#define NREACT 6

//Code for texture memory

__device__ void step(float *y, float t, unsigned *rngRegs, int tid){

    float d_y0= DT * ((1.0*(tex2D(param_tex,4,tid)*tex2D(param_tex,8,tid))-1.0*(tex2D(param_tex,5,tid)*y[0])-1.0*(tex2D(param_tex,6,tid)*y[0]*y[2]/(y[0]+tex2D(param_tex,7,tid)*tex2D(param_tex,8,tid))))/tex2D(param_tex,0,tid));
    float d_y1= DT * ((1.0*(tex2D(param_tex,1,tid)*y[0])-1.0*(tex2D(param_tex,2,tid)*y[1]))/tex2D(param_tex,0,tid));
    float d_y2= DT * ((1.0*(tex2D(param_tex,2,tid)*y[1])-1.0*(tex2D(param_tex,3,tid)*y[2]))/tex2D(param_tex,0,tid));

    float rand4 = randNormal(rngRegs,sqrt(DT));

    d_y0 += ((1.0*sqrt(tex2D(param_tex,4,tid)*tex2D(param_tex,8,tid))*randNormal(rngRegs,sqrt(DT))-1.0*sqrt(tex2D(param_tex,5,tid)*y[0])*randNormal(rngRegs,sqrt(DT))-1.0*sqrt(tex2D(param_tex,6,tid)*y[0]*y[2]/(y[0]+tex2D(param_tex,7,tid)*tex2D(param_tex,8,tid)))*randNormal(rngRegs,sqrt(DT)))/tex2D(param_tex,0,tid));
    d_y1 += ((1.0*sqrt(tex2D(param_tex,1,tid)*y[0])*randNormal(rngRegs,sqrt(DT))-1.0*sqrt(tex2D(param_tex,2,tid)*y[1])*rand4)/tex2D(param_tex,0,tid));
    d_y2 += ((1.0*sqrt(tex2D(param_tex,2,tid)*y[1])*rand4-1.0*sqrt(tex2D(param_tex,3,tid)*y[2])*randNormal(rngRegs,sqrt(DT)))/tex2D(param_tex,0,tid));

    y[0] += d_y0;
    y[1] += d_y1;
    y[2] += d_y2;
}
//Code for shared memory

__device__ void step(float *parameter, float *y, float t, unsigned *rngRegs){

    float d_y0= DT * ((1.0*(parameter[4]*parameter[8])-1.0*(parameter[5]*y[0])-1.0*(parameter[6]*y[0]*y[2]/(y[0]+parameter[7]*parameter[8])))/parameter[0]);
    float d_y1= DT * ((1.0*(parameter[1]*y[0])-1.0*(parameter[2]*y[1]))/parameter[0]);
    float d_y2= DT * ((1.0*(parameter[2]*y[1])-1.0*(parameter[3]*y[2]))/parameter[0]);

    float rand4 = randNormal(rngRegs,sqrt(DT));

    d_y0+= ((1.0*sqrt(parameter[4]*parameter[8])*randNormal(rngRegs,sqrt(DT))-1.0*sqrt(parameter[5]*y[0])*randNormal(rngRegs,sqrt(DT))-1.0*sqrt(parameter[6]*y[0]*y[2]/(y[0]+parameter[7]*parameter[8]))*randNormal(rngRegs,sqrt(DT)))/parameter[0]);
    d_y1+= ((1.0*sqrt(parameter[1]*y[0])*randNormal(rngRegs,sqrt(DT))-1.0*sqrt(parameter[2]*y[1])*rand4)/parameter[0]);
    d_y2+= ((1.0*sqrt(parameter[2]*y[1])*rand4-1.0*sqrt(parameter[3]*y[2])*randNormal(rngRegs,sqrt(DT)))/parameter[0]);

    y[0] += d_y0;
    y[1] += d_y1;
    y[2] += d_y2;
}

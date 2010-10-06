#define NPMAX 7
#define NXMAX 3
#define omega 200

// initial conditions
int initial[] = { 0, 20, 160 };
// parameters
float pars[] = { 0.9, 0.3, 1.7, 1.2, 0.9, 0.8, 0.0001 };


__device__ float afun(float* x, int s, int tid){
  
  if(x[0]<0)
    x[0] = 0;
  if(x[1]<0)
    x[1] = 0;
  if(x[2]<0)
    x[2] = 0;

  if( s == 0 ) return tex2D(param_tex, 0, tid) * omega - tex2D(param_tex, 1, tid) * x[0] - tex2D(param_tex, 2, tid) * x[0] * x[2]/(x[0] + tex2D(param_tex, 6, tid) * omega);
  if( s == 1 ) return tex2D(param_tex, 3, tid) * x[0] - tex2D(param_tex, 4, tid) * x[1];
  if( s == 2 ) return tex2D(param_tex, 4, tid) * x[1] - tex2D(param_tex, 5, tid) * x[2];

  return 0;
}

__device__ float bfun(float* x, int s, int tid){

  if(x[0]<0)
    x[0] = 0;
  if(x[1]<0)
    x[1] = 0;
  if(x[2]<0)
    x[2] = 0;

  if( s == 0 ) return sqrt( tex2D(param_tex, 0, tid)*omega + tex2D(param_tex, 1, tid)*fabs(x[0]) + tex2D(param_tex, 2, tid)*fabs(x[0])*fabs(x[2])/(fabs(x[0])+tex2D(param_tex, 6, tid) * omega) );
  if( s == 1 ) return sqrt( tex2D(param_tex, 3, tid)*fabs(x[0]) + tex2D(param_tex, 4, tid)*fabs(x[1]) );
  if( s == 2 ) return sqrt( tex2D(param_tex, 4, tid)*fabs(x[1]) + tex2D(param_tex, 5, tid)*fabs(x[2]) );
  
  return 0;
}

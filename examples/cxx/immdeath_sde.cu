#define NPMAX 2
#define NXMAX 1

// initial conditions and parameters
int initial[] = { 0 };
float pars[] = { 1.0, 0.1 };


__device__ float afun(float* x, int s, int tid){

  if( s == 0 ){
    return tex2D(param_tex, 0, tid) - tex2D(param_tex, 1, tid)*x[0];
  }
  
  return 0;
}

__device__ float bfun(float* x, int s, int tid){

  if( s == 0 ){
    return sqrt( tex2D(param_tex, 0, tid) + tex2D(param_tex, 1, tid)*fabs(x[0]) );
  }
  
  return 0;
}


#define NR 2
#define NPMAX 2
#define NXMAX 1

// initial conditions
int initial[] = { 0 };
// parameters
float pars[] = { 1.0, 0.1 };

__constant__ int smatrix[] = {1, -1};

__device__ void stoichiometry(int *x, int r, int tid){
  x[0] = x[0] + smatrix[r];
}

__device__ void hazards(int *x, float *h, int tid){
  h[0] = tex2D(param_tex, 0, tid);
  h[1] = x[0] > 0 ? x[0]*tex2D(param_tex, 1, tid) : 0;
}


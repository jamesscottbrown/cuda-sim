
#define NR 6
#define NXMAX 3
#define NPMAX 7
#define omega 200

// initial conditions
int initial[] = { 0, 20, 160 };
// parameters
float pars[] = { 0.9, 0.3, 1.7, 1.2, 0.9, 0.8, 0.0001 };


// NR * NX stoch matrix
__constant__ int smatrix[] = { 1,  0,  0,
			      -1,  0,  0,
			      -1,  0,  0,
			       0,  1,  0,
			       0, -1,  1,
			       0,  0, -1 };

__device__ void stoichiometry(int *x, int r, int tid){
  for(int i=0; i<3; ++i){
    x[i] += smatrix[r*3 + i];
  }
}

__device__ void hazards(int *x, float *h, int tid){
  
  h[0] = tex2D(param_tex, 0, tid) * omega;
  h[1] = x[0] > 0 ? tex2D(param_tex, 1, tid) * x[0] : 0;
  h[2] = x[0] > 0 && x[2] > 0 ? tex2D(param_tex, 2, tid) * x[0] * x[2]/(x[0] + tex2D(param_tex, 6, tid) * omega) : 0;
  h[3] = x[0] > 0 ? tex2D(param_tex, 3, tid) * x[0] : 0;
  h[4] = x[1] > 0 ? tex2D(param_tex, 4, tid) * x[1] : 0;
  h[5] = x[2] > 0 ? tex2D(param_tex, 5, tid) * x[2] : 0;

}




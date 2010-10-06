// C++ headers
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
using namespace std;

// C headers
#include <stdio.h>
#include <cassert>
#include <cstdio>
#include <sys/time.h>
#include <gsl/gsl_rng.h>

/////////////////////////////////////////
// Application code
/////////////////////////////////////////
gsl_rng * r;

#define TWOPI 6.283185

// mimic the texture code
vector<vector<float> > param;
int param_tex = 0;

float tex2D(const int& dummy, const int& p, const int& tid){
  return param[tid][p];
}

#define __constant__ 
#define __device__ 
#define __global__

#include "app.cu"

// mimic the block and thread info
struct b_id{
  int x;
} blockIdx;
struct b_dim{
  int x;
} blockDim;
struct t_id{
  int x;
} threadIdx;

float dt = 0.01;
float sdt = sqrt(dt);

__constant__ int d_it;
__constant__ int vxp;


__global__ void EulerMaruyama_one_step(unsigned *seed, float* vx){
  
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  int i, k;
  float u1, u2, n1, x[NXMAX];

  for(i=0; i<NXMAX; ++i){
    x[i] = ((float*)( (char*) vx + tid * vxp))[i];
  }
  
  // Loop over iterations
  for(k=0; k<d_it; ++k){
    
    // Loop over species
    for(i=0; i<NXMAX; ++i){ 
      // Uniform variables
      //float u1 = rand()/( (float) RAND_MAX);
      //float u2 = rand()/( (float) RAND_MAX);
      float u1 = gsl_rng_uniform (r);
      float u2 = gsl_rng_uniform (r);

      // Box-Muller transformation to N(mean=0,var=dt)
      n1 = sdt*sqrt( -2*log(u1) )*cos(TWOPI*u2);

      // Update species i
      x[i] += afun(x,i,tid)*dt + bfun(x,i,tid)*n1;
    }     
  }

  for(i=0; i<NXMAX; ++i){ 
    ((float*)( (char*) vx + tid * vxp))[i] = x[i];
  }
}

int main(int argc, char **argv)
{

  if( argc != 6 ){
    cout << "Incorrect number of arguments. should be ./main_ssa_cpu <grid> <block> <ntimes> <outputntimes> <outfile>" << endl; 
    exit(1);
  }

  const gsl_rng_type * T;
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  
  int gridSize = atoi(argv[1]);
  int blockSize = atoi(argv[2]);
  int ntimes = atoi(argv[3]); 
  int outntimes = atoi(argv[4]);

  char* fout = argv[5];
  int nthread = gridSize*blockSize;
  blockDim.x = blockSize;

  timeval start, stop, result;
  gettimeofday(&start,NULL);

  // create the time series from 0 to ntimes-1  
  float* times = (float*) malloc((outntimes)*sizeof(float));
  float outputStep = ntimes / (float)outntimes;

  for(int i=0; i<outntimes+1; ++i){
    times[i] = (float) i*outputStep;
  }
   
  // output array
  vector<vector<vector<double> > > results(nthread, vector<vector<double> >(NXMAX, vector<double>(outntimes,0)));
  
  // timer and integration params
  double t = times[0];
 
  // 2D species arrays
  float *x;
  x = (float*) malloc(nthread*NXMAX*sizeof(float));
  vxp = NXMAX*sizeof(float);
 
  
  // parameter texture 
  for(int i=0; i<nthread; ++i){
    param.push_back( vector<float>(NPMAX,0) );
  }

  // initialise
  for(int i=0; i<nthread; ++i){
   
    for(int j=0; j<NXMAX; ++j){
      x[i*NXMAX + j] = initial[j];
      results[i][j][0] = x[i*NXMAX + j];
    }

    for(int j=0; j<NPMAX; ++j){
      param[i][j] = pars[j];
    }
  }
  
  for(int i=1; i<outntimes; ++i){
   
    // Calculate number of iterations until next time point
    unsigned it = 0;
    while(t < times[i]){
      t += dt;
      ++it;
    }
    d_it = it;

    for(int n1=0; n1<gridSize; ++n1){
      blockIdx.x = n1;
      for(int n2=0; n2<blockSize; ++n2){
	threadIdx.x = n2;

	EulerMaruyama_one_step(0, x );
	
	int nt = blockDim.x * blockIdx.x + threadIdx.x;
	//cout << "nt:" << nt << endl;

	for(int j=0; j<NXMAX; ++j){
	  results[nt][j][i] = x[nt*NXMAX + j];
	}
      }
    }
  }  
  gettimeofday(&stop,NULL);

  if( strcmp(fout,"NULL") != 0){
    ofstream out(fout);
    for(int nt=0; nt<nthread; ++nt){
      for(int j=0; j<NXMAX; ++j){
 	out << j;
 	for(int i=0; i<results[nt][j].size(); ++i){
 	  out << "\t" << results[nt][j][i];
 	}
 	out << endl;
      }
    }
    out.close();
  }

  timersub(&stop,&start,&result);
  double delta_t = result.tv_sec + result.tv_usec/1000000.0;
  cout << "timing\t" << nthread << "\t" << ntimes << "\t" << delta_t << endl;

  return 0;
}

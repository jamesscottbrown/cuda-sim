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


__constant__ float tmax;
__constant__ int vxp;
__constant__ int vxtp;

__device__ int sample(int nh, float* h, float u){
  
  int i = 0;
  for(i=0; i<nh; ++i){
    if( u < h[i] ) break;
    u = u - h[i];
  }
  return i;
}

__global__ void Gillespie_one_step(float *vt, int* vx, int* vxt){

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int i, x[NXMAX];
  float h0, h[NR];
 
  for(i=0; i<NXMAX; ++i){
    x[i] = ((int*)( (char*) vx + tid * vxp))[i];
  }

  float t  = vt[ tid ];
   
  while( t < tmax ){

    // calculate hazards
    hazards( x, h, tid );

    h0 = 0;
    for(i=0; i<NR; ++i) h0 += h[i];

    if(h0 <= 0){
      
      // copy the current local values over to xt
      for(i=0; i<NXMAX; ++i){
        ((int*)( (char*) vxt + tid * vxtp))[i] = x[i];
      }
      break;
      
    }else{
      //float u1 = MersenneTwisterGenerate(&(MTS[tid]), tid)/ 4294967295.0f;
      //float u2 = MersenneTwisterGenerate(&(MTS[tid]), tid)/ 4294967295.0f;
      //float u1 = rand()/( (float) RAND_MAX);
      //float u2 = rand()/( (float) RAND_MAX);
      float u1 = gsl_rng_uniform (r);
      float u2 = gsl_rng_uniform (r);

      // increment the time
      t += -log(u1)/h0;
    
      if( t >= tmax ){

        // copy the current local values over to xt
        for(i=0; i<NXMAX; ++i){
          ((int*)( (char*) vxt + tid * vxtp))[i] = x[i];
        }  
      }

      // sample reaction
      i = sample(NR, h, h0*u2);

      // update stochiometry
      stoichiometry( x ,i, tid );
      }
    }

    vt[tid] = t;
    for(i=0; i<NXMAX; ++i){
      ((int*)( (char*) vx + tid * vxp))[i] = x[i];
    }
}

int main(int argc, char **argv)
{

  if( argc != 6 ){
    cout << "Incorrect number of arguments. should be ./main_ssa_cpu <grid> <block> <ntimes> <outntimes> <outfile>" << endl; 
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

  //cout << "args:" << gridSize << "," << blockSize << "," << ntimes << "," << nthread << "," << blockDim.x << endl;

  // create the time series from 0 to ntimes-1  
  float* times = (float*) malloc((outntimes)*sizeof(float));
  float outputStep = ntimes / (float)outntimes;

  for(int i=0; i<outntimes+1; ++i){
    times[i] = (float) i*outputStep;
  }
  

  // output array
  vector<vector<vector<double> > > results(nthread, vector<vector<double> >(NXMAX, vector<double>(outntimes,0)));

  // times array
  float *t;
  t = (float*) malloc(nthread*sizeof(float));
  
  // 2D species arrays
  int *x, *xt;
  x = (int*) malloc(nthread*NXMAX*sizeof(int));
  xt = (int*) malloc(nthread*NXMAX*sizeof(int));
  vxp = NXMAX*sizeof(int);
  vxtp = NXMAX*sizeof(int);

  // model texture
  
  // parameter texture 
  for(int i=0; i<nthread; ++i){
    param.push_back( vector<float>(NPMAX,0) );
  }

  // initialise
  for(int i=0; i<nthread; ++i){
    t[i] = times[0];

    for(int j=0; j<NXMAX; ++j){
      x[i*NXMAX + j] = initial[j];
      xt[i*NXMAX + j] = x[i*NXMAX + j];
      results[i][j][0] = x[i*NXMAX + j];
    }

    for(int j=0; j<NPMAX; ++j){
      param[i][j] = pars[j];
    }
  }
  
  for(int i=1; i<outntimes; ++i){
    tmax = times[i];

    for(int n1=0; n1<gridSize; ++n1){
      blockIdx.x = n1;
      for(int n2=0; n2<blockSize; ++n2){
	threadIdx.x = n2;

	Gillespie_one_step(t, x, xt);
	
	int nt = blockDim.x * blockIdx.x + threadIdx.x;
	for(int j=0; j<NXMAX; ++j){
	  results[nt][j][i] = xt[nt*NXMAX + j];
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

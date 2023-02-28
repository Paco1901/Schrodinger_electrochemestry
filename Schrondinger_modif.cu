#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <complex>
 
using namespace std;
typedef complex<long double> long_double_complex;
long_double_complex Il=long_double_complex(0.0,1.0);
 
const long double Cl_SI_HBAR           =1.054571596e-34;
const long double Cl_SI_H=1.054571596e-34*2.0*M_PI;
const long double Cl_SI_MASS_ELECTRON  =9.10938188e-31;
const long double Cl_SI_CHARGE_ELECTRON=1.602176462e-19;
const long double Cl_SI_EPSILON0       =8.854187817e-12;
const long double Cl_SI_KB             = 1.3806503e-23;
 
#ifdef _GPU
#include <cuda.h>
#include <cufft.h>
#else
#include <fftw3.h>
#endif
 
#ifdef _GPU
 
#define BLOCK_DIM  16
#define BLOCK_SIZE 256
 
__global__ void GPUMatrixElementMult(double2* iA, double2* iB, double2* oC, int N, double scale)
{
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
   double2 z;
   if (idx<N)
   {
      z.x = iA[idx].x * iB[idx].x - iA[idx].y * iB[idx].y;
      z.y = iA[idx].x * iB[idx].y + iA[idx].y * iB[idx].x;
      oC[idx].x = z.x *scale;
      oC[idx].y = z.y *scale;
   }
}
 
/* GPU Correlation requires to adapt the "reduction" example from the SDK,
since we have to avoid memory synchronization problems when we write the results
from all threads in a single double */
 
/* slow serial kernel */
__global__ void  GPUSummation_serial( const double2  *iA, double2* oC, int N, int it )
{
    if( threadIdx.x == 0 )
    {
        oC[it].x = 0.0;
        oC[it].y = 0.0;
        for(int idx = 0; idx < N; idx++)
        {
            oC[it].x += iA[idx].x;
            oC[it].y += iA[idx].y;
        }
    }
}
 
__global__ void  GPUSummation_parallel_partial( const double2  *iA, double2* g_odata, int N )
{
    __shared__ double2 sdata[BLOCK_SIZE];
 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE * gridDim.x;
    double2 accum;
    
    accum.x = iA[idx].x;
    accum.y = iA[idx].y;
    idx+=gridSize;
 
    while (idx < N)
    {
        accum.x += iA[idx].x;
        accum.y += iA[idx].y;
        idx += gridSize;
    }
 
    sdata[tid].x=accum.x;
    sdata[tid].y=accum.y;
 
    __syncthreads();
 
    if (BLOCK_SIZE >= 512) { if (tid < 256) 
    { sdata[tid].x += sdata[tid + 256].x; sdata[tid].y += sdata[tid + 256].y; } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) 
    { sdata[tid].x += sdata[tid + 128].x; sdata[tid].y += sdata[tid + 128].y; } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) 
    { sdata[tid].x += sdata[tid +  64].x; sdata[tid].y += sdata[tid +  64].y; } __syncthreads(); }
 
    if (tid < 32)
    {
        if (BLOCK_SIZE >=  64) { sdata[tid].x += sdata[tid + 32].x; sdata[tid].y += sdata[tid + 32].y;  __syncthreads(); }
        if (BLOCK_SIZE >=  32) { sdata[tid].x += sdata[tid + 16].x; sdata[tid].y += sdata[tid + 16].y;  __syncthreads(); }
        if (BLOCK_SIZE >=  16) { sdata[tid].x += sdata[tid +  8].x; sdata[tid].y += sdata[tid +  8].y;  __syncthreads(); }
        if (BLOCK_SIZE >=   8) { sdata[tid].x += sdata[tid +  4].x; sdata[tid].y += sdata[tid +  4].y;  __syncthreads(); }
        if (BLOCK_SIZE >=   4) { sdata[tid].x += sdata[tid +  2].x; sdata[tid].y += sdata[tid +  2].y;  __syncthreads(); }
        if (BLOCK_SIZE >=   2) { sdata[tid].x += sdata[tid +  1].x; sdata[tid].y += sdata[tid +  1].y;  __syncthreads(); }
    }
    // write result for this block to global mem 
    if (tid == 0) 
    {
       g_odata[blockIdx.x].x = sdata[0].x; 
       g_odata[blockIdx.x].y = sdata[0].y; 
    }
}
 
#else
 
// Store in oC the matrix element wise multiplication of iA and iC, all objects have N elements, scale the result
void CPUMatrixElementMultScale(fftw_complex* iA, fftw_complex* iB, fftw_complex* oC, int N, double scale)
{
   int idx;
   fftw_complex z;
   for(idx=0;idx<N;idx++)
   {
      z[0] = iA[idx][0] * iB[idx][0] - iA[idx][1] * iB[idx][1];
      z[1] = iA[idx][0] * iB[idx][1] + iA[idx][1] * iB[idx][0];
      oC[idx][0] = z[0] * scale;
      oC[idx][1] = z[1] * scale;
   }
}
 
void CPUSummation(fftw_complex* iA, fftw_complex* oC, int N, int it)
{
   int idx;
 
   oC[it][0]=0.0;
   oC[it][1]=0.0;
 
   for(idx=0;idx<N;idx++)
   {
      oC[it][0] +=  iA[idx][0];
      oC[it][1] +=  iA[idx][1];
   }
}
 
#endif
 
long double psiInit(long double x, long double y, long double a )
{
   return exp(-(x*x+y*y)/(2.0*a*a))/(a*sqrt(M_PI));
}
 
long double potential( long double x, long double y )
{
   return 10000.0*Cl_SI_CHARGE_ELECTRON*x;
}
 
int main( void )
{
   int NT,NX,NY,N,in,it;
   double *xGrid,*yGrid,*kxGrid,*kyGrid,*kxGridShift,*kyGridShift;
   long double x0,y0,x1,y1,DX,DY,dx,dy;
   long double dt,h,m,a,meff,err;
   FILE *fd;
 
#ifdef _GPU
   cufftHandle plan;
#else
   fftw_plan  fftplanAll;
   fftw_plan ifftplanAll;
#endif
 
#ifdef _GPU
   double2 *dev_psiPosGrid, *dev_psiPosInitGrid, *dev_psiMomGrid, *dev_expTGrid, *dev_expVGrid, *dev_act ,*dev_psiCor,*dev_g_odata; // device
   double2 *psiPosGrid,*psiPosInitGrid,*psiMomGrid,*expTGrid,*expVGrid,*act,*psiCor;
#else
   fftw_complex *psiPosGrid,*psiPosInitGrid,*psiMomGrid,*expTGrid,*expVGrid,*act,*psiCor;
#endif
 
   size_t sizeN,sizeNT;
   clock_t c0,c1;
 
   NX=1024; // grid points in x-direction
   NY=1024; // grid points in y-direction
 
   DX=0.4E-6; // half-width of potential in x-direction
   DY=0.4E-6; // half-width of potential in y-direction
 
   dx=2.0*DX/(double)(NX); // grid step size in x-direction
   dy=2.0*DY/(double)(NY); // grid step size in y-direction
   x0=-DX; // lower left corner x-coordinate
   y0=-DY; // lower left corner y-coordinate
   x1=x0+2.0*DX; // upper right corner x-coordinate
   y1=y0+2.0*DY; // upper right corner y-coordinate
   N=NX*NY; // total number of grid points
 
   NT=1000; // number of time-propagtion steps
   dt=100.0E-15; // time step
 
   meff=0.067; // effective mass
   a=80.0E-9; // gaussian width of initial wavepacket
   h=Cl_SI_HBAR; // hbar
   m=meff*Cl_SI_MASS_ELECTRON; // electron mass in kg
 
#ifdef _GPU
   sizeN  = N  * sizeof(double2);
   sizeNT = NT * sizeof(double2);
   psiPosGrid     = (double2*)malloc(sizeN);
   psiPosInitGrid = (double2*)malloc(sizeN);
   psiMomGrid     = (double2*)malloc(sizeN);
   psiCor         = (double2*)malloc(sizeN);
   expTGrid       = (double2*)malloc(sizeN);
   expVGrid       = (double2*)malloc(sizeN);
   act            = (double2*)malloc(sizeNT);
 
   cudaMalloc((void**)&dev_psiPosGrid,sizeN);
   cudaMalloc((void**)&dev_psiPosInitGrid,sizeN);
   cudaMalloc((void**)&dev_psiMomGrid,sizeN);
   cudaMalloc((void**)&dev_psiCor,sizeN);
   cudaMalloc((void**)&dev_expTGrid,sizeN);
   cudaMalloc((void**)&dev_expVGrid,sizeN);
   cudaMalloc((void**)&dev_act,sizeNT);
   cudaMalloc((void**)&dev_g_odata,BLOCK_SIZE);
   cufftPlan2d(&plan, NX, NY, CUFFT_Z2Z);
#else
   sizeN  = N  * sizeof(fftw_complex);
   sizeNT = NT * sizeof(fftw_complex);
   psiPosGrid     = (fftw_complex*)fftw_malloc(sizeN);
   psiPosInitGrid = (fftw_complex*)fftw_malloc(sizeN);
   psiMomGrid     = (fftw_complex*)fftw_malloc(sizeN);
   psiCor         = (fftw_complex*)fftw_malloc(sizeN);
   expTGrid       = (fftw_complex*)fftw_malloc(sizeN);
   expVGrid       = (fftw_complex*)fftw_malloc(sizeN);
   act            = (fftw_complex*)fftw_malloc(sizeNT);
 
   fftplanAll =fftw_plan_dft_2d(NX,NY,(fftw_complex*)psiPosGrid,(fftw_complex*)psiMomGrid,FFTW_BACKWARD,FFTW_MEASURE);
   ifftplanAll=fftw_plan_dft_2d(NX,NY,(fftw_complex*)psiMomGrid,(fftw_complex*)psiPosGrid,FFTW_FORWARD ,FFTW_MEASURE);
#endif
 
   // initialize the position space grid
   // initialize the momentum space grid and shift it
   xGrid      = (double*) malloc(sizeof(double)*NX);
   kxGrid     = (double*) malloc(sizeof(double)*NX);
   kxGridShift= (double*) malloc(sizeof(double)*NX);
   kyGrid     = (double*) malloc(sizeof(double)*NY);
   kyGridShift= (double*) malloc(sizeof(double)*NY);
   yGrid      = (double*) malloc(sizeof(double)*NY);
   for(int ix=0;ix<NX;ix++)
   {
      xGrid[ix]=x0+ix*dx;
      kxGrid[ix]=-M_PI/dx+double(ix)*2.0*M_PI/double(NX)/dx;
   }
   for(int ix=0;ix<NX/2;ix++)
   {
      kxGridShift[ix]=kxGrid[NX/2+ix];
   }
   for(int ix=NX/2;ix<NX;ix++)
   {
      kxGridShift[ix]=kxGrid[ix-NX/2];
   }
   for(int iy=0;iy<NY;iy++)
   {
      yGrid[iy]=y0+iy*dy;
      kyGrid[iy]=-M_PI/dy+double(iy)*2.0*M_PI/double(NY)/dy;
   }
   for(int iy=0;iy<NY/2;iy++)
   {
      kyGridShift[iy]=kyGrid[NY/2+iy];
   }
   for(int iy=NY/2;iy<NY;iy++)
   {
      kyGridShift[iy]=kyGrid[iy-NY/2];
   }
 
   for(int iy=0;iy<NY;iy++)
   {
      for(int ix=0;ix<NX;ix++)
      {
         int in=ix*NY+iy;
         // do all intermediate calculations in long double to avoid any out of range errors, which DO happen if one uses double for the exp()
         long double V=potential(xGrid[ix],yGrid[iy]);
         long_double_complex expV=exp(Il*(long double)(-(V                                                                                        )*dt/h));
         long_double_complex expT=exp(Il*(long double)(-(h*h*kxGridShift[ix]*kxGridShift[ix]/(2.0l*m)+h*h*kyGridShift[iy]*kyGridShift[iy]/(2.0l*m))*dt/h));
         long_double_complex psi=psiInit(xGrid[ix],yGrid[iy],a);
         // demote long double results to double
#ifdef _GPU
         expVGrid[in].x=expV.real();
         expVGrid[in].y=expV.imag();
         expTGrid[in].x=expT.real();
         expTGrid[in].y=expT.imag();
         psiPosGrid[in].x=(double)psi.real();
         psiPosGrid[in].y=(double)psi.imag();
         psiPosInitGrid[in].x=(double)psi.real();
         psiPosInitGrid[in].y=(double)psi.imag();
#else
         expVGrid[in][0]=(double)expV.real();
         expVGrid[in][1]=(double)expV.imag();
         expTGrid[in][0]=(double)expT.real();
         expTGrid[in][1]=(double)expT.imag();
         psiPosGrid[in][0]=(double)psi.real();
         psiPosGrid[in][1]=(double)psi.imag();
         psiPosInitGrid[in][0]=(double)psi.real();
         psiPosInitGrid[in][1]=(double)psi.imag();
#endif
      }
   }
 
#ifdef _GPU
   for(int it=0;it<NT;it++)
   {
      act[it].x=0.0;
      act[it].y=0.0;
   }
   cudaMemcpy(dev_psiPosGrid,psiPosGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_psiMomGrid,psiMomGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_psiPosInitGrid,psiPosInitGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_expTGrid,expTGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_expVGrid,expVGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_act,act,sizeNT,cudaMemcpyHostToDevice);
#endif
 
   fd=fopen("result_ini.dat","w");
   for(in=0;in<N;in+=100)
   {
#ifdef _GPU
      fprintf(fd,"GPU psiPosInitGrid[%i]=(%e,%e)\n",in,(double)psiPosInitGrid[in].x,(double)psiPosInitGrid[in].y);
#else
      fprintf(fd,"CPU psiPosInitGrid[%i]=(%e,%e)\n",in,(double)psiPosInitGrid[in][0],(double)psiPosInitGrid[in][1]);
#endif
   }
   fclose(fd);
 
   fprintf(stderr,"Initializing finished. Starting timer ...\n");
 
   c0=clock();
   for(it=0;it<NT;it++)
   {
#ifdef _GPU
      GPUMatrixElementMult<<<N/256,256>>>(dev_expVGrid,dev_psiPosGrid,dev_psiPosGrid,N,1.0);
      cudaThreadSynchronize();
      cufftExecZ2Z(plan, dev_psiPosGrid, dev_psiMomGrid, CUFFT_INVERSE);
      cudaThreadSynchronize();
      GPUMatrixElementMult<<<N/256,256>>>(dev_expTGrid,dev_psiMomGrid,dev_psiMomGrid,N,1.0/(double)N);
      cudaThreadSynchronize();
      cufftExecZ2Z(plan, dev_psiMomGrid, dev_psiPosGrid, CUFFT_FORWARD);
      cudaThreadSynchronize();
      GPUMatrixElementMult<<<N/256,256>>>(dev_psiPosGrid,dev_psiPosInitGrid,dev_psiCor,N,1.0);
      cudaThreadSynchronize();
      GPUSummation_parallel_partial<<<BLOCK_SIZE,BLOCK_SIZE>>>(dev_psiCor,dev_g_odata,(unsigned int)N);
      cudaThreadSynchronize();
      GPUSummation_serial<<<1,1>>>(dev_g_odata,dev_act,BLOCK_SIZE,it);
      cudaThreadSynchronize();
#else
      CPUMatrixElementMultScale(expVGrid,psiPosGrid,psiPosGrid,N,1.0);
      fftw_execute(fftplanAll);
      CPUMatrixElementMultScale(expTGrid,psiMomGrid,psiMomGrid,N,1.0/(double)N);
      fftw_execute(ifftplanAll);
      CPUMatrixElementMultScale(psiPosGrid,psiPosInitGrid,psiCor,N,1.0);
      CPUSummation(psiCor,act,N,it);
#endif
   }
   c1=clock();
 
   fprintf(stderr,"Propagation took %.2f s\n",(double)(c1-c0)/(double)CLOCKS_PER_SEC);
 
#ifdef _GPU
   cudaMemcpy(act,dev_act,sizeNT,cudaMemcpyDeviceToHost);
   cudaMemcpy(psiPosGrid,dev_psiPosGrid,sizeN,cudaMemcpyDeviceToHost);
 
   fd=fopen("result_gpu_act_dp.dat","w");
   // write recorded autocorrelation function at each time-step
   for(it=0;it<NT;it++)
   {
      fprintf(fd,"%e %e %e\n",(double)(it*dt),(double)(act[it].x*dx*dy),(double)(act[it].y*dx*dy));
   }
   fclose(fd);
#else
   fd=fopen("result_cpu_act_dp.dat","w");
   // write recorded autocorrelation function at each time-step
   for(it=0;it<NT;it++)
   {
      fprintf(fd,"%e %e %e\n",(double)(it*dt),(double)(act[it][0]*dx*dy),(double)(act[it][1]*dx*dy));
   }
   fclose(fd);
#endif
 
   // all memory frees are missing ...
 
   return 0;
}

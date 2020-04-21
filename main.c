#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <emmintrin.h>
#include "mpi.h"

struct timeval t1, t2;

/*
 * N, M, K are the global sizes, A matrix is N x M,
 *   B matrix is M x K
 *   C matrix is N x K
 * assuming that A and C matrices are stored in a row partition format
 *               B is stored in column partition format
 *   local matrix A is a N/nprocs x M
 *                B is a M x K/nprocs
 *                C is a N/procs x K
 * The last node (nprocs -1) has the leaf-over rows of the matrics
 */

void  my_mm6_sse2_mpi(int localN, int localK, int N, int M, int K, double *a, double *b, double *c)
{
  //A = localN*M
  //B = M*localK
  //C =localN*K

  //A goes from 0 - M in that go from 0 to local N
  //B goes from 0 - M in that go from 0 to local K
  //C goes from 0 - K int that go from 0 to locak N

  //base size to deal with on on Prosses Rank * blockSize (size of matix / num of prossess)

  int i, j, k, g;

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int pRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &pRank);


  for (g=0; g < world_size; g++){
    for (i=0; i < localK; i++){
      for (j=0; j < localN; j++){
        for (k=0; k < M; k++){
          *(c+ g*localK*localN + i*localN + j) = *(c+ g*localK*localN + i*localN + j) + *(a+j*M + k)*  *(b + k*localK + i);
  
          //think we might be fliping them
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
    //MPI_Send(,1, B)
    //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,MPI_Comm comm, MPI_Status * status)
    //MPI_Recv(,1, B)

    //receive from rank below and send to rank above
    //use if statment to check if curr rank = last rank then send to first rank
    //use if statment to check if curr rank - first rank then resive from last rank

    if (pRank == 0) { // first
      printf("prank 0 running\n");
      MPI_Send(b, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(b, 1, MPI_DOUBLE, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (pRank == world_size - 1) { // last
      printf("prank world_size - 1 running\n");
      MPI_Send(b,1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(b,1, MPI_DOUBLE, pRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else { // in between
      printf("prank in between running\n");
      MPI_Send(b,1, MPI_DOUBLE, pRank + 1, 0, MPI_COMM_WORLD);
      MPI_Recv(b,1, MPI_DOUBLE, pRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }


  // this is the routine that you must implement
}

int main( int argc, char *argv[])
{
  double *A, *B, *C, *W, *Z, *WORK;
  int  N, M, K, I, iter, i, j;
  int method;
  int myid, nprocs;
  int localN, localK;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc < 6) {
    if (myid == 0)
      printf("Usage: a.out N M K iter method\n");
    MPI_Finalize();
    exit(0);
  }

  N = atoi(argv[1]);
  M = atoi(argv[2]);
  K = atoi(argv[3]);
  iter = 1;
  if (argc >=5)
    iter = atoi(argv[4]);

  method = 0;
  if (argc >= 6)
    method = atoi(argv[5]);

  /*
  A = malloc(N*M*sizeof(double));
  B = malloc(M*K*sizeof(double));
  C = malloc(N*K*sizeof(double));

  W = malloc(M*K*sizeof(double)); // reorder B
  */


  if (myid != nprocs -1) {
    localN = N/nprocs;
    localK = K/nprocs;
  } else {
    localN = N - (N/nprocs*(nprocs-1));
    localK =  K-K/nprocs*(nprocs-1);
  }

  posix_memalign((void **)&A, 16, localN*M*sizeof(double));
  posix_memalign((void **)&B, 16, M*localK*sizeof(double));
  posix_memalign((void **)&C, 16, localN*K*sizeof(double));

  W = malloc(2*localN*K*sizeof(double));

  srand48(100+myid);

  for (i=0; i<localN*M; ++i) {
    //      A[i] = drand48();
    A[i] = 1.0;
    C[i] = 0.0;
  }
  for (i=0; i<M*localK; ++i) {
    //      B[i] = drand48();
    //      B[i] = myid*1.0;
    B[i] = 1.0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&t1, NULL);
  for (i=0; i<iter; i++) {
    if (method == 0) {
      //      printf("A[0][0] = %lf, B[0][0] = %lf\n", *A, *B);
      printf("calling mpi function...\n");
      my_mm6_sse2_mpi(localN, localK, N, M, K, A, B, C);
   } else {
      printf("Method not supported.\n");
      exit(0);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&t2, NULL);

  if (myid == 0)
    printf("Time for the matrix multiplication using method %d is %d milliseconds\n",
	 method,
         (t2.tv_sec - t1.tv_sec)*1000 +
         (t2.tv_usec - t1.tv_usec) / 1000);

#ifdef CHECK
  {

   if (myid == 0) {
     FILE *fd;
     if ((fd = fopen("tmp333", "w")) == NULL) {
       printf("Cannot open tmp333\n"); exit(0);
     }

     for (i=0; i<localN*K; i++)
       fprintf(fd, "%6.2lf\n", C[i]);
     for (i=1; i<nprocs; i++) {
       int size;
       if (i != nprocs -1) size = localN * K;
       else size = (N - N/nprocs *(nprocs-1)) *K;
       MPI_Recv(W, size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);

       for (j=0; j<size; j++)
	 fprintf(fd, "%6.2lf\n", W[j]);
     }
     fclose(fd);
   } else {
     int size;
     if (myid != nprocs -1) size = localN * K;
     else size = (N - N/nprocs *(nprocs-1)) *K;
     MPI_Send(C, size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
   }

  }
#endif
  MPI_Finalize();
  return 0;
}

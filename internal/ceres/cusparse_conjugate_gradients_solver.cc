#ifndef CERES_NO_CUDA

#include "ceres/cusparse_conjugate_gradients_solver.h"

#include "ceres/linear_solver.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

CusparseConjugateGradientsSolver::CusparseConjugateGradientsSolver(
    const LinearSolver::Options& options)
    : options_(options) {
  int devID = 0;
  cudaSetDevice(devID);

  cublasHandle = 0;
  cublasStatus = cublasCreate(&cublasHandle);

  cusparseHandle = 0;
  cusparseStatus = cusparseCreate(&cusparseHandle);
}

CusparseConjugateGradientsSolver::~CusparseConjugateGradientsSolver() {
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cudaDeviceReset();
}

LinearSolver::Summary CusparseConjugateGradientsSolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  const time_t start_time = time(NULL);
  int k, M = 0, N = 0, nz = 0;
  const int *I = NULL, *J = NULL;
  int *d_col, *d_row;
  double *y;
  double *d_val, *d_x;
  double *d_r, *d_p, *d_omega, *d_y;
  const double *val = NULL;
  double r0, r1, alpha, beta;
  const double tol = 1e-8f;
  const int max_iter = 100;
  const double doubleone = 1.0;
  const double doublezero = 0.0;
  double dot, nalpha;

  N = A->num_cols();
  M = A->num_rows();
  nz = (N-2)*3 + 4;
  I = A->rows(); //csr row pointers for matrix A
  J = A->cols(); // csr column indices for matrix A
  val = A->values(); // csr values for matrix A

  cublasStatus = cublasCreate(&cublasHandle);
  cusparseStatus = cusparseCreate(&cusparseHandle);

  /* Description of the A matrix*/
  cusparseMatDescr_t descr = 0;
  cusparseStatus = cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

  cudaMalloc((void**)&d_col, nz*sizeof(int));
  cudaMalloc((void**)&d_row, (N+1)*sizeof(int));
  cudaMalloc((void**)&d_val, nz*sizeof(double));
  cudaMalloc((void**)&d_x, N*sizeof(double));
  cudaMalloc((void**)&d_y, N*sizeof(double));
  cudaMalloc((void**)&d_r, N*sizeof(double));
  cudaMalloc((void**)&d_p, N*sizeof(double));
  cudaMalloc((void**)&d_omega, N*sizeof(double));

  cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, b, N*sizeof(double), cudaMemcpyHostToDevice);
  
  k = 0;
  r0 = 0;
  cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
  while (r1 > tol*tol && k <= max_iter) {
    k++;
    if ( k == 1 ) {
      cublasDcopy (cublasHandle, N, d_r, 1, d_p, 1);
    } else {
      beta = r1/r0;
      cublasDscal (cublasHandle, N, &beta, d_p, 1);
      cublasDaxpy (cublasHandle, N, &doubleone, d_r, 1, d_p, 1);
    }
    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &doubleone,
                        descr, d_val, d_row, d_col, d_p, &doublezero, d_omega);
    cublasDdot (cublasHandle, N, d_p, 1, d_omega, 1, &dot);
    alpha = r1/dot;
    cublasDaxpy (cublasHandle, N, &alpha, d_p, 1, d_x, 1);
	nalpha = -alpha;
    cublasDaxpy (cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
    r0 = r1;
    cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
  }

  cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_col);
  cudaFree(d_row);
  cudaFree(d_val);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_omega);

  LinearSolver::Summary summary;
  return summary;
}

}   // namespace internal
}   // namespace ceres

#endif  // CERES_NO_CUDA

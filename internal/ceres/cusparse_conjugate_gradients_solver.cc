#ifndef CERES_NO_CUDA

#include "ceres/cusparse_conjugate_gradients_solver.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "ceres/linear_solver.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

CusparseConjugateGradientsSolver::CusparseConjugateGradientsSolver(
    const LinearSolver::Options& options)
    : options_(options) {}

CusparseConjugateGradientsSolver::~CusparseConjugateGradientsSolver() {}

LinearSolver::Summary CusparseConjugateGradientsSolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  const time_t start_time = time(NULL);
  const int num_cols = A->num_cols();

  LinearSolver::Summary summary;
  
  return summary;
}

}   // namespace internal
}   // namespace ceres

#endif  // CERES_NO_CUDA

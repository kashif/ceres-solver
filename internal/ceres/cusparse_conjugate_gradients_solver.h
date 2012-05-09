// Preconditioned Conjugate Gradients based solver for positive
// semidefinite linear systems on the GPU using CUBLAS and CUSPARSE.

#ifndef CERES_INTERNAL_CUSPARSE_CONJUGATE_GRADIENTS_SOLVER_H_
#define CERES_INTERNAL_CUSPARSE_CONJUGATE_GRADIENTS_SOLVER_H_

#ifndef CERES_NO_CUDA

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "ceres/linear_solver.h"
#include "ceres/internal/macros.h"

namespace ceres {
namespace internal {

class CompressedRowSparseMatrix;

class CusparseConjugateGradientsSolver : public CompressedRowSparseMatrixSolver {
 public:
  explicit CusparseConjugateGradientsSolver(const LinearSolver::Options& options);
  virtual ~CusparseConjugateGradientsSolver();

 private:
  virtual LinearSolver::Summary SolveImpl(
      CompressedRowSparseMatrix* A,
      const double* b,
      const LinearSolver::PerSolveOptions& options,
      double* x);

  const LinearSolver::Options options_;
  cublasHandle_t cublasHandle;
  cublasStatus_t cublasStatus;
  cusparseHandle_t cusparseHandle;
  cusparseStatus_t cusparseStatus;
  DISALLOW_COPY_AND_ASSIGN(CusparseConjugateGradientsSolver);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CUDA

#endif  // CERES_INTERNAL_CUSPARSE_CONJUGATE_GRADIENTS_SOLVER_H_

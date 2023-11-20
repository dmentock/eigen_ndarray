#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <memory>

using namespace std;

template <typename TensorType, size_t M>
static bool recursive_compare(const TensorType& tensor1, const TensorType& tensor2, Eigen::array<Eigen::Index, M> index, int dim, double epsilon) {
  if (dim == M) {
    double diff = std::abs(tensor1(index) - tensor2(index));
    if (diff > epsilon) {
      double max_abs = std::max(std::abs(tensor1(index)), std::abs(tensor2(index)));
      double rel_diff = (max_abs > 0) ? diff / max_abs : 0;
      cout << "Mismatch at index [";
      for (int i = 0; i < M; ++i) {
        cout << index[i];
        if (i < M - 1) cout << ", ";
      }
      cout << "]: abs diff = " << diff << ", rel diff = " << rel_diff
          << ", values = " << tensor1(index) << " vs " << tensor2(index) << endl;
      return false;
    }
    return true;
  } else {
    for (Eigen::Index i = 0; i < tensor1.dimension(dim); ++i) {
      index[dim] = i;
      if (!recursive_compare(tensor1, tensor2, index, dim + 1, epsilon)) {
        return false;
      }
    }
    return true;
  }
}

template<typename T, size_t M, size_t N>
template<size_t K, size_t J>
bool Ndarray<T, M, N>::allclose(const Ndarray<T, K, J>& other, double epsilon) const {
  Tensor<T, M> tensor1 = this->get_array();
  Tensor<T, K> tensor2 = other.get_array();
  if (tensor1.dimensions() != tensor2.dimensions()) {
    throw std::runtime_error("Arrays to compare have a different dimension");
  }
  return recursive_compare(tensor1, tensor2, Eigen::array<Eigen::Index, M>(), 0, epsilon);
}
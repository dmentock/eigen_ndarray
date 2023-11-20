#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <memory>

template<typename T, size_t M, size_t N, size_t K, size_t J, typename Op>
Ndarray<T, M> applyOperation(const Ndarray<T, M, N>& a, const Ndarray<T, K, J>& b, Op op) {
  static_assert(M == K, "Operation only possible for Arrays with same dimensionality");
  if (!std::equal(a.indices.extents.begin(), a.indices.extents.end(), b.indices.extents.begin())) {
    throw std::runtime_error("Operation only possible for Arrays with same extents");
  }
  auto res = std::make_shared<Tensor<T, M>>(a.indices.extents);
  if (N == M && J == K) {
    *res = op(*a.base_array, *b.base_array);
  } else {
    *res = op(a.get_array(), b.get_array());
  }
  return Ndarray<T, M>(res);
}

// Function to apply operation between Ndarray and scalar
template<typename T, size_t M, size_t N, typename Op>
Ndarray<T, M> applyScalarOperation(const Ndarray<T, M, N>& a, T scalar, Op op) {
  auto res = std::make_shared<Tensor<T, M>>(a.indices.extents);
  if (N == M) {
    *res = op(*a.base_array, scalar);
  } else {
    *res = op(a.get_array(), scalar);
  }
  return Ndarray<T, M>(res);
}

// Free function for scalar-Ndarray algebraic operations
template<typename F, size_t K, size_t J>
Ndarray<F, K> operator*(F scalar, const Ndarray<F, K, J>& ndarray) {
  return applyScalarOperation(ndarray, scalar, std::multiplies<>());
}

template<typename F, size_t K, size_t J>
Ndarray<F, K> operator+(F scalar, const Ndarray<F, K, J>& ndarray) {
  return applyScalarOperation(ndarray, scalar, std::plus<>());
}

template<typename F, size_t K, size_t J>
Ndarray<F, K> operator-(F scalar, const Ndarray<F, K, J>& ndarray) {
  return applyScalarOperation(ndarray, scalar, std::minus<>());
}

template<typename F, size_t K, size_t J>
Ndarray<F, K> operator/(F scalar, const Ndarray<F, K, J>& ndarray) {
  return applyScalarOperation(ndarray, scalar, std::divides<>());
}

template<typename T, size_t M, size_t N>
T matmul_slice_vec(const Ndarray<T, M, N>* this_, const Ndarray<T, M, N>& other) {
  auto tensor1 = this_->get_array();
  auto tensor2 = other.get_array();
  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(0, 0)};
  Tensor<T, 0> resultTensor = tensor1.contract(tensor2, contract_dims).eval();
  return resultTensor(0);
}

template<typename T, size_t M, size_t N>
template<size_t K, size_t J>
typename std::enable_if<M == 1 && K == 1 && N == M && J == K, T>::type
Ndarray<T, M, N>::matmul(const Ndarray<T, K, J>& other) const {
  if (is_sliced) {
    // TODO: separate cases for this is/is not slice and other is/is not slice
    return matmul_slice_vec(this, other);
  } else {
    Map<const Matrix<T, Dynamic, 1>> vec1(this->base_array->data(), this->base_array->size());
    Map<const Matrix<T, Dynamic, 1>> vec2(other.base_array->data(), other.base_array->size());
    return vec1.dot(vec2);
  }
}

// Implement multidimensional matmul functionality
template<typename T, size_t M, size_t N>
template<size_t K, size_t J>
typename std::enable_if<M == 1 && K == 1 && !(N == M && J == K), T>::type
Ndarray<T, M, N>::matmul(const Ndarray<T, K, J>& other) const {
  return matmul_slice_vec(this, other);
}

template<typename T, size_t M, size_t N>
template<size_t K, size_t J>
typename std::enable_if<M == 2 && K == 2 && M == N && K == J, Ndarray<T, 2, 2>>::type
Ndarray<T, M, N>::matmul(const Ndarray<T, K, J>& other) const {
  Map<const Matrix<T, Dynamic, Dynamic>> mat1(this->base_array->data(), this->base_array->dimension(0), this->base_array->dimension(1));
  Map<const Matrix<T, Dynamic, Dynamic>> mat2(other.base_array->data(), other.base_array->dimension(0), other.base_array->dimension(1));
  auto res = std::make_shared<Tensor<T, 2>>(this->base_array->dimension(0), other.base_array->dimension(1));
  Map<Matrix<T, Dynamic, Dynamic>>(res->data(), this->base_array->dimension(0), other.base_array->dimension(1)) = mat1 * mat2;
  return Ndarray<T, 2, 2>(res);
}

// TODO: make templating for tensor contraction more fine-grained
template<typename T, size_t M, size_t N>
template<size_t K, size_t J>
typename std::enable_if<(M == 2 && K == 2 && !(M == N && K == J)) || (M > 2 && K > 2), Ndarray<T, 2, 2>>::type
Ndarray<T, M, N>::matmul(const Ndarray<T, K, J>& other) const {
  std::shared_ptr<Tensor<T, 2>> res = std::make_shared<Tensor<T, 2>>(indices.extents[1], other.indices.extents[0]);
  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
  *res = this->get_array().contract(other.get_array(), contract_dims);
  return Ndarray<T, 2, 2>(res);
}
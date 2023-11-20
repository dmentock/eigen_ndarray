#include <iostream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#pragma GCC diagnostic pop

#include <memory>
#include <iomanip>

using namespace Eigen;
using namespace std;

class R {
public:
  long start, end;
  R(long start, long end) : start(start), end(end) {
    if (end < start) {
      throw std::invalid_argument("Range end must be greater than or equal to start");
    }
  }
  long size() const {
    return end - start;
  }
};

template<size_t M, size_t N>
struct Indices {
    Eigen::array<Eigen::Index, M> offsets;
    Eigen::array<Eigen::Index, M> extents;
    std::array<std::pair<int, int>, N-M> chip_indices; // (dimension, row)
    bool is_sliced = false;
};

template<typename T, size_t M, size_t N=M>
class Ndarray {
public:
  using SharedTensor = std::shared_ptr<Tensor<T, N>>;
  SharedTensor base_array;
  Indices<M, N> indices;
  bool is_sliced = false;
  int size;

  Ndarray(std::initializer_list<size_t> dims)
    : base_array(std::make_shared<Eigen::Tensor<T, N>>(initializer_to_array<N>(dims))),
      indices(generate_full_indices(*base_array)),
      size(base_array->size()) {
        set_constant(0);
      }

  Ndarray(const SharedTensor& base)
    : base_array(base),
      indices(generate_full_indices(*base_array)),
      size(base_array->size()) {}

  Ndarray(const SharedTensor& base, Indices<M, N>& slice_indices)
    : base_array(base),
      indices(slice_indices),
      is_sliced(indices.is_sliced),
      size(1) {
        for (auto extent : indices.extents) size *= extent;
      }

  // get_array for non-sliced tensors
  template<size_t P = M, typename std::enable_if<N == P, int>::type = 0>
  auto get_array() const -> decltype(auto) {
      return base_array->slice(indices.offsets, indices.extents);
  }
  // get_array for sliced tensors
  template<size_t P = M, typename std::enable_if<N != P, int>::type = 0>
  auto get_array() const -> decltype(auto) {
    auto chipped_view = chip_tensor(*base_array, indices.chip_indices);
    return chipped_view.slice(indices.offsets, indices.extents);
  }

  int dimension(int dim) const {
    return indices.extents[dim];
  }

  void set_constant(T constant) {
    get_array().setConstant(constant);
  }

  T& operator()(const Eigen::Index& index) {
    return (*base_array)(index);
  }

  const T& operator()(const Eigen::Index& index) const {
    return (*base_array)(index);
  }

  template<typename... Indices>
  T& operator()(Indices... indices) {
    return (*base_array)(indices...);
  }

  template<typename... Indices>
  const T& operator()(Indices... indices) const {
    return (*base_array)(indices...);
  }

  template<typename... Slices>
  auto slice(const Slices&... slices);


  template<size_t K, size_t J>
  typename std::enable_if<M == 1 && K == 1 && N == M && J == K, T>::type
  matmul(const Ndarray<T, K, J>& other) const;

  template<size_t K, size_t J>
  typename std::enable_if<M == 1 && K == 1 && !(N == M && J == K), T>::type
  matmul(const Ndarray<T, K, J>& other) const;

  template<size_t K, size_t J>
  typename std::enable_if<M == 2 && K == 2 && M == N && K == J, Ndarray<T, 2, 2>>::type
  matmul(const Ndarray<T, K, J>& other) const;

  template<size_t K, size_t J>
  typename std::enable_if<(M == 2 && K == 2 && !(M == N && K == J)) || (M > 2 && K > 2), Ndarray<T, 2, 2>>::type
  matmul(const Ndarray<T, K, J>& other) const;

  Ndarray<T, M> operator+(const Ndarray<T, M, N>& other) const {
    return applyOperation(*this, other, std::plus<>());
  }

  Ndarray<T, M> operator-(const Ndarray<T, M, N>& other) const {
    return applyOperation(*this, other, std::minus<>());
  }

  Ndarray<T, M> operator*(const Ndarray<T, M, N>& other) const {
    return applyOperation(*this, other, std::multiplies<>());
  }

  Ndarray<T, M> operator/(const Ndarray<T, M, N>& other) const {
    return applyOperation(*this, other, std::divides<>());
  }

  // Overloaded operators for Ndarray-scalar operations
  Ndarray<T, M> operator+(T scalar) const {
    return applyScalarOperation(*this, scalar, std::plus<>());
  }

  Ndarray<T, M> operator-(T scalar) const {
    return applyScalarOperation(*this, scalar, std::minus<>());
  }

  Ndarray<T, M> operator*(T scalar) const {
    return applyScalarOperation(*this, scalar, std::multiplies<>());
  }

  Ndarray<T, M> operator/(T scalar) const {
    return applyScalarOperation(*this, scalar, std::divides<>());
  }

  template<typename F, size_t K, size_t J>
  friend Ndarray<F, K> operator+(F scalar, const Ndarray<F, K, J>& ndarray);

  template<typename F, size_t K, size_t J>
  friend Ndarray<F, K> operator-(F scalar, const Ndarray<F, K, J>& ndarray);

  template<typename F, size_t K, size_t J>
  friend Ndarray<F, K> operator*(F scalar, const Ndarray<F, K, J>& ndarray);

  template<typename F, size_t K, size_t J>
  friend Ndarray<F, K> operator/(F scalar, const Ndarray<F, K, J>& ndarray);


  template<size_t K, size_t J>
  bool allclose(const Ndarray<T, K, J>& other, double epsilon = 1e-8) const;

  void print(const std::string& label = "Slice", const std::string& style = "cpp") const;
  void print_base(const std::string& label = "Base", const std::string& style = "cpp") const;


private:
  template<typename TensorType>
  auto chip_tensor(TensorType &tensor, const std::array<std::pair<int, int>, N-M> &chip_indices) const;

  template <int J>
  Eigen::array<Eigen::Index, J> initializer_to_array(const std::initializer_list<size_t>& list) {
    Eigen::array<Eigen::Index, J> eigen_dims;
    std::copy_n(list.begin(), J, eigen_dims.begin());
    return eigen_dims;
  }

  Indices<N, N> generate_full_indices(Tensor<T, N>& base_array_) const {
    Indices<N, N> full_indices;
    for (int i = 0; i < N; ++i) {
      full_indices.offsets[i] = 0;
      full_indices.extents[i] = base_array_.dimension(i);
    }
    return full_indices;
  }
};

#include "compare.h"
#include "print.h"
#include "slice.h"
#include "algebra.h"
#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <memory>

template<int slice_dim, int base_dim, size_t P, size_t N, typename First, typename... Rest>
void process_slices(Indices<P, N>& indices_, Eigen::array<Eigen::Index, N>& base_dims, const First& first, const Rest&... rest) {
  if constexpr (std::is_same_v<std::decay_t<First>, R>) {
    indices_.offsets[slice_dim] = first.start;
    indices_.extents[slice_dim] = first.size();
    if (indices_.is_sliced == false && (first.start != 0 || indices_.extents[slice_dim] != base_dims[base_dim])) {
      indices_.is_sliced = true;
    }
    if constexpr (sizeof...(Rest) > 0) {
      process_slices<slice_dim + 1, base_dim + 1, P>(indices_, base_dims, rest...);
    }
  } else {
    size_t chipIndex = slice_dim - P;
    indices_.chip_indices[chipIndex] = std::make_pair(slice_dim, first);
    if constexpr (sizeof...(Rest) > 0) {
      process_slices<slice_dim, base_dim + 1, P>(indices_, base_dims, rest...);
    }
  }
}

template<typename... Slices>
static constexpr size_t count_R_objects() {
  return (... + (std::is_same_v<std::decay_t<Slices>, R> ? 1 : 0));
}

template<typename T, size_t M, size_t N>
template<typename... Slices>
auto Ndarray<T, M, N>::slice(const Slices&... slices) {
  static_assert(sizeof...(Slices) == N, "Number of slices must match tensor dimensions");
  constexpr size_t n_dims_slice = count_R_objects<Slices...>();
  Indices<n_dims_slice, N> indices_;
  Eigen::array<Eigen::Index, N> base_dims = base_array->dimensions();
  process_slices<0, 0, n_dims_slice>(indices_, base_dims, slices...);
  return Ndarray<T, n_dims_slice, N>(base_array, indices_);
}

template<typename TensorType, size_t I, size_t O=I>
auto chip_tensor_recursive(TensorType &tensor, const std::array<std::pair<int, int>, O> &chip_indices) {
  if constexpr (I == 0) {
    return tensor;
  } else {
    int dim = chip_indices[I - 1].first;
    int row = chip_indices[I - 1].second;
    auto chippedTensor = tensor.chip(row, dim);
    return chip_tensor_recursive<decltype(chippedTensor), I-1, O>(chippedTensor, chip_indices);
  }
}

template<typename T, size_t M, size_t N>
template<typename TensorType>
auto Ndarray<T, M, N>::chip_tensor(TensorType &tensor, const std::array<std::pair<int, int>, N-M> &chip_indices) const {
  return chip_tensor_recursive<TensorType, N-M>(tensor, chip_indices);
}
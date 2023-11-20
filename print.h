#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <memory>

std::vector<int> parsePrefix(const std::string& prefix) {
    std::vector<int> indices;
    std::stringstream ss(prefix);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            indices.push_back(std::stoi(item));
        }
    }
    return indices;
}

template<typename T, size_t M>
void print_cpp_style(const Eigen::Tensor<T, M>& tensor, int depth = 0, const std::string& prefix = "") {
    if (depth == M - 1) {
        // Base case: Print the elements of the last dimension
        std::cout << "{ ";
        for (int i = 0; i < tensor.dimension(depth); ++i) {
            Eigen::array<Eigen::Index, M> indices;
            auto parsedIndices = parsePrefix(prefix + std::to_string(i) + ",");
            std::copy_n(parsedIndices.begin(), M, indices.begin());
            std::cout << tensor.coeff(indices) << (i < tensor.dimension(depth) - 1 ? ", " : " }");
        }
    } else {
        // Recursive case: Iterate through this dimension and recurse
        std::cout << "{ ";
        for (int i = 0; i < tensor.dimension(depth); ++i) {
            print_cpp_style<T, M>(tensor, depth + 1, prefix + std::to_string(i) + ",");
            if (i < tensor.dimension(depth) - 1) std::cout << ", ";
        }
        std::cout << " }\n";
    }
}

template <typename T, size_t M>
void print_fortran_style(const Tensor<T, M>& tensor) {
  int total_elements = tensor.size();
  for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
      double value = tensor.coeff(linear_idx);
      int int_part = static_cast<int>(value);
      int num_digits = 1;
      while (int_part /= 10) ++num_digits;
      int precision = 18 - num_digits - 1;  // subtracting 1 for the dot
      if (linear_idx > 0) {
          if (value < 0) {
              std::cout  << "       ";
          } else {
              std::cout  << "        ";
          }
      }
      std::cout << std::setw(18) << std::fixed << std::setprecision(precision) << value;
  }
  std::cout << "\n";
}

template <typename T, size_t M>
void print_ndarray(const std::string& label, const Tensor<T, M>& tensor, const std::string& format = "cpp") {
    std::cout << label << ":\n";
    if (format == "cpp") {
        print_cpp_style<T, M>(tensor, 0, "");
    } else {
        print_fortran_style<T, M>(tensor);
    }
}

template<typename T, size_t M, size_t N>
void Ndarray<T, M, N>::print(const std::string& label, const std::string& style) const {
    Eigen::Tensor<T, M> tensor_slice = get_array().eval();
    print_ndarray<T, M>(label, tensor_slice, style);
}

template<typename T, size_t M, size_t N>
void Ndarray<T, M, N>::print_base(const std::string& label, const std::string& style) const {
    print_ndarray<T, M>(label, *base_array, style);
}


#include <gtest/gtest.h>
#include "ndarray.h" 

TEST(NdarrayTest, Initialization) {
  Ndarray<double, 3> testarray({3, 2, 1});
  EXPECT_EQ(testarray.dimension(0), 3);
  EXPECT_EQ(testarray.dimension(1), 2);
  EXPECT_EQ(testarray.dimension(2), 1);
}

TEST(NdarrayTest, SetZero) {
  Ndarray<double, 3> testarray({3, 2, 1});
  testarray.set_constant(0);
      for (int i = 0; i < testarray.size; ++i) {
      EXPECT_EQ(testarray.base_array->data()[i], 0.0);
  }
}

// Test setting values
TEST(NdarrayTest, SetValues) {
  Ndarray<double, 1> vec1({3});
  vec1.base_array->setValues({1.0, 2.0, 3.0});
  EXPECT_EQ((*vec1.base_array)(0), 1.0);
  EXPECT_EQ((*vec1.base_array)(1), 2.0);
  EXPECT_EQ((*vec1.base_array)(2), 3.0);
}

TEST(NdarrayTest, BraceAssignment) {
  Ndarray<double, 2> mat({2, 3});    
  mat(0,0) = 1;
  mat(1,1) = 2;

  Ndarray<double, 2> expected({2, 3});
  expected.base_array->setValues({
    {1, 0, 0},
    {0, 2, 0}
  });

  EXPECT_TRUE(mat.allclose(expected));
  EXPECT_DOUBLE_EQ(mat(1, 1), 2);
}

TEST(NdarrayTest, SlicedViewFullDims) {
  Ndarray<double, 2> mat({2, 3});
  mat.base_array->setValues({{1, 2, 3},
                             {4, 5, 6}});

  Ndarray<double, 2> full_range = mat.slice(R(0,2), R(0,3));
  EXPECT_FALSE(full_range.is_sliced);
  Ndarray<double, 1, 2> chipped_full_range = mat.slice(1, R(0,3));
  EXPECT_FALSE(chipped_full_range.is_sliced);
  Ndarray<double, 2> sliced = mat.slice(R(0,1), R(0,3));
  EXPECT_TRUE(sliced.is_sliced);
  Ndarray<double, 1, 2> sliced_and_chipped = mat.slice(R(0,1), 2);
  EXPECT_TRUE(sliced_and_chipped.is_sliced);
}

TEST(NdarrayTest, SlicedViewNoCopy) {
  Ndarray<double, 2> mat({2, 3});    
  mat.base_array->setValues({{1, 2, 3},
                             {4, 5, 6}});
  Ndarray<double, 1, 2> vec = mat.slice(0, R(0,2));

  vec(0) = 7;

  Ndarray<double, 2> expected_mat({2, 3});
  expected_mat.base_array->setValues({{7, 2, 3},
                                      {4, 5, 6}});
  EXPECT_TRUE(mat.allclose(expected_mat));
                                      
  mat(0, 1) = 8;
  Ndarray<double, 1> expected_vec({2});
  expected_vec.base_array->setValues({7, 8});
  EXPECT_TRUE(vec.allclose(expected_vec));
}

TEST(NdarrayTest, ScalarMultiplicationOnBase) {
  Ndarray<double, 3> tensor({1, 2, 2});
  tensor.base_array->setValues({{{1, 2},
                                 {3, 4}}});
  double scalar = 5;
  Ndarray<double, 3> expected({1, 2, 2});
  expected.base_array->setValues({{{ 5, 10},
                                   {15, 20}}});

  Ndarray<double, 3> result = tensor * scalar;
  EXPECT_TRUE(result.allclose(expected));

  Ndarray<double, 3> result_flipped = scalar * tensor;
  EXPECT_TRUE(result_flipped.allclose(expected));
}

TEST(NdarrayTest, ScalarMultiplicationOnSlice) {
  Ndarray<double, 3> tensor({3, 3, 3});  
  tensor.base_array->setValues({
    {{ 0,  1,  2}, { 3,  4,  5}, { 6,  7,  8}},
    {{ 9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
    {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}                
  });
  Ndarray<double, 2, 3> mat = tensor.slice(R(0,2), R(0,3), 0);

  double scalar = 5;
  Ndarray<double, 2> expected({2, 3});
  expected.base_array->setValues({{ 0, 15, 30}, { 45, 60, 75}});

  Ndarray<double, 2> result = mat * scalar;
  EXPECT_TRUE(result.allclose(expected));

  Ndarray<double, 2> result_flipped = scalar * mat;
  EXPECT_TRUE(result_flipped.allclose(expected));
}

TEST(NdarrayTest, TensorMultiplicationOnBase) {
  Ndarray<double, 3> tensor1({1, 2, 2});
  Ndarray<double, 3> tensor2({1, 2, 2});

  tensor1.base_array->setValues({{{1, 2},
                                  {3, 4}}});
  tensor2.base_array->setValues({{{2, 4,},
                                  {6, 8,}}});

  Ndarray<double, 3> result = tensor1 * tensor2;

  Ndarray<double, 3> expected({1, 2, 2});
  expected.base_array->setValues({{{2,  8},
                                  {18, 32}}});

  EXPECT_TRUE(result.allclose(expected));
}

TEST(NdarrayTest, TensorMultiplicationOnSlice) {
  Ndarray<double, 4> tensor1({2, 2, 2, 2});
  tensor1.base_array->setValues({
    {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}},
    {{{8, 9}, {10, 11}}, {{12, 13}, {14, 15}}}
  });
  Ndarray<double, 3, 4> tensor2 = tensor1.slice(R(0,2), R(0,2), R(0,2), 0);
  Ndarray<double, 3, 4> tensor3 = tensor1.slice(0, R(0,2), R(0,2), R(0,2));
  auto result = tensor2 * tensor3;

  Ndarray<double, 3> expected({2, 2, 2});
  expected.base_array->setValues({
    {{  0,  2}, { 8, 18}}, 
    {{ 32, 50}, {72, 98}}
  });

  // [[[ 8, 9 ], [ 10, 11 ]], [[ 12, 13 ], [ 14, 15]]] * [[[ 0, 2], [ 4, 6]], [[ 8, 10], [ 12, 14]]]
  EXPECT_TRUE(result.allclose(expected));
}

TEST(NdarrayTest, VectorDotProductOnBase) {
  Ndarray<double, 1> vec1({3});
  Ndarray<double, 1> vec2({3});
  
  vec1.base_array->setValues({1.0, 2.0, 3.0});
  vec2.base_array->setValues({4.0, 5.0, 6.0});

  EXPECT_DOUBLE_EQ(vec1.matmul(vec2), 32);
}

TEST(NdarrayTest, VectorDotProductOnSlice) {
  Ndarray<double, 2> mat1({2, 3});
  Ndarray<double, 2> mat2({3, 2});
  
  mat1.base_array->setValues({{1.0, 2.0, 3.0},
                              {4.0, 5.0, 6.0}});
  mat2.base_array->setValues({{7.0, 8.0},
                              {9.0, 10.0},
                              {11.0, 12.0}});

  Ndarray<double, 1, 2> vec1 = mat1.slice(0, R(0,3));
  Ndarray<double, 1, 2> vec2 = mat2.slice(R(0,3), 1);
  /// [1, 2, 3] * [8, 10, 12]
  EXPECT_DOUBLE_EQ(vec1.matmul(vec2), 64);
}

TEST(NdarrayTest, MatrixMultiplicationOnBase) {
  Ndarray<double, 2> mat1({2, 3});
  Ndarray<double, 2> mat2({3, 2});
  
  mat1.base_array->setValues({{1.0, 2.0, 3.0},
                              {4.0, 5.0, 6.0}});
  mat2.base_array->setValues({{7.0, 8.0},
                              {9.0, 10.0},
                              {11.0, 12.0}});
  Ndarray<double, 2> result = mat1.matmul(mat2);

  Ndarray<double, 2> expected({2, 2});
  expected.base_array->setValues({
    {58, 64},
    {139, 154}
  });

  EXPECT_TRUE(result.allclose(expected));
}

TEST(NdarrayTest, MatrixMultiplicationOnSlice) {
  Ndarray<double, 3> tensor({3, 3, 3});  
  tensor.base_array->setValues({
    {{ 0,  1,  2}, { 3,  4,  5}, { 6,  7,  8}},
    {{ 9, 10, 11}, {12, 13, 14}, {15, 16, 17}},
    {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}                
  });

  Ndarray<double, 2, 3> mat1 = tensor.slice(R(0,2),R(0,3), 0);
  Ndarray<double, 2, 3> mat2 = tensor.slice(R(0,3),R(0,2), 1);
  Ndarray<double, 2, 2> result = mat1.matmul(mat2);

  Ndarray<double, 2> expected({2, 2});
  expected.base_array->setValues({
    {144, 171},
    {414, 522}
  });
  // [[0, 3, 6], [9,12,15]] * [[1, 4], [10, 13], [19, 22]]
  EXPECT_TRUE(result.allclose(expected));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

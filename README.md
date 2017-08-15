## Benchmarks

Mainly to test the performance of the [OpenBlas](https://github.com/xianyi/OpenBLAS), [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [NNPACK](https://github.com/Maratyszcza/NNPACK), [ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) library's matrix multiplication function and convolution function on various devices.

### [Matrix Multiplication](https://github.com/hedaoyuan/Function/blob/master/src/matmul/README.md)
Compare the matrix multiplication performance of Eigen and OpenBlas library.

ARMv7 environment the matrix multiplication of Eigen faster than OpenBlas, ARMv8 environment the matrix multiplication of OpenBlas faster than Eigen.


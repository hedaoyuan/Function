## Benchmarks

Mainly to test the performance of the [OpenBlas](https://github.com/xianyi/OpenBLAS), [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [NNPACK](https://github.com/Maratyszcza/NNPACK), [ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) library's matrix multiplication function and convolution function on various devices.

### [Matrix Multiplication](https://github.com/hedaoyuan/Function/blob/master/src/matmul/README.md)
Compare the matrix multiplication performance of Eigen and OpenBlas library.

### [Depthwise Convolution](https://github.com/hedaoyuan/Function/blob/master/src/conv/README.md)
For the depthwise convolution in mobilenet, the performance of the NeonDepthwiseConv(a convolution optimization based on ARM-Neon instruction) is 7-10 times higher than the GemmConv based on matrix multiplication(use openblas).


## Benchmarks

Mainly to test the performance of the [OpenBlas](https://github.com/xianyi/OpenBLAS), [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [NNPACK](https://github.com/Maratyszcza/NNPACK), [ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) library's matrix multiplication function and convolution function on various devices.

### Convolution
Compare convolution function performance based on various implementations, such as OpenBlas based GemmConv, NNPACK based NNPACKConv Functions.
```
----------------------------------------------------------------------------------------------------
Benchmark                                             Time           CPU Iterations UserCounters...
----------------------------------------------------------------------------------------------------
BM_Convolution/GemmConv-CPU/3/64/108/3/1/1         6755 us       6662 us        102 gflops=5.63496G/s
BM_Convolution/GemmConv-CPU/64/64/54/3/1/1        24752 us      24445 us         28 gflops=8.19091G/s
BM_Convolution/GemmConv-CPU/64/128/54/3/1/1       43815 us      43238 us         16 gflops=9.26158G/s
BM_Convolution/GemmConv-CPU/128/128/27/3/1/1      23365 us      22961 us         30 gflops=8.72014G/s
BM_Convolution/GemmConv-CPU/64/128/54/1/1/0        6437 us       6302 us        125 gflops=7.06061G/s
BM_Convolution/GemmConv-CPU/128/128/27/3/1/1      23494 us      23096 us         30 gflops=8.66927G/s
BM_Convolution/GemmConv-CPU/128/256/27/3/1/1      42800 us      42086 us         17 gflops=9.51517G/s
BM_Convolution/GemmConv-CPU/256/256/14/3/1/1      22863 us      22459 us         32 gflops=9.58772G/s
BM_Convolution/GemmConv-CPU/128/256/27/1/1/0       5766 us       5634 us        121 gflops=7.8982G/s
BM_Convolution/GemmConv-CPU/256/256/14/3/1/1      22714 us      22333 us         31 gflops=9.64186G/s
BM_Convolution/GemmConv-CPU/256/512/14/3/1/1      44780 us      44008 us         16 gflops=9.78604G/s
BM_Convolution/GemmConv-CPU/512/512/7/3/1/1       26900 us      26407 us         27 gflops=8.1543G/s
BM_Convolution/GemmConv-CPU/256/512/14/1/1/0       5406 us       5303 us        134 gflops=9.02427G/s
BM_Convolution/GemmConv-CPU/512/512/7/3/1/1       26803 us      26287 us         27 gflops=8.19143G/s
```
```
---------------------------------------------------------------------------------------------------------------------
Benchmark                                                              Time           CPU Iterations UserCounters...
---------------------------------------------------------------------------------------------------------------------
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/3/32/192/3/2/1         3112 us       3063 us        228 gflops=4.84295G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/3/32/224/3/2/1         4278 us       4211 us        166 gflops=4.79359G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/3/32/300/3/2/1         7854 us       7725 us         90 gflops=4.68737G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/3/64/108/3/2/1         1498 us       1477 us        474 gflops=6.35344G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/64/64/54/3/1/1        33737 us      33223 us         21 gflops=6.02671G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/64/128/54/3/1/1       54427 us      53494 us         13 gflops=7.48586G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/128/128/27/3/1/1      31663 us      30783 us         23 gflops=6.50435G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/128/256/27/3/1/1      56230 us      54696 us         13 gflops=7.32142G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/256/256/14/3/1/1      32067 us      31148 us         22 gflops=6.9132G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/256/512/14/3/1/1      64058 us      62136 us         12 gflops=6.93105G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/512/512/7/3/1/1       50530 us      49052 us         13 gflops=4.38986G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/64/64/54/3/1/1                 7040 us       6940 us        100 gflops=28.853G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/64/128/54/3/1/1               13051 us      12891 us         54 gflops=31.0642G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/128/128/27/3/1/1              13625 us      13422 us         54 gflops=14.9181G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/128/256/27/3/1/1              27190 us      26816 us         26 gflops=14.9332G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/256/256/14/3/1/1              17936 us      17719 us         39 gflops=12.1525G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/256/512/14/3/1/1              36919 us      36467 us         19 gflops=11.8096G/s
BM_Convolution/NNPACKConv-CPU"wt8x8"/512/512/7/3/1/1               64446 us      63677 us         11 gflops=3.38165G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/64/128/54/1/1/0        6134 us       6009 us        116 gflops=7.40477G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/128/256/27/1/1/0       5366 us       5240 us        137 gflops=8.4921G/s
BM_Convolution/NNPACKConv-CPU"implicit-gemm"/256/512/14/1/1/0       6295 us       6129 us        117 gflops=7.80745G/s
```

### [Matrix Multiplication](https://github.com/hedaoyuan/Function/blob/master/src/matmul/README.md)
Compare the matrix multiplication performance of Eigen and OpenBlas library.

[**TODO**] Add a matrix multiplication function based on ARM ComputeLibrary and its performance test.

### [Depthwise Convolution](https://github.com/hedaoyuan/Function/blob/master/src/conv/README.md)
For the depthwise convolution in mobilenet, the performance of the NeonDepthwiseConv(a convolution optimization based on ARM-Neon instruction) is **7-10 times** higher than the GemmConv based on matrix multiplication(use openblas).


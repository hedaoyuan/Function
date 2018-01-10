## Benchmarks

Mainly to test the performance of the [OpenBlas](https://github.com/xianyi/OpenBLAS), [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [NNPACK](https://github.com/Maratyszcza/NNPACK), [ComputeLibrary](https://github.com/ARM-software/ComputeLibrary) library's matrix multiplication function and convolution function on various devices.

### Convolution
Compare convolution function performance based on various implementations, such as OpenBlas based GemmConv, NNPACK based NNPACKConv Functions.
```
Stat=GemmConv-CPU/64/64/54/3/1/1    total=414.027  avg=20.701   max=20.956   min=20.563   count=20  gflops=10.386
Stat=GemmConv-CPU/64/128/54/3/1/1   total=787.517  avg=39.375   max=40.723   min=38.644   count=20  gflops=10.92
Stat=GemmConv-CPU/128/128/27/3/1/1  total=397.255  avg=19.862   max=20.455   min=19.692   count=20  gflops=10.824
Stat=GemmConv-CPU/128/256/27/3/1/1  total=791.64   avg=39.582   max=41.642   min=38.299   count=20  gflops=10.863
Stat=GemmConv-CPU/256/256/14/3/1/1  total=428.59   avg=21.429   max=22.277   min=20.497   count=20  gflops=10.79
Stat=GemmConv-CPU/256/512/14/3/1/1  total=834.855  avg=41.742   max=43.073   min=40.486   count=20  gflops=11.078
Stat=GemmConv-CPU/512/512/7/3/1/1   total=511.348  avg=25.567   max=25.877   min=24.856   count=20  gflops=9.043
Stat=GemmConv-CPU/16/32/375/3/1/1   total=2802.19  avg=140.109  max=140.252  min=139.925  count=20  gflops=9.25
Stat=GemmConv-CPU/16/32/188/3/1/1   total=680.885  avg=34.044   max=34.181   min=33.907   count=20  gflops=9.568
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


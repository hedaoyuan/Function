## Benchmark of DepthwiseConvolution

Environment: MI 5, Android 7.0, Snapdragon 820 1.8GHz

### Compiled as ARMv8 version (gcc version 4.9.x)
```
-------------------------------------------------------------------------------------------------------------------
Benchmark                                                            Time           CPU Iterations UserCounters...
-------------------------------------------------------------------------------------------------------------------
BM_Convolution/NeonDepthwiseConv-CPU/32/32/96/3/1/1/32            1038 us       1028 us        677 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/64/64/96/3/2/1/64            1504 us       1490 us        460 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/128/128/48/3/1/1/128         1072 us       1061 us        674 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/128/128/48/3/2/1/128          647 us        640 us       1097 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/256/256/24/3/1/1/256          586 us        579 us       1259 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/256/256/24/3/2/1/256          400 us        396 us       1832 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/512/512/12/3/1/1/512          334 us        331 us       2102 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/512/512/12/3/1/1/512          348 us        345 us       2109 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/512/512/12/3/1/1/512          349 us        346 us       2033 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/512/512/12/3/1/1/512          346 us        343 us       2036 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/512/512/12/3/1/1/512          355 us        351 us       2062 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/512/512/12/3/2/1/512          261 us        258 us       2704 gflops=0/s
BM_Convolution/NeonDepthwiseConv-CPU/1024/1024/6/3/1/1/1024        273 us        270 us       2589 gflops=0/s
-------------------------------------------------------------------------------------------------------------------
BM_Convolution/GemmConv-CPU/32/32/96/3/1/1/32                    11232 us      10968 us         65 gflops=0/s
BM_Convolution/GemmConv-CPU/64/64/96/3/2/1/64                     6073 us       5964 us        104 gflops=0/s
BM_Convolution/GemmConv-CPU/128/128/48/3/1/1/128                 12078 us      11771 us         60 gflops=0/s
BM_Convolution/GemmConv-CPU/128/128/48/3/2/1/128                  4241 us       4145 us        173 gflops=0/s
BM_Convolution/GemmConv-CPU/256/256/24/3/1/1/256                  6011 us       5941 us        109 gflops=0/s
BM_Convolution/GemmConv-CPU/256/256/24/3/2/1/256                  1570 us       1549 us        458 gflops=0/s
BM_Convolution/GemmConv-CPU/512/512/12/3/1/1/512                  2956 us       2923 us        239 gflops=0/s
BM_Convolution/GemmConv-CPU/512/512/12/3/1/1/512                  2890 us       2866 us        245 gflops=0/s
BM_Convolution/GemmConv-CPU/512/512/12/3/1/1/512                  2936 us       2905 us        243 gflops=0/s
BM_Convolution/GemmConv-CPU/512/512/12/3/1/1/512                  2960 us       2929 us        244 gflops=0/s
BM_Convolution/GemmConv-CPU/512/512/12/3/1/1/512                  2926 us       2898 us        244 gflops=0/s
BM_Convolution/GemmConv-CPU/512/512/12/3/2/1/512                  1047 us       1040 us        635 gflops=0/s
BM_Convolution/GemmConv-CPU/1024/1024/6/3/1/1/1024                2068 us       2051 us        345 gflops=0/s
```

### Compiled as ARMv7 version


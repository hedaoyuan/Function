
## Benchmark of MatMul

Environment: MI 5, Android 7.0, Snapdragon 820 1.8GHz, gcc 4.9.x

### Compiled as ARMv8 version

```
----------------------------------------------------------------------------------------
Benchmark                                 Time           CPU Iterations UserCounters...
----------------------------------------------------------------------------------------
BM_MatMul/Eigen-CPU/32/32/32             15 us         15 us      48345 gflops=4.20593G/s
BM_MatMul/Eigen-CPU/64/64/64            111 us        111 us       6317 gflops=4.4067G/s
BM_MatMul/Eigen-CPU/96/96/96            338 us        336 us       2090 gflops=4.89865G/s
BM_MatMul/Eigen-CPU/128/128/128         869 us        864 us        819 gflops=4.5204G/s
BM_MatMul/Eigen-CPU/256/256/256        7268 us       7088 us         98 gflops=4.40898G/s
BM_MatMul/Eigen-CPU/384/384/384       20277 us      19883 us         35 gflops=5.30455G/s
BM_MatMul/Eigen-CPU/512/512/512       47354 us      46563 us         15 gflops=5.36908G/s
BM_MatMul/Eigen-CPU/64/9216/32         9236 us       9038 us         77 gflops=3.88968G/s
BM_MatMul/Eigen-CPU/128/2304/64        8457 us       8228 us         87 gflops=4.27252G/s
BM_MatMul/Eigen-CPU/128/2304/128      15879 us      15502 us         47 gflops=4.53567G/s
BM_MatMul/Eigen-CPU/256/576/128        8044 us       7822 us         92 gflops=4.49448G/s
BM_MatMul/Eigen-CPU/256/576/256       15506 us      15105 us         46 gflops=4.65484G/s
BM_MatMul/Eigen-CPU/512/144/256        9117 us       8822 us         79 gflops=3.98524G/s
BM_MatMul/Eigen-CPU/512/144/512       17687 us      17171 us         42 gflops=4.09479G/s
BM_MatMul/Eigen-CPU/1024/36/512       10051 us       9859 us         70 gflops=3.56579G/s
BM_MatMul/Eigen-CPU/1024/36/1024      22937 us      22351 us         32 gflops=3.14578G/s
BM_MatMul/Eigen-CPU/9/128/256           172 us        171 us       4132 gflops=3.2144G/s
BM_MatMul/Eigen-CPU/16/64/256           106 us        106 us       6507 gflops=4.60688G/s
BM_MatMul/Eigen-CPU/48/64/256           313 us        312 us       2256 gflops=4.69598G/s
BM_MatMul/Eigen-CPU/48/96/64            115 us        115 us       6226 gflops=4.7835G/s
BM_MatMul/Eigen-CPU/48/104/64           127 us        126 us       5660 gflops=4.71362G/s
BM_MatMul/Eigen-CPU/64/96/64            152 us        152 us       4654 gflops=4.83112G/s
BM_MatMul/Eigen-CPU/64/104/64           168 us        167 us       4248 gflops=4.74641G/s
BM_MatMul/Eigen-CPU/128/128/256        1746 us       1722 us        404 gflops=4.5361G/s
----------------------------------------------------------------------------------------
BM_MatMul/Blas-CPU/32/32/32              10 us         10 us      71877 gflops=6.21705G/s
BM_MatMul/Blas-CPU/64/64/64              53 us         53 us      13080 gflops=9.19257G/s
BM_MatMul/Blas-CPU/96/96/96             161 us        160 us       4373 gflops=10.298G/s
BM_MatMul/Blas-CPU/128/128/128          354 us        353 us       1984 gflops=11.0705G/s
BM_MatMul/Blas-CPU/256/256/256         3149 us       3096 us        218 gflops=10.094G/s
BM_MatMul/Blas-CPU/384/384/384        12571 us      12214 us         57 gflops=8.6353G/s
BM_MatMul/Blas-CPU/512/512/512        28202 us      27550 us         26 gflops=9.07451G/s
BM_MatMul/Blas-CPU/64/9216/32          6271 us       6159 us        121 gflops=5.70788G/s
BM_MatMul/Blas-CPU/128/2304/64         4851 us       4734 us        159 gflops=7.42624G/s
BM_MatMul/Blas-CPU/128/2304/128        8213 us       8019 us         94 gflops=8.76875G/s
BM_MatMul/Blas-CPU/256/576/128         4006 us       3912 us        181 gflops=8.98696G/s
BM_MatMul/Blas-CPU/256/576/256         8256 us       8021 us         85 gflops=8.76613G/s
BM_MatMul/Blas-CPU/512/144/256         4170 us       4073 us        168 gflops=8.63195G/s
BM_MatMul/Blas-CPU/512/144/512         8568 us       8384 us         87 gflops=8.38656G/s
BM_MatMul/Blas-CPU/1024/36/512         5292 us       5198 us        100 gflops=6.76316G/s
BM_MatMul/Blas-CPU/1024/36/1024       16942 us      16481 us         42 gflops=4.26638G/s
BM_MatMul/Blas-CPU/9/128/256             87 us         87 us       8050 gflops=6.31739G/s
BM_MatMul/Blas-CPU/16/64/256             57 us         57 us      12235 gflops=8.52942G/s
BM_MatMul/Blas-CPU/48/64/256            143 us        142 us       4914 gflops=10.2907G/s
BM_MatMul/Blas-CPU/48/96/64              56 us         56 us      12501 gflops=9.79802G/s
BM_MatMul/Blas-CPU/48/104/64             61 us         61 us      11464 gflops=9.74602G/s
BM_MatMul/Blas-CPU/64/96/64              73 us         73 us       9575 gflops=10.0285G/s
BM_MatMul/Blas-CPU/64/104/64             80 us         79 us       8816 gflops=9.98762G/s
BM_MatMul/Blas-CPU/128/128/256          710 us        708 us        980 gflops=11.0422G/s
```

### Compiled as ARMv7 version
```
----------------------------------------------------------------------------------------
Benchmark                                 Time           CPU Iterations UserCounters...
----------------------------------------------------------------------------------------
BM_MatMul/Eigen-CPU/32/32/32             14 us         14 us      48122 gflops=4.22552G/s
BM_MatMul/Eigen-CPU/64/64/64             92 us         92 us       7610 gflops=5.31079G/s
BM_MatMul/Eigen-CPU/96/96/96            251 us        250 us       2796 gflops=6.58182G/s
BM_MatMul/Eigen-CPU/128/128/128         732 us        726 us        963 gflops=5.37885G/s
BM_MatMul/Eigen-CPU/256/256/256        5999 us       5823 us        120 gflops=5.36677G/s
BM_MatMul/Eigen-CPU/384/384/384       18239 us      17845 us         40 gflops=5.91026G/s
BM_MatMul/Eigen-CPU/512/512/512       43444 us      42685 us         16 gflops=5.8569G/s
BM_MatMul/Eigen-CPU/64/9216/32         7458 us       7319 us         95 gflops=4.80335G/s
BM_MatMul/Eigen-CPU/128/2304/64        6877 us       6706 us        111 gflops=5.24257G/s
BM_MatMul/Eigen-CPU/128/2304/128      12580 us      12349 us         59 gflops=5.69358G/s
BM_MatMul/Eigen-CPU/256/576/128        6570 us       6418 us        106 gflops=5.47787G/s
BM_MatMul/Eigen-CPU/256/576/256       12599 us      12308 us         59 gflops=5.71257G/s
BM_MatMul/Eigen-CPU/512/144/256        6956 us       6781 us        104 gflops=5.18489G/s
BM_MatMul/Eigen-CPU/512/144/512       13702 us      13367 us         53 gflops=5.26032G/s
BM_MatMul/Eigen-CPU/1024/36/512        8069 us       7930 us         90 gflops=4.43345G/s
BM_MatMul/Eigen-CPU/1024/36/1024      16706 us      16310 us         44 gflops=4.31092G/s
BM_MatMul/Eigen-CPU/9/128/256           158 us        157 us       4481 gflops=3.49667G/s
BM_MatMul/Eigen-CPU/16/64/256            91 us         90 us       7755 gflops=5.41487G/s
BM_MatMul/Eigen-CPU/48/64/256           256 us        255 us       2748 gflops=5.75435G/s
BM_MatMul/Eigen-CPU/48/96/64             87 us         87 us       8066 gflops=6.33591G/s
BM_MatMul/Eigen-CPU/48/104/64           105 us        105 us       6669 gflops=5.67284G/s
BM_MatMul/Eigen-CPU/64/96/64            115 us        115 us       6111 gflops=6.39488G/s
BM_MatMul/Eigen-CPU/64/104/64           139 us        138 us       5055 gflops=5.7293G/s
BM_MatMul/Eigen-CPU/128/128/256        1418 us       1398 us        497 gflops=5.58857G/s
----------------------------------------------------------------------------------------
BM_MatMul/Blas-CPU/32/32/32              16 us         16 us      45062 gflops=3.93605G/s
BM_MatMul/Blas-CPU/64/64/64             105 us        104 us       6694 gflops=4.67706G/s
BM_MatMul/Blas-CPU/96/96/96             338 us        337 us       2079 gflops=4.89398G/s
BM_MatMul/Blas-CPU/128/128/128          790 us        786 us        894 gflops=4.97135G/s
BM_MatMul/Blas-CPU/256/256/256         6678 us       6570 us        106 gflops=4.75648G/s
BM_MatMul/Blas-CPU/384/384/384        23961 us      23240 us         30 gflops=4.53834G/s
BM_MatMul/Blas-CPU/512/512/512        56427 us      54950 us         13 gflops=4.54962G/s
BM_MatMul/Blas-CPU/64/9216/32          8192 us       8043 us         87 gflops=4.37102G/s
BM_MatMul/Blas-CPU/128/2304/64         7750 us       7560 us         91 gflops=4.65043G/s
BM_MatMul/Blas-CPU/128/2304/128       15363 us      14948 us         47 gflops=4.70388G/s
BM_MatMul/Blas-CPU/256/576/128         7636 us       7459 us         93 gflops=4.71323G/s
BM_MatMul/Blas-CPU/256/576/256        15533 us      15103 us         46 gflops=4.65553G/s
BM_MatMul/Blas-CPU/512/144/256         8037 us       7828 us         89 gflops=4.49128G/s
BM_MatMul/Blas-CPU/512/144/512        16949 us      16461 us         42 gflops=4.27156G/s
BM_MatMul/Blas-CPU/1024/36/512         9216 us       9008 us         80 gflops=3.9028G/s
BM_MatMul/Blas-CPU/1024/36/1024       17896 us      17487 us         40 gflops=4.02079G/s
BM_MatMul/Blas-CPU/9/128/256            142 us        141 us       4967 gflops=3.88525G/s
BM_MatMul/Blas-CPU/16/64/256            105 us        104 us       6707 gflops=4.67733G/s
BM_MatMul/Blas-CPU/48/64/256            301 us        300 us       2332 gflops=4.8795G/s
BM_MatMul/Blas-CPU/48/96/64             116 us        116 us       6052 gflops=4.75081G/s
BM_MatMul/Blas-CPU/48/104/64            125 us        125 us       5607 gflops=4.76675G/s
BM_MatMul/Blas-CPU/64/96/64             154 us        154 us       4567 gflops=4.76389G/s
BM_MatMul/Blas-CPU/64/104/64            166 us        166 us       4217 gflops=4.79248G/s
BM_MatMul/Blas-CPU/128/128/256         1569 us       1563 us        449 gflops=4.99967G/s
```

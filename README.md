BitGPU is implemented in Ubuntu 14.04.

#### Prerequisites
To use the code, please install and configure:
- [Nvidia CUDA-7.5](https://developer.nvidia.com/cuda-downloads). Our code works with CUDA version 7.5. We have not tested other versions.
- A modified version of [Gappa++](https://github.com/YeDeheng/gappa), a tool for verifying numerical properties.

#### User Guide
1. Compile the GPU code and ASA (adaptive simulated annealing) code:

    ```sh
    $ make
    $ cd ./asa & make & cd ..
    ```

2. Compile the C-style benchmarks into DFG-style assembly code using gcc's GIMPLE backend:

    ```sh
    $ ./prep_bench.sh
    ```
3. Prune the search space for range analysis and bitwidth allocation:

    ```sh
    $ ./prune.sh
    ```

4. To perform range analysis on GPU and CPU:

    ```sh
    $ ./range.sh    % this script invokes GPU range analysis, and calculates the GPU runtime.
    $ ./range_gappa.sh  % this script invokes Gappa range analysis running on the CPU, and calculates the CPU runtime, which is compared to the above GPU runtime.
    ```
5. To perform bitwidth allocation:
  * Using ASA running on the CPU:

    ```sh
    $ ./quality_time_asa.sh
    ```
  * Using GPU: 
    - For small benchmarks:

    ```sh
    $ ./quality_time_bitslice.sh
    ```

    - For medium-sized and large benchmarks:

    ```sh
    $ ./quality_time_hybrid.sh
    ```

#### Developer Guide

In this developer guide, we show how to write your own benchmarks and how to hack the code. 

We exemplify the concrete steps using a simple example. 
Consider a 3-order polynomial benchmark named `poly3` which consists of the operation: 
    
``` c++
y = (a-(b-c*x)*x)*x
```

where `a`, `b` and `c` are input constants, and x is the input variable, and `y` is the output. 

One has to hand code two files `poly3.c` and `poly3.range`, which specify the `poly3` function and the range information of all inputs, respectively, and put them under `./bench` folder.
Let's assume that `a` and `b` are input integers with values `1` and `0.5`, respectively, i.e.,
    
``` c++
a = 1 
b = 0.5
```

while `c` is an input decimal with value `0.3`, i.e.,
    
``` c++
c = 0.3
```

In this case, `a` and `b` can be represented with no truncation or rounding errors using certain number of bits, as mentioned in our paper, while there will always be a rounding or truncation error when representing `c`. 

We have prepared these two files. One can refer to them for the coding syntax. `poly3.c` is a piece of C-style code, containing a function that returns the result of the polynomial operation. Since `a` and `b` are input integers, no bitwidth optimization process is required for these two inputs, so we directly write their numerical values in `poly3.c` as follows: 
    
``` c++
double poly3(double x, double c) {
    return (1-(0.5-c*x)*x)*x;
}
```

In `poly3.range`, we specify the ranges of input variables, with one variable one line: 
    
``` c++
x,-1,1
c,0.3,0.3
```

After the benchmark `poly3` is properly written, one can supply the benchmark name into those `Shell scripts` mentioned above, and execute these scripts as mentioned. We show how to do these in detail as follows. 

1. edit `prep_bench.sh` to make it look like: 

  ``` sh
  for DESIGN in poly3
  do
      pushd ./scripts
      ./gimple_to_asm.sh ../bench/$DESIGN.c
      popd
  done
  ```

  The script `gimple_to_asm.sh` generates GIMPLE and Assembly code for the poly3 function and puts them under `./data/poly3/`. 

2. Add the benchmark name `poly3` into `prune.sh`. Run `./prune.sh` to prune the search space of `poly3`, you are expected to see something similar to the following: 

  ```
  benchmark:  poly3
  search space after pruning:  poly3  729
  ```

3. Add the benchmark name `poly3` into `range.sh` and `range_gappa.sh`, respectively. 
  Run `./range.sh`, which runs range analysis on the GPU, you are expected to see: 

```
design,runtime(ms),block_size,intervals
poly3,0.037056,32,1
poly3,0.037344,64,1
poly3,0.036384,128,1
poly3,0.036992,256,1
poly3,0.036416,512,1
poly3,0.037664,32,2
poly3,0.037664,64,2
poly3,0.036480,128,2
poly3,0.036832,256,2
poly3,0.037248,512,2
poly3,0.036864,32,3
poly3,0.036768,64,3
poly3,0.037536,128,3
poly3,0.037440,256,3
poly3,0.036928,512,3
poly3,0.037952,32,4
poly3,0.038368,64,4
poly3,0.038080,128,4
poly3,0.037056,256,4
poly3,0.037920,512,4
poly3,0.039488,32,5
poly3,0.038592,64,5
poly3,0.038336,128,5
poly3,0.038816,256,5
poly3,0.038016,512,5
poly3,0.038528,32,6
poly3,0.039328,64,6
poly3,0.038400,128,6
poly3,0.038368,256,6
poly3,0.038016,512,6
poly3,0.037536,32,7
poly3,0.039264,64,7
poly3,0.038048,128,7
poly3,0.038784,256,7
poly3,0.037920,512,7
poly3,0.039072,32,8
poly3,0.039104,64,8
poly3,0.039424,128,8
poly3,0.039264,256,8
poly3,0.037824,512,8
poly3,0.039872,32,16
poly3,0.039680,64,16
poly3,0.040224,128,16
poly3,0.039136,256,16
poly3,0.039488,512,16
poly3,0.042112,32,32
poly3,0.096160,64,32
poly3,0.042496,128,32
poly3,0.041760,256,32
poly3,0.041440,512,32
poly3,0.041536,32,64
poly3,0.040352,64,64
poly3,0.040128,128,64
poly3,0.040640,256,64
poly3,0.040704,512,64
poly3,0.041440,32,128
poly3,0.041344,64,128
poly3,0.041888,128,128
poly3,0.041376,256,128
poly3,0.041024,512,128
poly3,0.042016,32,256
poly3,0.040416,64,256
poly3,0.040960,128,256
poly3,0.041984,256,256
poly3,0.040352,512,256
poly3,0.041248,32,512
poly3,0.040512,64,512
poly3,0.071328,128,512
poly3,0.042048,256,512
poly3,0.048672,512,512
poly3,0.040960,32,1024
poly3,0.041312,64,1024
poly3,0.040672,128,1024
poly3,0.041120,256,1024
poly3,0.048736,512,1024
poly3,0.039200,32,2048
poly3,0.040896,64,2048
poly3,0.042368,128,2048
poly3,0.042144,256,2048
poly3,0.048512,512,2048
poly3,0.044288,32,4096
poly3,0.044128,64,4096
poly3,0.045184,128,4096
poly3,0.049216,256,4096
poly3,0.049504,512,4096
poly3,0.076896,32,8192
poly3,0.053696,64,8192
poly3,0.054048,128,8192
poly3,0.057408,256,8192
poly3,0.064352,512,8192

```

  The `range.sh` script tries different GPU block_size, as we can see from the third column of the printings. The second column of the above printings refers to the GPU runtime in milliseconds. 

  Run `./range_gappa.sh`, which runs range analysis on the CPU, you are expected to see: 

```
design,splits,runtime(us)
poly3,1, 1316 
poly3,8, 4228 
poly3,16, 7253 
poly3,32, 17120 
poly3,64, 25446 
poly3,128, 52716 
poly3,256, 108975 
poly3,512, 218961 
poly3,1024, 465178 
poly3,2048, 988430 
poly3,4096, 2000420 
poly3,8192, 4196468 
```

  As we can see, we are able to speed up range analysis significantly using GPUs.  

4. Add the benchmark name `poly3` into `quality_time_asa.sh` and `quality_time_bitslice.sh`, respectively. 

  Run `./quality_time_bitslice.sh`, which runs bitwidth allocation on the GPU. The result will be stored in file `./data/quality_time_bitslice_poly3.dat`, including both the runtime(ms) and the optimized bitwidth combination. For this `poly3` example, the runtime using GPU is `0.000126656 ms`. 

  Run `./quality_time_asa.sh`, which runs bitwidth allocation on the CPU. The result will be stored in file `./data/quality_time_asa_poly3.dat`, including both the runtime(us) and the optimized bitwidth combination. For this `poly3` example, the runtime using CPU is `0.0141652 s`

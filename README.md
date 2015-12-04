BitGPU is implemented in Ubuntu 14.04. 
BitGPU is released under the GNU General Public License (GPL). 

#### Prerequisites
To use the code, please install and configure:
- [Nvidia CUDA-7.5](https://developer.nvidia.com/cuda-downloads). Our code works with CUDA version 7.5. We have not tested other versions.
- A modified version of [Gappa++](https://github.com/YeDeheng/gappa), a tool for verifying numerical properties.
- [Adaptive simulated annealing (ASA)](https://www.ingber.com/#ASA). We have packaged ASA in our tool release.  

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
  ...
  ...
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

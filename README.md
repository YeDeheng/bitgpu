BitGPU is released under the GNU General Public License (GPL). 

#### Prerequisites
To use the code, please install and configure:
- [Nvidia CUDA-7.5](https://developer.nvidia.com/cuda-downloads). Our code works has been confirmed to work with CUDA version 7.5 and tested on NVIDIA K20 GPU.
- A modified version of [Gappa++](https://github.com/YeDeheng/gappa), a tool for verifying numerical properties. Strictly speaking, you do not need this, but if you want to confirm correctness of our bounds, you can try the tool. We have modified Gappa to support exp and log operations that should work for most cases but do not provide certificates of correctness for those operators.
- [Adaptive simulated annealing (ASA)](https://www.ingber.com/#ASA). We have already packaged ASA in our tool release, so you do not need to download/configure it. In any case, we have modified ASA to suit our needs.  

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
3. Prune the search space for range analysis and bitwidth allocation. This corresponds to Algorithm 1 from FPGA 2016 paper "GPU-Accelerated High-Level Synthesis for Bitwidth Optimization of FPGA Datapaths":
    ```sh
    $ ./prune.sh
    ```

4. To perform range analysis:
  * Using Gappa running on CPU:
    ```sh
    $ ./range_gappa.sh  
    ```
  * Using GPU-acceleration:
    ```sh
    $ ./range.sh    
    ```

5. To perform bitwidth allocation:
  * Using ASA running on the CPU only (no GPU acceleration):

    ```sh
    $ ./quality_time_asa.sh
    ```
  * Using GPU: 
    - For small benchmarks, you can simply brute-force all possible bitwidth combinations:

    ```sh
    $ ./quality_time_bitslice.sh
    ```

    - For medium-sized and large benchmarks, you need to run ASA-assisted pruning first:

    ```sh
    $ ./quality_time_hybrid.sh
    ```

#### Developer Guide

In this brief developer guide, we show how to write your own benchmarks and how to hack the code. We show the concrete steps using a simple example. 

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

After the benchmark `poly3` is setup correctly, one can supply the benchmark name into those `Shell scripts` mentioned above, and execute these scripts as mentioned. We show how to do these in detail as follows. 

1. You can compile the benchmark into intermediate ASM form 

  ```sh
  pushd ./scripts
  ./gimple_to_asm.sh ../bench/poly3.c
  popd
  ```

  This generates GIMPLE and associate IR code for the poly3 function and puts them under `./data/poly3/`. 

2. Run the pruning step to prune the search space of `poly3` using the prune driver. You must supply your own error threshold as desired by your application and an upper-limit for bitwidth search to help guides the search process.
  ```sh
  /bin/prune_driver ./data/poly3/poly3.asm $ERROR_THRESH $UPPER_BOUND
  ```

Once executed, you should see something similar to the following: 

  ```
  benchmark:  poly3
  search space after pruning:  poly3  729
  ```

3. You can run range analysis to generate right interval bounds for your application. The GPU implementation runtime depends on the number of desired sub-intervals to be evaluated and the precise block configuration for a GPU mapping.
  ```sh
  ./bin/range_driver ./data/poly3/poly3.asm $SPLITS $BLOCKS
  ```

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

  The GPU block_size, shown in the third column of the stdout log is purely a runtime optimization and does not affect correctness. 
  
  For this `poly3` example, the fastest range analysis runtime using GPU is `?? ms`, while the runtime using CPU is `?? s`

4. Finally run the bitwidth allocation algorithm. Again, you must provide your desired application-specific error threshold *$ERROR_THRESH* and the GPU block configuration *$BLOCKS*.
  ```sh
  ./bin/bitslice_driver ./data/poly3/poly3.asm $ERROR_THRESH 1 1 $BLOCKS
  ```
 
  You may also run this purely on the CPU using ASA alone. Here, we first generate the C file that calculates error and cost for ASA and then compiles ASA customized for this problem instance. Then we run the ASA algorithm to generate resulting bitwidth.
  ```sh
  ./create_asa_opt.sh ../data/poly3/poly3.asm $ERROR_THRESH $UPPER_BOUND
  ./asa_run ../data/poly3/poly3.asm $ERROR_THRESH 100000000
  ```

  For this `poly3` example, the runtime using GPU is `0.000126656 ms`, while the runtime using CPU is `0.0141652 s`

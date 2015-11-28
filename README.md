BitGPU is implemented in Ubuntu 14.04.

#### Prerequisites
To use the code, please install and configure:
- [Nvidia CUDA-7.5](https://developer.nvidia.com/cuda-downloads). Our code works with CUDA version 7.5. We have not tested other versions.
- A modified version of [Gappa++](https://github.com/YeDeheng/gappa), a tool for verifying numerical properties.

#### User guide
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

#### Adding new benchmarks
We exemplify the concrete steps using a simple example. 
Consider a 3-order polynomial benchmark named `poly3` which consists of the operation: 
    
    ```cpp
    y = (a-(b-c*x)*x)*x
    ```

where `a`, `b` and `c` are input constants, and x is the input variable, and `y` is the output. 

One has to hand code two files `poly3.c` and `poly3.range`, which specify the `poly3` function and the range information of all inputs, respectively, and put them under `./bench` folder.
Let's assume that `a` and `b` are input integers with values `1` and `0.5`, respectively, i.e.,
    
    ```cpp
    a = 1 
    b = 0.5
    ```

while `c` is an input decimal with value `0.3`, i.e.,
    
```cpp
c = 0.3
```

In this case, `a` and `b` can be represented with no truncation or rounding errors using certain number of bits, as mentioned in our paper, while there will always be a rounding or truncation error when representing `c`. 

We have prepared these two files. One can refer to them for the coding syntax. `poly3.c` is a piece of C-style code, containing a function that returns the result of the polynomial operation. Since `a` and `b` are input integers, no bitwidth optimization process is required for these two inputs, so we directly write their numerical values in `poly3.c` as follows: 
    
    ```cpp
    double poly3(double x, double c) {
            return (1-(0.5-c*x)*x)*x;
    }
    ```

In `poly3.range`, we specify the ranges of input variables, with one variable one line: 
    
    ``` c++
    x,-1,1
    c,0.3,0.3
    ```

After the benchmark `poly3` is properly written, one can supply the benchmark name into those `Shell scripts` mentioned above, and execute these scripts as mentioned. 

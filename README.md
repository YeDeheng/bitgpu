BitGPU is implemented in Ubuntu 14.04. 
BitGPU is released under the GNU General Public License (GPL). 

#### Prerequisites
To use the code, please install and configure:
- [Nvidia CUDA-7.5](https://developer.nvidia.com/cuda-downloads). Our code works with CUDA version 7.5. We have not tested other versions.
- A modified version of [Gappa++](https://github.com/YeDeheng/gappa), a tool for verifying numerical properties.

#### User guide
1. Compile the GPU code and ASA (adaptive simulated annealing) code:

    ```sh
    $ ./make
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
Consider an adder benchmark named `add` which only consists of one operation `c=a+b`, where `a` and `b` are the input variables, and `c` is the output. 
One has to hand code two files `add.c` and `add.range`, and put them under `./bench` folder.
We have prepared these two files. One can refer to them for the coding syntax. `add.c` is a piece of C-style code, containing a function that returns the result of `a+b`. In `add.range`, we specify the ranges of input variables, with one variable one line. 
After the benchmark `add` is written, one can add the benchmark name into those Shell scripts mentioned above, and execute these scripts as mentioned. 
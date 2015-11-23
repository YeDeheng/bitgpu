BitGPU is implemented in Ubuntu 14.04.  

##### Prerequisites 
To use the code, please install and configure: 
- [Nvidia CUDA-7.5](https://developer.nvidia.com/cuda-downloads). Our code works with CUDA version 7.5. We have not tested other versions. 
- A modified version of [Gappa++](https://github.com/YeDeheng/gappa), a tool for verifying numerical properties. 

##### User guide
1. Compile the GPU code and ASA code: 
   ./make
   cd ./asa
   ./make
2. Compile the C-style benchmarks into DFG-style assemby code using gcc's GIMPLE backend: 
   ./prep_bench.sh
3. Prune the search space for range analysis and bitwidth allocation: 
   ./prune.sh
4. To perform range analysis: 
   ./range.sh  --  this script invokes GPU range analysis, and calculates the GPU runtime.  
   ./range_gappa.sh  -- this script invokes Gappa range analysis running on the CPU, and calculates the CPU runtime, which is compared to the above GPU runtime. 
5. To perform bitwidth allocation: 
  * For small benchmarks: 
  * For medium-sized benchmarks: 
    ./
  * For large benchmarks: 
   


##### Developer guide 
###### Adding your own benchmarks



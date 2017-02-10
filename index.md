### BitGPU
BitGPU is a GPU approach to solve the bitwidth optimization problem in FPGA datapaths. 

Bitwidth optimization of FPGA datapaths can save hardware resources by choosing the fewest number of bits required for each datapath variable to achieve a desired quality of result. However, it is an NP-hard problem that requires unacceptably long runtimes when using sequential CPU-based heuristics. We show how to parallelize the key steps of bitwidth optimization on the GPU by performing a fast brute-force search over a carefully constrained search space. We develop a high-level synthesis methodology suitable for rapid prototyping of bitwidth-annotated RTL code generation.

### Authors
This project is contributed by [Dr. Deheng Ye](http://yedeheng.weebly.com/) ([叶德珩](http://yedeheng.weebly.com/) in Chinese) and his PhD supervisor [Dr. Nachiket Kapre](http://nachiket.github.io/). 

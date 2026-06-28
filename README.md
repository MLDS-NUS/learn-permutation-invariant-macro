# learn-permutation-invariant-macro
The official implementation of paper `Learning Permutation-invariant Macroscopic Dynamics`.

## Getting started

### Dependencies
This repository depends on `PyTorch` and `LAMMPS`. We recommand using conda to manage the environment. We run the code with:  
--python: 3.10.19  
--lammps: 29 Aug 2024  
--PyTorch: 2.5.1 with CUDA: 12.4  

Besides, the normalizing flow implementation depends on the [nflow](https://github.com/bayesiains/nflows) library.

### Hardware of our experiments
The simulations are conducted on a server with multiple-core CPU.   
For the learning part, we assume GPU is available. We run most experiments on the A100 gpu card.

## Reproduce experiments in the paper
Detailed step-by-step instructions are provided within each folder.


## License

This project is licensed under the GNU Lesser General Public License v3.0 or later. See `LICENSE`for the complete terms.


## References

Han, Zhichao and Chen, Mengyi and Li, Qianxiao. "Learning Permutation-invariant Macroscopic Dynamics." ICML 2026. [arXiv:2605.30812](https://arxiv.org/abs/2605.30812).
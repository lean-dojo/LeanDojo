LeanDojo: Machine Learning for Theorem Proving in Lean
======================================================

![LeanDojo](https://github.com/lean-dojo/LeanDojo/blob/main/images/LeanDojo.jpg)

[LeanDojo](https://leandojo.org/) is a Python library for learning–based theorem provers in Lean, supporting both [Lean 3](https://github.com/leanprover-community/lean) and [Lean 4](https://leanprover.github.io/). It provides two main features:

* Extracting data (proof states, tactics, premises, etc.) from Lean repos.
* Interacting with Lean programmatically.


[![Documentation Status](https://readthedocs.org/projects/leandojo/badge/?version=latest)](https://leandojo.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/lean-dojo)](https://pypi.org/project/lean-dojo/) [![GitHub license](https://img.shields.io/github/license/lean-dojo/LeanDojo)](https://github.com/lean-dojo/LeanDojo/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

______________________________________________________________________

## Requirements

* Supported platforms: Linux, Windows (WSL), and macOS (:warning: experimental for Apple silicon)
* Git >= 2.25
* 3.9 <= Python <= 3.10
* wget
* Docker strongly recommended


## Installation

LeanDojo is available on [PyPI](https://pypi.org/project/lean-dojo/) and can be installed via pip:
```bash
pip install lean-dojo
```

It can also be installed locally from the Git repo:
```bash
pip install .
```


## Documentation

* [Getting Started](https://leandojo.readthedocs.io/en/latest/getting-started.html)
* Demos: [Lean 3](https://github.com/lean-dojo/LeanDojo/blob/main/scripts/demo-lean3.ipynb), [Lean 4](https://github.com/lean-dojo/LeanDojo/blob/main/scripts/demo-lean4.ipynb)
* [Full documentation](https://leandojo.readthedocs.io/en/latest/index.html)


## Questions and Bugs

* For general questions and discussions, please use [GitHub Discussions](https://github.com/lean-dojo/LeanDojo/discussions).  
* To report a potential bug, please open an issue. In the issue, please include your OS information, the version of LeanDojo, the exact steps to reproduce the error, and complete logs in debug mode (setting the environment variable `VERBOSE` to 1). The more details you provide, the better we will be able to help you. 


## Related Links

* [LeanDojo Website](https://leandojo.org/): The official website of LeanDojo.
* [LeanDojo Benchmark](https://zenodo.org/record/8242196) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8242196.svg)](https://doi.org/10.5281/zenodo.8242196): The dataset used in our paper, consisting of 98,641 theorems and proofs extracted from [mathlib](https://github.com/leanprover-community/mathlib/commits/32a7e535287f9c73f2e4d2aef306a39190f0b504) by [generate-benchmark-lean3.ipynb](./scripts/generate-benchmark-lean3.ipynb). 
* [LeanDojo Benchmark 4](https://zenodo.org/record/8242200) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8242200.svg)](https://doi.org/10.5281/zenodo.8242200): The Lean 4 version of LeanDojo Benchmark, consisting of 100,780 theorems and proofs extracted from [mathlib4](https://github.com/leanprover-community/mathlib4/commit/5a919533f110b7d76410134a237ee374f24eaaad) by [generate-benchmark-lean4.ipynb](./scripts/generate-benchmark-lean4.ipynb).
* [ReProver](https://github.com/lean-dojo/ReProver): The ReProver (Retrieval-Augmented Prover) model in our paper.
* [LeanDojo ChatGPT Plugin](https://github.com/lean-dojo/LeanDojoChatGPT)


## Citation

[LeanDojo: Theorem Proving with Retrieval-Augmented Language Models](https://leandojo.org/)      
Under review, NeurIPS (Datasets and Benchmarks Track), 2023  
[Kaiyu Yang](https://yangky11.github.io/), [Aidan Swope](https://aidanswope.com/about), [Alex Gu](https://minimario.github.io/), [Rahul Chalamala](https://rchalamala.github.io/),  
[Peiyang Song](https://peiyang-song.github.io/), [Shixing Yu](https://billysx.github.io/), [Saad Godil](https://www.linkedin.com/in/saad-godil-9728353/), [Ryan Prenger](https://www.linkedin.com/in/ryan-prenger-18797ba1/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)

```bibtex
@article{yang2023leandojo,
  title={{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
  author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2306.15626},
  year={2023}
}
```

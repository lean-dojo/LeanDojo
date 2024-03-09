LeanDojo: Machine Learning for Theorem Proving in Lean
======================================================

![LeanDojo](https://github.com/lean-dojo/LeanDojo/blob/main/images/LeanDojo.jpg)

[LeanDojo](https://leandojo.org/) is a Python library for learning–based theorem provers in Lean, providing two main features:

* Extracting data (proof states, tactics, premises, etc.) from Lean repos.
* Interacting with Lean programmatically.

LeanDojo's current version is compatible with Lean 4 `v4.3.0-rc2` or later. We strongly suggest using the current version. However, you may use the [`legacy`](https://github.com/lean-dojo/LeanDojo/tree/legacy) branch if you want to work with earlier versions (including Lean 3).

[![Documentation Status](https://readthedocs.org/projects/leandojo/badge/?version=latest)](https://leandojo.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/lean-dojo)](https://pypi.org/project/lean-dojo/) [![GitHub license](https://img.shields.io/github/license/lean-dojo/LeanDojo)](https://github.com/lean-dojo/LeanDojo/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

______________________________________________________________________

## Requirements

* Supported platforms: Linux, Windows WSL, and macOS
* Git >= 2.25
* 3.9 <= Python < 3.11
* wget
* [elan](https://github.com/leanprover/elan)
* Recommended: Generate a [GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#personal-access-tokens-classic) and set the environment variable `GITHUB_ACCESS_TOKEN` to it


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
* [LeanDojo Benchmark](https://doi.org/10.5281/zenodo.8016385) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8016385.svg)](https://doi.org/10.5281/zenodo.8016385): The dataset used in our paper, consisting of 98,734 theorems and proofs extracted from [mathlib](https://github.com/leanprover-community/mathlib/commits/19c869efa56bbb8b500f2724c0b77261edbfa28c) by [generate-benchmark-lean3.ipynb](./scripts/generate-benchmark-lean3.ipynb). 
* [LeanDojo Benchmark 4](https://doi.org/10.5281/zenodo.8040109) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8040109.svg)](https://doi.org/10.5281/zenodo.8040109): The Lean 4 version of LeanDojo Benchmark, consisting of 106,446 theorems and proofs extracted from [mathlib4](https://github.com/leanprover-community/mathlib4/commit/3c307701fa7e9acbdc0680d7f3b9c9fed9081740) by [generate-benchmark-lean4.ipynb](./scripts/generate-benchmark-lean4.ipynb).
* [ReProver](https://github.com/lean-dojo/ReProver): The ReProver (Retrieval-Augmented Prover) model in our paper.
* [LeanDojo ChatGPT Plugin](https://github.com/lean-dojo/LeanDojoChatGPT)
* [Lean Copilot: Running language models as copilots for theorem proving in Lean](https://github.com/lean-dojo/LeanCopilot)

## Citation

[LeanDojo: Theorem Proving with Retrieval-Augmented Language Models](https://leandojo.org/)      
Neural Information Processing Systems (NeurIPS), 2023  
[Kaiyu Yang](https://yangky11.github.io/), [Aidan Swope](https://aidanswope.com/about), [Alex Gu](https://minimario.github.io/), [Rahul Chalamala](https://rchalamala.github.io/),  
[Peiyang Song](https://peiyang-song.github.io/), [Shixing Yu](https://billysx.github.io/), [Saad Godil](https://www.linkedin.com/in/saad-godil-9728353/), [Ryan Prenger](https://www.linkedin.com/in/ryan-prenger-18797ba1/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)

```bibtex
@inproceedings{yang2023leandojo,
  title={{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
  author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

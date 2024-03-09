LeanDojo: Machine Learning for Theorem Proving in Lean
======================================================

.. image:: _static/images/LeanDojo.jpg
  :width: 800
  :alt: LeanDojo overview

`LeanDojo <https://leandojo.org/>`_ is a Python library for learning–based theorem provers in Lean, 
supporting both `Lean 3 <https://github.com/leanprover-community/lean>`_ and `Lean 4 <https://leanprover.github.io/>`_. 
It provides two main features:

* Extracting data (proof states, tactics, premises, etc.) from Lean repos.
* Interacting with Lean programmatically.


Related Links
*************

* `LeanDojo Website <https://leandojo.org/>`_: The official website of LeanDojo.
* `LeanDojo Benchmark <https://zenodo.org/doi/10.5281/zenodo.8016385>`_: The Lean 3 dataset used in `our paper <https://arxiv.org/abs/2306.15626>`_, consisting of 98,734 theorems and proofs extracted from `mathlib <https://github.com/leanprover-community/mathlib/commits/19c869efa56bbb8b500f2724c0b77261edbfa28c>`_ by `generate-benchmark-lean3.ipynb <https://github.com/lean-dojo/LeanDojo/blob/main/scripts/generate-benchmark-lean3.ipynb>`_.
* `LeanDojo Benchmark 4 <https://zenodo.org/doi/10.5281/zenodo.8040109>`_: The Lean 4 version of LeanDojo Benchmark, consisting of 106,446 theorems and proofs extracted from `mathlib4 <https://github.com/leanprover-community/mathlib4/commit/3c307701fa7e9acbdc0680d7f3b9c9fed9081740>`_ by `generate-benchmark-lean4.ipynb <https://github.com/lean-dojo/LeanDojo/blob/main/scripts/generate-benchmark-lean4.ipynb>`_.
* `ReProver <https://github.com/lean-dojo/ReProver>`_: The ReProver (Retrieval-Augmented Prover) model in our paper.
* `LeanInfer <https://github.com/lean-dojo/LeanInfer>`_: Native neural network inference for running ReProver directly in Lean 4.

Citation
********

.. code-block:: bibtex

   @inproceedings{yang2023leandojo,
     title={{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
     author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
     booktitle={Neural Information Processing Systems (NeurIPS)},
     year={2023}
   }
   

Contents
********

.. toctree::
   :maxdepth: 2

   getting-started
   user-guide
   troubleshooting
   developer-guide
   limitations
   credits
   api-reference



Indices and Tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

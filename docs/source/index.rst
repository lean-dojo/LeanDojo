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
* `LeanDojo Benchmark <https://zenodo.org/record/8016386>`_: The dataset used in our paper, consisting of 96,962 theorems and proofs extracted from `mathlib <https://github.com/leanprover-community/mathlib/commits/8c1b484d6a214e059531e22f1be9898ed6c1fd47>`_ by `generate-benchmark-lean3.ipynb <./scripts/generate-benchmark-lean3.ipynb>`_.
* `LeanDojo Benchmark 4 <https://zenodo.org/record/8040110>`_: The Lean 4 version of LeanDojo Benchmark, consisting of 91,766 theorems and proofs extracted from `mathlib4 <https://github.com/leanprover-community/mathlib4/commit/5a919533f110b7d76410134a237ee374f24eaaad>`_ by `generate-benchmark-lean4.ipynb <./scripts/generate-benchmark-lean4.ipynb>`_.
* `ReProver <https://github.com/lean-dojo/ReProver>`_: The ReProver (Retrieval-Augmented Prover) model in our paper.


Citation
********

.. code-block:: lean

   @article{yang2023leandojo,
      title={{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
      author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
      journal={arXiv preprint arXiv:2306.15626},
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

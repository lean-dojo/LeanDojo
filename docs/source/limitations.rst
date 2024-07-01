.. _limitations:

Limitations
===========

LeanDojo has the following limitations. Addressing them won't be our priority in the near future, but we welcome contributions:

* LeanDojo cannot extract data from the `lean4 <https://github.com/leanprover/lean4>`_ repo itself nor interact with theorems in it.
* Currently, LeanDojo cannot process Lean repos that use FFI, e.g., [LeanCopilot](https://github.com/lean-dojo/LeanCopilot).
* LeanDojo does not support term-based proofs or proofs that mixes tactics and terms.
* Theorems extracted by LeanDojo are "syntactic theorems", i.e., they are Lean constants defined using keywords :code:`theorem` or :code:`lemma`. First, they are not guaranteed to be real theorems (Lean constants of type :code:`Prop`). Second, theorems defined in other ways (e.g., using :code:`def`) are not extracted.

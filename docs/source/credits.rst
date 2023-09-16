Credits and Acknowledgements
============================

Contributors
************

If your code has been merged into `the LeanDojo repo <https://github.com/lean-dojo/LeanDojo>`_ and you would 
like your name to appear on this page, please feel free to edit this page and open a PR.


Developers
----------

* `Kaiyu Yang <https://yangky11.github.io/>`_: Postdoctoral Scholar at Caltech
* `Peiyang Song <https://peiyang-song.github.io/>`_: Undergrad at UC Santa Barbara
* `Rahul Chalamala <https://rchalamala.github.io/>`_: Undergrad at Caltech


Advisors
--------

* `Anima Anandkumar <http://tensorlab.cms.caltech.edu/users/anima/>`_: Bren Professor at Caltech, Senior Director of AI Research at NVIDIA


Related Tools
*************

LeanDojo draws on numerous existing tools for interacting with and extracting data from proof assistants. 
The author benefited a lot from the lessons he learned when designing and implementing 
`CoqGym <https://github.com/princeton-vl/CoqGym>`_. 
LeanDojo's interaction implementation partially incorporates `lean-gym <https://github.com/openai/lean-gym>`_'s code but 
fixes some critical issues and expanded its functionality. When prototyping the data extraction part of LeanDojo, the author 
found `Jason Rute <https://jasonrute.github.io/>`_'s `lean_proof_recording <https://github.com/jasonrute/lean_proof_recording>`_ helpful.
Its design was a hacky but extremely clever piece of art. Fortunately, we didn't have to follow that design thanks to Lean's AST exporting mechanism 
(:code:`lean --ast --tsast --tspp`) implemented by `Mario Carneiro <https://www.cmu.edu/hoskinson/people/mario-carneiro.html>`_, 
`Gabriel Ebner <https://gebner.org/>`_, and `Daniel Selsam <https://dselsam.github.io/>`_. 
LeanDojo's mechanism for exporting Lean 4 ASTs is inspired by the conversations with Mario Carneiro during Kaiyu Yang's visit to `Hoskinson Center for Formal Mathematics <https://www.cmu.edu/hoskinson/>`_ (directed by `Jeremy Avigad <https://www.andrew.cmu.edu/user/avigad/>`_) at CMU.

Below is a list of related tools we're aware of. Please reach out if we missed your work!

Lean 3
------

* `lean_proof_recording <https://github.com/jasonrute/lean_proof_recording>`_
* `lean-gym <https://github.com/openai/lean-gym>`_
* `lean-client-python <https://github.com/leanprover-community/lean-client-python>`_

Lean 4
------
* `LeanInk <https://github.com/leanprover/LeanInk>`_
* `lean-gym for Lean 4 <https://github.com/dselsam/lean-gym>`_
* `Daniel Selsam's Lean 4 fork <https://github.com/dselsam/lean4/tree/experiment-trace-tactics>`_
* `repl <https://github.com/leanprover-community/repl>`_


Coq
---

* `CoqGym <https://github.com/princeton-vl/CoqGym>`_
* `GamePad <https://github.com/ml4tp/gamepad>`_

Isabelle
--------

* `PISA <https://github.com/albertqjiang/Portal-to-ISAbelle#pisa-portal-to-isabelle>`_

HOL Light
---------

* `HOList <https://sites.google.com/view/holist/home>`_

Others
------
* `INT <https://github.com/albertqjiang/INT>`_


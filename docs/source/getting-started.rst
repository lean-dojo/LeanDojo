.. _getting-started:

Getting Started
===============

This tutorial walks you through a simple example of using LeanDojo to extract data and interact with Lean. 


Requirements
************

* Supported platforms: Linux, Windows (WSL), and macOS (experimental for Apple silicon)
* Git >= 2.25
* 3.9 <= Python < 3.11
* wget
* `elan <https://github.com/leanprover/elan>`_
* Recommended: Generate a `GitHub personal access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#personal-access-tokens-classic>`_ and set the environment variable :code:`GITHUB_ACCESS_TOKEN` to it

Installation
************

LeanDojo is available on `PyPI <https://pypi.org/project/lean-dojo/>`_ and can be installed via :code:`pip install lean-dojo`.
Alternatively, you can install the most recent version by running :code:`pip install .` locally in the root directory of `LeanDojo's GitHub repo <https://github.com/lean-dojo/LeanDojo>`_.


.. _extracting-data-from-lean4:

Extracting Data from Lean 4
***************************
LeanDojo can also extract data from Lean 4 repos. We use `lean4-example <https://github.com/yangky11/lean4-example>`_ as a simple example,
which has a single Lean file with the theorem:

.. code-block:: lean
   :caption: Lean4Example.lean

    open Nat (add_assoc add_comm)

    theorem hello_world (a b c : Nat) 
      : a + b + c = a + c + b := by 
      rw [add_assoc, add_comm b, ←add_assoc]

    theorem foo (a : Nat) : a + 1 = Nat.succ a := by rfl


We use LeanDojo to trace the repo in Python by specifying its URL and a commit hash:

.. code-block:: python

   from lean_dojo import LeanGitRepo, trace

   repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "04e29174a45eefaccb49b835a372aa762321194e")
   trace(repo, dst_dir="traced_lean4-example")

After a few minutes, it generates a :file:`traced_lean4-example` directory with the subdirectories below.
Please check out :ref:`troubleshooting` if you encounter any issue.

::

   traced_lean4-example
   └─lean4-example
     ├─.lake
     │ ├─packages
     │ │ └─lean4
     │ └─build
     │   ├─ir
     │   │ ├─Lean4Example.dep_paths
     │   │ ├─Lean4Example.ast.json
     │   │ └─Lean4Example.trace.xml
     │   └─lib
     │     └─Lean4Example.olean
     ├─Lean4Example.lean
     └─...

:file:`lean4` is the traced `Lean 4 repo <https://github.com/leanprover/lean4>`_, 
and :file:`lean-example` is the traced example repo. We call them "traced" because each
:file:`*.lean` file is accompanied by the following files:
 
* :file:`*.olean`: Lean's compiled object file. 
* :file:`*.dep_paths`: Paths of dependencies imported by the current file. 
* :file:`*.ast.json`: ASTs exported by `ExtractData.lean <https://github.com/lean-dojo/LeanDojo/blob/main/src/lean_dojo/data_extraction/ExtractData.lean>`_.
* :file:`*.trace.xml`: Syntactic and semantic information extracted from Lean.  

The most important one is :file:`*.trace.xml`.
For example, below is :file:`traced_lean4-example/lean4-example/.lake/build/ir/Lean4Example.trace.xml`:

.. code-block::
   :caption: Lean4Example.trace.xml

   <TracedFile path="Lean4Example.lean" md5="f8eb6563cd78c62389ff6cf40f485a1e">
     <FileNode start="(1, 1)" end="(7, 53)">
       <ModuleHeaderNode>
         <NullNode/>
         <NullNode/>
       </ModuleHeaderNode>
       <CommandOpenNode start="(1, 1)" end="(1, 30)">
         <AtomNode start="(1, 1)" end="(1, 5)" leading="" trailing=" " val="open"/>
         <CommandOpenonlyNode start="(1, 6)" end="(1, 30)">
           <IdentNode start="(1, 6)" end="(1, 9)" leading="" trailing=" " raw_val="Nat" val="Nat"/>
           <AtomNode start="(1, 10)" end="(1, 11)" leading="" trailing="" val="("/>
           <NullNode start="(1, 11)" end="(1, 29)">
             <IdentNode start="(1, 11)" end="(1, 20)" leading="" trailing=" " raw_val="add_assoc" val="add_assoc" full_name="Nat.add_assoc" mod_name="Init.Data.Nat.Basic" def_path=".lake/packages/lean4/src/lean/Init/Data/Nat/Basic.lean" def_start="(138, 19)" def_end="(138, 28)"/>
             <IdentNode start="(1, 21)" end="(1, 29)" leading="" trailing="" raw_val="add_comm" val="add_comm" full_name="Nat.add_comm" mod_name="Init.Data.Nat.Basic" def_path=".lake/packages/lean4/src/lean/Init/Data/Nat/Basic.lean" def_start="(131, 19)" def_end="(131, 27)"/>
           </NullNode>
           <AtomNode start="(1, 29)" end="(1, 30)" leading="" trailing="&#10;&#10;" val=")"/>
         </CommandOpenonlyNode>
       </CommandOpenNode>
       <CommandDeclarationNode start="(3, 1)" end="(5, 41)" name="hello_world" full_name="hello_world">
         <CommandDeclmodifiersNode>
           <NullNode/>
           <NullNode/>
           <NullNode/>
           <NullNode/>
           <NullNode/>
           <NullNode/>
         </CommandDeclmodifiersNode>
         <CommandTheoremNode start="(3, 1)" end="(5, 41)" name="hello_world" full_name="hello_world" _is_private_decl="False">
           <AtomNode start="(3, 1)" end="(3, 8)" leading="" trailing=" " val="theorem"/>
           <CommandDeclidNode start="(3, 9)" end="(3, 20)">
             <IdentNode start="(3, 9)" end="(3, 20)" leading="" trailing=" " raw_val="hello_world" val="hello_world"/>
             <NullNode/>
           </CommandDeclidNode>
           <CommandDeclsigNode start="(3, 21)" end="(4, 26)">
             <NullNode start="(3, 21)" end="(3, 34)">
               <TermExplicitbinderNode start="(3, 21)" end="(3, 34)">
                 <AtomNode start="(3, 21)" end="(3, 22)" leading="" trailing="" val="("/>
                 <NullNode start="(3, 22)" end="(3, 27)">
                   <IdentNode start="(3, 22)" end="(3, 23)" leading="" trailing=" " raw_val="a" val="a"/>
                   <IdentNode start="(3, 24)" end="(3, 25)" leading="" trailing=" " raw_val="b" val="b"/>
                   <IdentNode start="(3, 26)" end="(3, 27)" leading="" trailing=" " raw_val="c" val="c"/>
                 </NullNode>
                 <NullNode start="(3, 28)" end="(3, 33)">
                   <AtomNode start="(3, 28)" end="(3, 29)" leading="" trailing=" " val=":"/>
                   <IdentNode start="(3, 30)" end="(3, 33)" leading="" trailing="" raw_val="Nat" val="Nat" full_name="Nat" mod_name="Init.Prelude" def_path=".lake/packages/lean4/src/lean/Init/Prelude.lean" def_start="(1059, 11)" def_end="(1059, 14)"/>
                 </NullNode>
                 <NullNode/>
                 <AtomNode start="(3, 33)" end="(3, 34)" leading="" trailing="&#10;  " val=")"/>
               </TermExplicitbinderNode>
             </NullNode>
             <TermTypespecNode start="(4, 3)" end="(4, 26)">
               <AtomNode start="(4, 3)" end="(4, 4)" leading="" trailing=" " val=":"/>
               <OtherNode start="(4, 5)" end="(4, 26)" kind="«term_=_»">
                 <OtherNode start="(4, 5)" end="(4, 14)" kind="«term_+_»">
                   <OtherNode start="(4, 5)" end="(4, 10)" kind="«term_+_»">
                     <IdentNode start="(4, 5)" end="(4, 6)" leading="" trailing=" " raw_val="a" val="a"/>
                     <AtomNode start="(4, 7)" end="(4, 8)" leading="" trailing=" " val="+"/>
                     <IdentNode start="(4, 9)" end="(4, 10)" leading="" trailing=" " raw_val="b" val="b"/>
                   </OtherNode>
                   <AtomNode start="(4, 11)" end="(4, 12)" leading="" trailing=" " val="+"/>
                   <IdentNode start="(4, 13)" end="(4, 14)" leading="" trailing=" " raw_val="c" val="c"/>
                 </OtherNode>
                 <AtomNode start="(4, 15)" end="(4, 16)" leading="" trailing=" " val="="/>
                 <OtherNode start="(4, 17)" end="(4, 26)" kind="«term_+_»">
                   <OtherNode start="(4, 17)" end="(4, 22)" kind="«term_+_»">
                     <IdentNode start="(4, 17)" end="(4, 18)" leading="" trailing=" " raw_val="a" val="a"/>
                     <AtomNode start="(4, 19)" end="(4, 20)" leading="" trailing=" " val="+"/>
                     <IdentNode start="(4, 21)" end="(4, 22)" leading="" trailing=" " raw_val="c" val="c"/>
                   </OtherNode>
                   <AtomNode start="(4, 23)" end="(4, 24)" leading="" trailing=" " val="+"/>
                   <IdentNode start="(4, 25)" end="(4, 26)" leading="" trailing=" " raw_val="b" val="b"/>
                 </OtherNode>
               </OtherNode>
             </TermTypespecNode>
           </CommandDeclsigNode>
           <CommandDeclvalsimpleNode start="(4, 27)" end="(5, 41)">
             <AtomNode start="(4, 27)" end="(4, 29)" leading="" trailing=" " val=":="/>
             <TermBytacticNode start="(4, 30)" end="(5, 41)">
               <AtomNode start="(4, 30)" end="(4, 32)" leading="" trailing="&#10;  " val="by"/>
               <TacticTacticseqNode start="(5, 3)" end="(5, 41)">
                 <TacticTacticseq1IndentedNode start="(5, 3)" end="(5, 41)">
                   <NullNode start="(5, 3)" end="(5, 41)">
                     <OtherNode start="(5, 3)" end="(5, 41)" kind="Lean.Parser.Tactic.rwSeq" state_before="a b c : Nat&#10;⊢ a + b + c = a + c + b" state_after="no goals" tactic="rw [add_assoc, add_comm b, ←add_assoc]">
                       <AtomNode start="(5, 3)" end="(5, 5)" leading="" trailing=" " val="rw"/>
                       <NullNode/>
                       <OtherNode start="(5, 6)" end="(5, 41)" kind="Lean.Parser.Tactic.rwRuleSeq">
                         <AtomNode start="(5, 6)" end="(5, 7)" leading="" trailing="" val="["/>
                         <NullNode start="(5, 7)" end="(5, 40)">
                           <OtherNode start="(5, 7)" end="(5, 16)" kind="Lean.Parser.Tactic.rwRule">
                             <NullNode/>
                             <IdentNode start="(5, 7)" end="(5, 16)" leading="" trailing="" raw_val="add_assoc" val="add_assoc" full_name="Nat.add_assoc" mod_name="Init.Data.Nat.Basic" def_path=".lake/packages/lean4/src/lean/Init/Data/Nat/Basic.lean" def_start="(138, 19)" def_end="(138, 28)"/>
                           </OtherNode>
                           <AtomNode start="(5, 16)" end="(5, 17)" leading="" trailing=" " val=","/>
                           <OtherNode start="(5, 18)" end="(5, 28)" kind="Lean.Parser.Tactic.rwRule">
                             <NullNode/>
                             <OtherNode start="(5, 18)" end="(5, 28)" kind="Lean.Parser.Term.app">
                               <IdentNode start="(5, 18)" end="(5, 26)" leading="" trailing=" " raw_val="add_comm" val="add_comm" full_name="Nat.add_comm" mod_name="Init.Data.Nat.Basic" def_path=".lake/packages/lean4/src/lean/Init/Data/Nat/Basic.lean" def_start="(131, 19)" def_end="(131, 27)"/>
                               <NullNode start="(5, 27)" end="(5, 28)">
                                 <IdentNode start="(5, 27)" end="(5, 28)" leading="" trailing="" raw_val="b" val="b"/>
                               </NullNode>
                             </OtherNode>
                           </OtherNode>
                           <AtomNode start="(5, 28)" end="(5, 29)" leading="" trailing=" " val=","/>
                           <OtherNode start="(5, 30)" end="(5, 40)" kind="Lean.Parser.Tactic.rwRule">
                             <NullNode start="(5, 30)" end="(5, 31)">
                               <OtherNode start="(5, 30)" end="(5, 31)" kind="patternIgnore">
                                 <OtherNode start="(5, 30)" end="(5, 31)" kind="token.«← »">
                                   <AtomNode start="(5, 30)" end="(5, 31)" leading="" trailing="" val="←"/>
                                 </OtherNode>
                               </OtherNode>
                             </NullNode>
                             <IdentNode start="(5, 31)" end="(5, 40)" leading="" trailing="" raw_val="add_assoc" val="add_assoc" full_name="Nat.add_assoc" mod_name="Init.Data.Nat.Basic" def_path=".lake/packages/lean4/src/lean/Init/Data/Nat/Basic.lean" def_start="(138, 19)" def_end="(138, 28)"/>
                           </OtherNode>
                         </NullNode>
                         <AtomNode start="(5, 40)" end="(5, 41)" leading="" trailing="&#10;&#10;" val="]"/>
                       </OtherNode>
                       <NullNode/>
                     </OtherNode>
                   </NullNode>
                 </TacticTacticseq1IndentedNode>
               </TacticTacticseqNode>
             </TermBytacticNode>
             <OtherNode kind="Lean.Parser.Termination.suffix">
               <NullNode/>
               <NullNode/>
             </OtherNode>
             <NullNode/>
           </CommandDeclvalsimpleNode>
         </CommandTheoremNode>
       </CommandDeclarationNode>
       <CommandDeclarationNode start="(7, 1)" end="(7, 53)" name="foo" full_name="foo">
         <CommandDeclmodifiersNode>
           <NullNode/>
           <NullNode/>
           <NullNode/>
           <NullNode/>
           <NullNode/>
           <NullNode/>
         </CommandDeclmodifiersNode>
         <CommandTheoremNode start="(7, 1)" end="(7, 53)" name="foo" full_name="foo" _is_private_decl="False">
           <AtomNode start="(7, 1)" end="(7, 8)" leading="" trailing=" " val="theorem"/>
           <CommandDeclidNode start="(7, 9)" end="(7, 12)">
             <IdentNode start="(7, 9)" end="(7, 12)" leading="" trailing=" " raw_val="foo" val="foo"/>
             <NullNode/>
           </CommandDeclidNode>
           <CommandDeclsigNode start="(7, 13)" end="(7, 43)">
             <NullNode start="(7, 13)" end="(7, 22)">
               <TermExplicitbinderNode start="(7, 13)" end="(7, 22)">
                 <AtomNode start="(7, 13)" end="(7, 14)" leading="" trailing="" val="("/>
                 <NullNode start="(7, 14)" end="(7, 15)">
                   <IdentNode start="(7, 14)" end="(7, 15)" leading="" trailing=" " raw_val="a" val="a"/>
                 </NullNode>
                 <NullNode start="(7, 16)" end="(7, 21)">
                   <AtomNode start="(7, 16)" end="(7, 17)" leading="" trailing=" " val=":"/>
                   <IdentNode start="(7, 18)" end="(7, 21)" leading="" trailing="" raw_val="Nat" val="Nat" full_name="Nat" mod_name="Init.Prelude" def_path=".lake/packages/lean4/src/lean/Init/Prelude.lean" def_start="(1059, 11)" def_end="(1059, 14)"/>
                 </NullNode>
                 <NullNode/>
                 <AtomNode start="(7, 21)" end="(7, 22)" leading="" trailing=" " val=")"/>
               </TermExplicitbinderNode>
             </NullNode>
             <TermTypespecNode start="(7, 23)" end="(7, 43)">
               <AtomNode start="(7, 23)" end="(7, 24)" leading="" trailing=" " val=":"/>
               <OtherNode start="(7, 25)" end="(7, 43)" kind="«term_=_»">
                 <OtherNode start="(7, 25)" end="(7, 30)" kind="«term_+_»">
                   <IdentNode start="(7, 25)" end="(7, 26)" leading="" trailing=" " raw_val="a" val="a"/>
                   <AtomNode start="(7, 27)" end="(7, 28)" leading="" trailing=" " val="+"/>
                   <OtherNode start="(7, 29)" end="(7, 30)" kind="num">
                     <AtomNode start="(7, 29)" end="(7, 30)" leading="" trailing=" " val="1"/>
                   </OtherNode>
                 </OtherNode>
                 <AtomNode start="(7, 31)" end="(7, 32)" leading="" trailing=" " val="="/>
                 <OtherNode start="(7, 33)" end="(7, 43)" kind="Lean.Parser.Term.app">
                   <IdentNode start="(7, 33)" end="(7, 41)" leading="" trailing=" " raw_val="Nat.succ" val="Nat.succ" full_name="Nat.succ" mod_name="Init.Prelude" def_path=".lake/packages/lean4/src/lean/Init/Prelude.lean" def_start="(1065, 5)" def_end="(1065, 9)"/>
                   <NullNode start="(7, 42)" end="(7, 43)">
                     <IdentNode start="(7, 42)" end="(7, 43)" leading="" trailing=" " raw_val="a" val="a"/>
                   </NullNode>
                 </OtherNode>
               </OtherNode>
             </TermTypespecNode>
           </CommandDeclsigNode>
           <CommandDeclvalsimpleNode start="(7, 44)" end="(7, 53)">
             <AtomNode start="(7, 44)" end="(7, 46)" leading="" trailing=" " val=":="/>
             <TermBytacticNode start="(7, 47)" end="(7, 53)">
               <AtomNode start="(7, 47)" end="(7, 49)" leading="" trailing=" " val="by"/>
               <TacticTacticseqNode start="(7, 50)" end="(7, 53)">
                 <TacticTacticseq1IndentedNode start="(7, 50)" end="(7, 53)">
                   <NullNode start="(7, 50)" end="(7, 53)">
                     <OtherNode start="(7, 50)" end="(7, 53)" kind="Lean.Parser.Tactic.tacticRfl" state_before="a : Nat&#10;⊢ a + 1 = Nat.succ a" state_after="no goals" tactic="rfl">
                       <AtomNode start="(7, 50)" end="(7, 53)" leading="" trailing="&#10;" val="rfl"/>
                     </OtherNode>
                   </NullNode>
                 </TacticTacticseq1IndentedNode>
               </TacticTacticseqNode>
             </TermBytacticNode>
             <OtherNode kind="Lean.Parser.Termination.suffix">
               <NullNode/>
               <NullNode/>
             </OtherNode>
             <NullNode/>
           </CommandDeclvalsimpleNode>
         </CommandTheoremNode>
       </CommandDeclarationNode>
     </FileNode>
     <Comments/>
   </TracedFile>


Interacting with Lean 4
***********************

LeanDojo can also interact with Lean 4. Below we prove the :code:`hello_world` 
theorem in the previous example. Note that the `lean4-example <https://github.com/yangky11/lean4-example>`_ repo
has to be traced before interacting with any theorem in it. So the code below will first take some time 
to trace the repo if you haven't followed the steps in :ref:`extracting-data-from-lean4`.  The tracing has to be done 
only once, and the traced repo will be cached for future use. Some repos do not need to be traced locally and 
can be downloaded from `our AWS S3 <https://lean-dojo.s3.amazonaws.com>`_ (see :ref:`caching` for details).

.. code-block:: python

   from lean_dojo import *

   repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "fd14c4c8b29cc74a082e5ae6f64c2fb25b28e15e")
   theorem = Theorem(repo, "Lean4Example.lean", "hello_world")

   with Dojo(theorem) as (dojo, init_state):
     print(init_state)
     result = dojo.run_tac(init_state, "rw [add_assoc, add_comm b, ←add_assoc]")
     assert isinstance(result, ProofFinished)
     print(result)

.. code-block::
   :caption: Expected output:

   TacticState(pp='a b c : Nat\n⊢ a + b + c = a + c + b', id=0, message=None)
   ProofFinished(tactic_state_id=1, message='')


Next Steps
**********

This example is just a glimpse of what LeanDojo can do. Please continue to the demos (`Lean 3 <https://github.com/lean-dojo/LeanDojo/blob/main/scripts/demo-lean3.ipynb>`_, `Lean 4 <https://github.com/lean-dojo/LeanDojo/blob/main/scripts/demo-lean4.ipynb>`_) and the :ref:`user-guide`.

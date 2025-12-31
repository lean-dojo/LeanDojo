import Lean
import LeanCopilot.Frontend
import LeanCopilot.Models.External

open Lean Elab Tactic


set_option autoImplicit false

namespace LeanCopilot

/--
Pretty-print a list of goals.
-/
def ppTacticState : List MVarId → MetaM String
  | [] => return "no goals"
  | [g] => return (← Meta.ppGoal g).pretty
  | goals =>
      return (← goals.foldlM (init := "") (fun a b => do return s!"{a}\n\n{(← Meta.ppGoal b).pretty}")).trim


/--
Pretty-print the current tactic state.
-/
def getPpTacticState : TacticM String := do
  let goals ← getUnsolvedGoals
  ppTacticState goals


/--
Generate a list of tactic suggestions.
-/
def suggestTactics (modelName : String) (targetPrefix : String) : TacticM (Array (String × Float)) := do
  let state ← getPpTacticState
  let suggestions ← generateRunpod state modelName targetPrefix
  return suggestions


syntax "suggest_tactics_deepseek" : tactic
syntax "suggest_tactics_deepseek" str : tactic


macro_rules
  | `(tactic | suggest_tactics_deepseek%$tac) => `(tactic | suggest_tactics_deepseek%$tac "")



elab_rules : tactic
  | `(tactic | suggest_tactics_deepseek%$tac $pfx:str) => do
    let tacticsWithScores ← suggestTactics "deepseek" pfx.getString
    let tactics := tacticsWithScores.map (·.1)
    let range : String.Range := { start := tac.getRange?.get!.start, stop := pfx.raw.getRange?.get!.stop }
    let ref := Syntax.ofRange range
    hint ref tactics True


end LeanCopilot

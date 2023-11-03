-- REPL for interacting with Lean 4 via the command line.
import Lean.Elab.Tactic
import Lean.Elab.Frontend

open Lean Lean.Meta Lean.Elab Lean.Elab.Command Lean.Elab.Tactic

namespace LeanDojo


/-- Print the response as JSON. --/
private def printResponse {α : Type _} [ToJson α] (res : α) : IO Unit := do
  let json := (toJson res).pretty 99999999999999999
  println! "REPL> {json}"
  (← IO.getStdout).flush


/-- Join a list of strings using a separator. --/
private def join (l : List String) (sep : String := "\n") : String :=
  match l with
  | [] => ""
  | first :: others => others.foldl (fun r s => r ++ sep ++ s) first


/-- A request to REPL. --/
structure Request where
  /-- Tactic/command state ID on which to execute the request. -/
  sid: Nat
  /-- Tactic/command. --/
  cmd: String
deriving FromJson, ToJson


/-- A response to REPL. --/
structure Response where
  /-- New tactic/command state ID. --/
  sid : Option Nat := none
  /-- Next tactic state. --/
  tacticState : Option String := none
  /-- Error message. --/
  error: Option String := none
deriving ToJson


/-- The state of the REPL. --/
structure ReplState (σ : Type _) where
  /-- Saved tactic/command states. --/
  savedStates : Array σ
  /-- The first solved tactic state. --/
  solvedState : Option σ


/-- Get the saved tactic state with the given ID. --/
private def getSavedState? (m : Type → Type) [Monad m] [MonadState (ReplState σ) m] (sid : Nat) : m (Option σ) := do
  return (← get).savedStates[sid]?


/-- Get the initial tactic state. --/
private def getInitialState! (m : Type → Type) [Monad m] [MonadState (ReplState σ) m] [MonadError m] : m σ := do
  let some ts ← getSavedState? m 0 | throwError "[fatal] no initial state"
  return ts


/-- Get the next state ID. --/
private def getNextSid (m : Type → Type) [Monad m] [MonadState (ReplState σ) m] : m Nat := do
  return (← get).savedStates.size


namespace TacticRepl


/-- The tactic REPL monad. --/
abbrev TacticReplM := StateT (ReplState Tactic.SavedState) TacticM


instance : MonadLift IO TacticReplM where
  monadLift x := liftM x


/-- Insert a tactic state into the REPL state. --/
private def insertTacticState (ts : Tactic.SavedState) : TacticReplM Unit := do
  let succeeded := ts.tactic.goals.isEmpty
  modifyGet fun s => ((), ⟨s.savedStates.push ts,
    match s.solvedState with
    | some _ => s.solvedState
    | none => if succeeded then ts else none
  ⟩)


/-- Pretty print the given tactic state. --/
def ppTacticState (ts : Tactic.SavedState) : TacticM String := do
    match ts.tactic.goals with
    | [] => return "no goals"
    | [g] => return (← Meta.ppGoal g).pretty
    | goals =>
      return (← goals.foldlM (fun a b => do return a ++ "\n\n" ++ (← Meta.ppGoal b).pretty) "").trim


/-- Initialize the REPL. --/
private def initializeTacticRepl : TacticM Tactic.SavedState := do
  if not (← isProp (← getMainTarget)) then
    throwError "[fatal] not_a_theorem"
  pruneSolvedGoals
  let ts ← Tactic.saveState
  let ts_str ← ppTacticState ts
  let res : Response := {sid := some 0, tacticState := ts_str}
  printResponse res
  return ts


private def levels2Names : List Level → NameSet
  | [] => NameSet.empty
  | Level.param n :: us => (levels2Names us).insert n
  | _ :: us => levels2Names us


private def collectFromLevel : Level → NameSet
| Level.zero => NameSet.empty
| Level.succ l => collectFromLevel l
| Level.param n => NameSet.empty.insert n
| Level.max l1 l2 => (collectFromLevel l1).union $ collectFromLevel l2
| Level.imax l1 l2 => (collectFromLevel l1).union $ collectFromLevel l2
| Level.mvar _ => NameSet.empty


private def collectLevelParams : Expr → NameSet
  | .sort u => collectFromLevel u
  | .const _ us => levels2Names us
  | .app fm arg => (collectLevelParams fm).union $ collectLevelParams arg
  | .lam _ binderType body _ => (collectLevelParams binderType).union $ collectLevelParams body
  | .forallE _ binderType body _ => (collectLevelParams binderType).union $ collectLevelParams body
  | .letE _ type value body _ => ((collectLevelParams type).union $ collectLevelParams value).union $ collectLevelParams body
  | .mdata _ expr => collectLevelParams expr
  | .proj _ _ struct => collectLevelParams struct
  | _ => NameSet.empty


private def collectFVarsAux : Expr → NameSet
  | .fvar fvarId => NameSet.empty.insert fvarId.name
  | .app fm arg => (collectFVarsAux fm).union $ collectFVarsAux arg
  | .lam _ binderType body _ => (collectFVarsAux binderType).union $ collectFVarsAux body
  | .forallE _ binderType body _ => (collectFVarsAux binderType).union $ collectFVarsAux body
  | .letE _ type value body _ => ((collectFVarsAux type).union $ collectFVarsAux value).union $ collectFVarsAux body
  | .mdata _ expr => collectFVarsAux expr
  | .proj _ _ struct => collectFVarsAux struct
  | _ => NameSet.empty


private def collectFVars (e : Expr) : MetaM (Array Expr) := do
  let names := collectFVarsAux e
  let mut fvars := #[]
  for ldecl in ← getLCtx do
    if ldecl.isImplementationDetail then
      continue
    if names.contains ldecl.fvarId.name then
      fvars := fvars.push $ .fvar ldecl.fvarId
  return fvars


private def abstractAllLambdaFVars (e : Expr) : MetaM Expr := do
  let mut e' := e
  while e'.hasFVar do
    let fvars ← collectFVars e'
    if fvars.isEmpty then
      break
    e' ← mkLambdaFVars fvars e'
  return e'


private def validateProof : TacticReplM Response := do
  let ts ← Tactic.saveState

  -- Go to the initial state and grab the goal's metavariable ID.
  let ts0 ← getInitialState! TacticReplM
  ts0.restore
  let [goalId] ← getGoals | throwError "[fatal] more than one initial goal"
  let tgt ← getMainTarget >>= instantiateMVars
  let tgt_fmt ← ppExpr tgt

  -- Check its assigned Expr in the current state.
  ts.restore
  let some pf ← getExprMVarAssignment? goalId | throwError "[fatal] goal not assigned"
  let pf ← instantiateMVars pf
  let pft ← inferType pf >>= instantiateMVars
  let pft_fmt ← ppExpr pft

  if ! (← withTransparency .all (isExprDefEq tgt pft)) then
    return {error := s!"proof type mismatch: {tgt_fmt} != {pft_fmt}"}

  ts0.restore
  let pf ← goalId.withContext $ abstractAllLambdaFVars pf
  let pft ← inferType pf >>= instantiateMVars

  ts.restore
  if pf.hasSorry then
    return {error := "proof contains `sorry`"}

  if pf.hasExprMVar then
    return {error := "proof contains metavariables"}

  -- Kernel type check.
  let lvls := (collectLevelParams pf).toList
  let decl := Declaration.thmDecl {
      name := Name.anonymous, type := pft, value := pf
      levelParams := lvls
  }
  try
    let _ ← addDecl decl
  catch ex =>
    return {error := s!"kernel type check failed: {← ex.toMessageData.toString}"}

  let ts_str ← ppTacticState ts
  let next_tsid ← getNextSid TacticReplM
  insertTacticState ts
  return {sid := next_tsid, tacticState := ts_str}


private def handleRunTac (req : Request) : TacticReplM Response := do
  match ← getSavedState? TacticReplM req.sid with
  | none => throwError s!"[fatal] unknown tsid: {req.sid}"
  | some ts =>
    match Parser.runParserCategory (← getEnv) `tactic req.cmd "<stdin>" with
    | .error err => return {error := err}
    | .ok stx =>
      ts.restore

      try
        monadLift $ commitIfNoEx (evalTactic stx)
        let s ← getThe Core.State
        if s.messages.hasErrors then
          let messages := s.messages.toList.filter fun m => m.severity == MessageSeverity.error
          return { error := join $ ← (messages.map Message.data).mapM fun md => md.toString }
      catch ex =>
        return {error := ← ex.toMessageData.toString}

      pruneSolvedGoals
      if (← getGoals).isEmpty then
        validateProof
      else
        let ts' ← Tactic.saveState
        let ts'_str ← ppTacticState ts'
        let next_tsid ← getNextSid TacticReplM
        insertTacticState ts'
        return {sid := next_tsid, tacticState := ts'_str}


end TacticRepl


private def loop (m : Type → Type) [Monad m] [MonadLift IO m] [MonadError m] (handler : Request → m Response) : m Unit := do
  while true do
    let line ← (← IO.getStdin).getLine
    if line.trim == "exit" then
      break
    match (Json.parse line) with
    | .error err => throwError s!"[fatal] failed to parse JSON {err}"
    | .ok cmd =>
      match (fromJson? cmd : Except String Request) with
      | .error err => throwError s!"[fatal] parse_failed: data={err}"
      | .ok req => (← handler req) |> printResponse


namespace TacticRepl

/--
{"sid": 0, "cmd": "skip"}
{"sid": 1, "cmd": "rw [add_assoc, add_comm b, ←add_assoc]"}
exit
--/
def repl : TacticM Unit := do
  withMainContext do
    -- Print the initial goal.
    let ts ← initializeTacticRepl
    -- Interaction through the command line.
    let loop := LeanDojo.loop TacticReplM handleRunTac
    let (_, s) ← loop.run {savedStates := #[ts], solvedState := none}
    -- Close the proof if we have found a solved tactic state.
    match s.solvedState with
    | none => return ()
    | some ts' => ts'.restore
  IO.Process.exit 0


end TacticRepl


namespace CommandRepl


/-- The REPL monad. --/
abbrev CommandReplM := StateT (ReplState Command.State) CommandElabM


instance : MonadLift IO CommandReplM where
  monadLift x := liftM x


/-- Insert a command state into the REPL state. --/
private def insertCommandState (cs : Command.State) : CommandReplM Unit := do
  modifyGet fun s => ((), ⟨s.savedStates.push cs, none⟩)


/-- Initialize the REPL. --/
private def initializeRepl : CommandElabM Command.State := do
  let res : Response := {sid := some 0}
  printResponse res
  return (← get)


private def handleRunCmd (req : Request) : CommandReplM Response := do
  match ← getSavedState? CommandReplM req.sid with
  | none => throwError s!"[fatal] unknown csid: {req.sid}"
  | some cs =>
    let inputCtx := Parser.mkInputContext req.cmd "<stdin>"
    let parserState := { : Parser.ModuleParserState }
    let cs' := (← IO.processCommands inputCtx parserState cs).commandState

    -- Collect error messages and print other messages.
    let messages := cs'.messages.toList
    let mut errors := #[]
    for msg in messages do
      let s ← msg.data.toString
      if msg.severity == MessageSeverity.error then
        errors := errors.push s
      else
        println! s.trim
    let err_msg := if errors.isEmpty then none else some (join errors.toList)

    let next_csid ← getNextSid CommandReplM
    insertCommandState cs'
    return {sid := next_csid, error := err_msg}


/--
{"sid": 0, "cmd": "#eval 1"}
{"sid": 1, "cmd": "#eval x"}
{"sid": 0, "cmd": "def x := 1"}
{"sid": 3, "cmd": "#eval x"}
exit
--/
def repl : CommandElabM Unit := do
  let cs ← initializeRepl
  let loop := LeanDojo.loop CommandReplM handleRunCmd
  let _ ← loop.run {savedStates := #[cs], solvedState := none}
  IO.Process.exit 0

end CommandRepl

end LeanDojo


/-- The `lean_dojo_repl` tactic. --/
elab "lean_dojo_repl" : tactic => LeanDojo.TacticRepl.repl


/-- The `#lean_dojo_repl` command. --/
elab "#lean_dojo_repl" : command => LeanDojo.CommandRepl.repl

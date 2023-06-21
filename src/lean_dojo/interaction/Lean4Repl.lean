import Lean.Elab.Tactic
open Lean Lean.Meta Lean.Elab Lean.Elab.Tactic

namespace LeanDojo

/-- A request to REPL. --/
structure Request where
  /-- Tactic state ID on which to execute the request. -/
  tsid: Nat
  /-- Tactic. --/
  tac: String
deriving FromJson, ToJson


/-- A response to REPL. --/
structure Response where
  /-- New tactic state ID. --/
  tsid : Option Nat := none
  /-- Next tactic state. --/
  tactic_state : Option String := none
  /-- Error message. --/
  error: Option String := none
deriving ToJson


/-- The state of the REPL. --/
structure ReplState where
  /-- Saved tactic states. --/
  saved_states : Array Tactic.SavedState
  /-- The first solved tactic state. --/
  solved_ts : Option Tactic.SavedState


/-- The REPL monad. --/
abbrev ReplM := StateT ReplState TacticM


/-- Get the saved tactic state with the given ID. --/
def getSavedState?(tsid : Nat) : ReplM (Option Tactic.SavedState) := do
  return (← get).saved_states[tsid]?


/-- Get the initial tactic state. --/
def getInitialState : ReplM Tactic.SavedState := do
  let some ts ← getSavedState? 0 | throwError "[fatal] no initial state"
  return ts


/-- Get the next tactic state ID. --/
def getNextTsid : ReplM Nat := do
  return (← get).saved_states.size


/-- Insert a tactic state into the REPL state. --/
def insert (ts : Tactic.SavedState) : ReplM Unit := do
  let succeeded := ts.tactic.goals.isEmpty
  modifyGet fun s => ((), ⟨s.saved_states.push ts,
    match s.solved_ts with
    | some _ => s.solved_ts
    | none => if succeeded then ts else none
  ⟩)


/-- Pretty print the given tactic state. --/
def ppTacticState (ts : Tactic.SavedState) : TacticM String := do
    match ts.tactic.goals with
    | [] => return "no goals"
    | [g] => return (← Meta.ppGoal g).pretty
    | goals =>
      return (← goals.foldlM (fun a b => do return a ++ "\n\n" ++ (← Meta.ppGoal b).pretty) "").trim


/-- Print the response as JSON. --/
private def printResponse (res : Response) : IO Unit := do
  let json := (toJson res).pretty 99999999999999999
  println! "REPL> {json}"
  (← IO.getStdout).flush


/-- Initialize the REPL. --/
private def initializeRepl : TacticM Tactic.SavedState := do
  if not (← isProp (← getMainTarget)) then throwError "[fatal] not_a_theorem"
  pruneSolvedGoals
  let ts ← Tactic.saveState
  let ts_str ← ppTacticState ts
  let res : Response := {tsid := some 0, tactic_state := ts_str}
  printResponse res
  return ts


private def levels2Names : List Level → NameSet
  | [] => NameSet.empty
  | Level.param n :: us => (levels2Names us).insert n
  | _ :: us => levels2Names us


private def collectLevelParams : Expr → NameSet
  | .sort (Level.param n) => NameSet.empty.insert n
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


private def validateProof : ReplM Response := do
  let ts ← Tactic.saveState

  -- Go to the initial state and grab the goal's metavariable ID.
  let ts0 ← getInitialState
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
  let next_tsid ← getNextTsid
  insert ts
  return {tsid := next_tsid, tactic_state := ts_str}


private def join (l : List String) (sep : String := " ") : String :=
  l.foldl (fun r s => r ++ sep ++ s) ""


private def handleRunTac (tsid : Nat) (tac : String) : ReplM Response := do
  match ← getSavedState? tsid with
  | none => throwError s!"[fatal] unknown tsid: {tsid}"
  | some ts =>
    match Parser.runParserCategory (← getEnv) `tactic tac "<stdin>" with
    | .error err => return {error := err}
    | .ok stx =>
      ts.restore

      try
        evalTactic stx
        let s ←  getThe Core.State
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
        let next_tsid ← getNextTsid
        insert ts'
        return {tsid := next_tsid, tactic_state := ts'_str}


private def loop : ReplM Unit := do
  while true do
    let line ← (← IO.getStdin).getLine
    if line.trim == "exit" then break
    match (Json.parse line) with
    | .error err => throwError s!"[fatal] failed to parse JSON {err}"
    | .ok cmd =>
      match (fromJson? cmd : Except String Request) with
      | .error err => throwError s!"[fatal] parse_failed: data={err}"
      | .ok req =>
        let res ← handleRunTac req.tsid req.tac
        printResponse res


/--
{"tsid": 0, "tac": "skip"}
{"tsid": 1, "tac": "rw [add_assoc, add_comm b, ←add_assoc]"}
exit
--/
def repl : TacticM Unit := do
  withMainContext do
    -- Print the initial goal.
    let ts ← initializeRepl
    -- Interaction through the command line.
    let (_, s) ← loop.run {saved_states := #[ts], solved_ts := none}
    -- Close the proof if we have found a solved tactic state.
    match s.solved_ts with
    | none => return ()
    | some ts' => ts'.restore


end LeanDojo


/-- The `lean_dojo_repl` tactic. --/
elab "lean_dojo_repl" : tactic => LeanDojo.repl

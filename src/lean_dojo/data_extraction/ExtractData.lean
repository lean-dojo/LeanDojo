import Lean

open Lean Elab System

instance : ToJson Substring where
  toJson s := toJson s.toString

instance : ToJson String.Pos where
  toJson n := toJson n.1

deriving instance ToJson for SourceInfo
deriving instance ToJson for Syntax.Preresolved
deriving instance ToJson for Syntax


namespace LeanDojo


structure TacticTrace where
  stateBefore: String
  stateAfter: String
  pos: String.Pos
  endPos: String.Pos
deriving ToJson


structure Trace where
  commandASTs : Array Syntax
  tactics: Array TacticTrace
deriving ToJson


abbrev TraceM := StateT Trace IO


namespace Pp


private def addLine (s : String) : String :=
  if s.isEmpty then s else s ++ "\n"


-- Similar to `Meta.ppGoal` but use String instead of Format to make sure local declarations are separted by "\n".
private def ppGoal (mvarId : MVarId) : MetaM String := do
  match (← getMCtx).findDecl? mvarId with
  | none          => return "unknown goal"
  | some mvarDecl =>
    let indent         := 2
    let lctx           := mvarDecl.lctx
    let lctx           := lctx.sanitizeNames.run' { options := (← getOptions) }
    Meta.withLCtx lctx mvarDecl.localInstances do
      -- The followint two `let rec`s are being used to control the generated code size.
      -- Then should be remove after we rewrite the compiler in Lean
      let rec pushPending (ids : List Name) (type? : Option Expr) (s : String) : MetaM String := do
        if ids.isEmpty then
          return s
        else
          let s := addLine s
          match type? with
          | none      => return s
          | some type =>
            let typeFmt ← Meta.ppExpr type
            return (s ++ (Format.joinSep ids.reverse (format " ") ++ " :" ++ Format.nest indent (Format.line ++ typeFmt)).group).pretty
      let rec ppVars (varNames : List Name) (prevType? : Option Expr) (s : String) (localDecl : LocalDecl) : MetaM (List Name × Option Expr × String) := do
        match localDecl with
        | .cdecl _ _ varName type _ _ =>
          let varName := varName.simpMacroScopes
          let type ← instantiateMVars type
          if prevType? == none || prevType? == some type then
            return (varName :: varNames, some type, s)
          else do
            let s ← pushPending varNames prevType? s
            return ([varName], some type, s)
        | .ldecl _ _ varName type val _ _ => do
          let varName := varName.simpMacroScopes
          let s ← pushPending varNames prevType? s
          let s  := addLine s
          let type ← instantiateMVars type
          let typeFmt ← Meta.ppExpr type
          let mut fmtElem  := format varName ++ " : " ++ typeFmt
          let val ← instantiateMVars val
          let valFmt ← Meta.ppExpr val
          fmtElem := fmtElem ++ " :=" ++ Format.nest indent (Format.line ++ valFmt)
          let s := s ++ fmtElem.group.pretty
          return ([], none, s)
      let (varNames, type?, s) ← lctx.foldlM (init := ([], none, "")) fun (varNames, prevType?, s) (localDecl : LocalDecl) =>
         if localDecl.isAuxDecl || localDecl.isImplementationDetail then
           -- Ignore auxiliary declarations and implementation details.
           return (varNames, prevType?, s)
         else
           ppVars varNames prevType? s localDecl
      let s ← pushPending varNames type? s
      let goalTypeFmt ← Meta.ppExpr (← instantiateMVars mvarDecl.type)
      let goalFmt := Meta.getGoalPrefix mvarDecl ++ Format.nest indent goalTypeFmt
      let s := s ++ "\n" ++ goalFmt.pretty
      match mvarDecl.userName with
      | Name.anonymous => return s
      | name           => return "case " ++ name.eraseMacroScopes.toString ++ "\n" ++ s


def ppGoals (ctx : ContextInfo) (goals : List MVarId) : IO String :=
  if goals.isEmpty then
    return "no goals"
  else
    let fmt := ctx.runMetaM {} (return Std.Format.prefixJoin "\n\n" (← goals.mapM (ppGoal ·)))
    return (← fmt).pretty.trim


end Pp


namespace Path


def relativeTo (path parent : FilePath) : Option FilePath :=
  let rec componentsRelativeTo (pathComps parentComps : List String) : Option FilePath :=
    match pathComps, parentComps with
    | _, [] => mkFilePath pathComps
    | [], _ => none
    | (h₁ :: t₁), (h₂ :: t₂) =>
      if h₁ == h₂ then
        componentsRelativeTo t₁ t₂
      else
        none

    componentsRelativeTo path.components parent.components


def isRelativeTo (path parent : FilePath) : Bool :=
  match relativeTo path parent  with
  | some _ => true
  | none => false


def toAbsolute (path : FilePath) : IO FilePath := do
  if path.isAbsolute then
    pure path
  else
    let cwd ← IO.currentDir
    pure $ cwd / path

private def trim (path : FilePath) : FilePath :=
  assert! path.isRelative
  match path.components with
  | "." :: tl => mkFilePath tl
  | _ => path


def toBuildDir (subDir: String) (path : FilePath) (ext : String) : Option FilePath :=
  let path' := (trim path).withExtension ext
  match relativeTo path' "lake-packages/lean4/src" with
  | some p => some $ "lake-packages/lean4/lib" / p
  | none => match relativeTo path' "lake-packages" with
    | some p =>
      match p.components with
      | [] => none
      | hd :: tl => some $ "lake-packages" / hd / "build" / subDir / (mkFilePath tl)
    | none => some $ "build" / subDir / path'


-- The reverse of `toBuildDir`.
def toSrcDir (path : FilePath) (ext : String) : Option FilePath :=
  let path' := (trim path).withExtension ext
  match relativeTo path' "lake-packages/lean4/lib" with
  | some p => some $ "lake-packages/lean4/src" / p
  | none =>
    match relativeTo path' "lake-packages" with
    | some p =>
      let comps := p.components
      assert! comps[1]! == "build"
      match comps with
      | _ :: _ :: _ :: tl => mkFilePath tl
      | _ => "invalid path"
    | none =>
    let comps := path'.components
      assert! comps[0]! == "build"
      match comps with
      | _ :: _ :: tl => mkFilePath tl
      | _ => "invalid path"


end Path


namespace Traversal


private def visitTacticInfo (ctx : ContextInfo) (ti : TacticInfo) (parent : InfoTree) : TraceM Unit := do
  match ti.stx.getKind with
  | `Lean.Parser.Term.byTactic =>
    match ti.stx with
    | .node _ _ #[.atom _ "by", .node _ `Lean.Parser.Tactic.tacticSeq _] => pure ()
    | _ => assert! false

  | `Lean.Parser.Tactic.tacticSeq =>
    match ti.stx with
    | .node _ _ #[.node _ `Lean.Parser.Tactic.tacticSeq1Indented _] => pure ()
    | .node _ _ #[.node _ `Lean.Parser.Tactic.tacticSeqBracketed _] => pure ()
    | _ => assert! false

  | _ => pure ()

  match parent with
  | .node (Info.ofTacticInfo i) _ =>
    match i.stx.getKind with
    | `Lean.Parser.Tactic.tacticSeq1Indented | `Lean.Parser.Tactic.tacticSeqBracketed =>
      let ctxBefore := { ctx with mctx := ti.mctxBefore }
      let ctxAfter := { ctx with mctx := ti.mctxAfter }
      let stateBefore ← Pp.ppGoals ctxBefore ti.goalsBefore
      let stateAfter ← Pp.ppGoals ctxAfter ti.goalsAfter
      let some posBefore := ti.stx.getPos? true | pure ()
      let some posAfter := ti.stx.getTailPos? true | pure ()
      match ti.stx with
      | .node _ _ _ =>
        modifyGet fun trace => ((),
          { trace with tactics := trace.tactics.push { stateBefore := stateBefore, stateAfter := stateAfter, pos := posBefore, endPos := posAfter } }
        )
      | _ => pure ()
    | _ => pure ()
  | _ => pure ()


private def visitInfo (ctx : ContextInfo) (i : Info) (parent : InfoTree) : TraceM Unit := do
  match i with
  | .ofTacticInfo ti => visitTacticInfo ctx ti parent
  | _ => pure ()


private partial def traverseTree (ctx: ContextInfo) (tree : InfoTree) (parent : InfoTree) : TraceM Unit := do
  match tree with
  | .context ctx' t => traverseTree ctx' t tree
  | .node i children =>
    visitInfo ctx i parent
    for x in children do
      traverseTree ctx x tree
  | _ => pure ()


private def traverseTopLevelTree (tree : InfoTree) : TraceM Unit := do
  match tree with
  | .context ctx t => traverseTree ctx t tree
  | _ => throw $ IO.userError "Errors in traverseTopLevelTree; aborting"


def traverseForest (trees : Array InfoTree) : TraceM Trace := do
  for t in trees do
    traverseTopLevelTree t
  get


end Traversal


unsafe def processFile (path : FilePath) : IO Unit := do
  println! path
  let input ← IO.FS.readFile path
  let opts := Options.empty.setBool `trace.Elab.info true
  enableInitializersExecution
  let inputCtx := Parser.mkInputContext input path.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx

  let (env, messages) ← processHeader header opts messages inputCtx

  if messages.hasErrors then
    for msg in messages.toList do
      if msg.severity == .error then
        println! "ERROR: {← msg.toString}"
    throw $ IO.userError "Errors during import; aborting"

  let some modName := path.fileStem | throw $ IO.userError s!"Invalid path: {path}"
  let env := env.setMainModule modName.toName
  let commandState := { Command.mkState env messages opts with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState
  let commands := s.commands.pop -- Remove EOI command.
  let trees := s.commandState.infoState.trees.toArray
  let trace ← (Traversal.traverseForest trees).run' ⟨#[header] ++ commands, #[]⟩

  let cwd ← IO.currentDir
  let is_lean := cwd.fileName == "lean4"

  let some relativePath := Path.relativeTo path cwd | throw $ IO.userError s!"Invalid path: {path}"
  let json_path := (
    if is_lean then
      mkFilePath $ "lib" :: (relativePath.withExtension "ast.json").components.tail!
    else
      (Path.toBuildDir "ir" relativePath "ast.json").get!
  )
  IO.FS.writeFile json_path (toJson trace).pretty

  -- Print imports, similar to `lean --deps` in Lean 3.
  let mut s := ""
  for dep in headerToImports header do
    let oleanPath ← findOLean dep.module
    if oleanPath.isRelative then
      let some leanPath := Path.toSrcDir oleanPath "lean" | throw $ IO.userError s!"Invalid path: {oleanPath}"
      s := s ++ "\n" ++ leanPath.toString
    else if !(oleanPath.toString.endsWith "/lib/lean/Init.olean") then
      s := s ++ "\n"
      if !is_lean then
        s := s ++ "lake-packages/lean4/"
      let mut found := false
      for c in (oleanPath.withExtension "lean").components do
        if c == "lib" then
          found := true
          s := s ++ "src"
          continue
        if found then
          s := s ++ FilePath.pathSeparator.toString ++ c

  let dep_path := (
    if is_lean then
      mkFilePath $ "lib" :: (relativePath.withExtension "dep_paths").components.tail!
    else
      (Path.toBuildDir "ir" relativePath "dep_paths").get!
  )
  IO.FS.writeFile dep_path s.trim


end LeanDojo

open LeanDojo


def shouldProcess (path : FilePath) : IO Bool := do
  if path.extension != "lean" then return false
  let cwd ← IO.currentDir
  let some relativePath := Path.relativeTo path cwd | throw $ IO.userError s!"Invalid path: {path}"
  if cwd.fileName == "lean4" then
    let oleanPath := mkFilePath $ "lib" :: (relativePath.withExtension "olean").components.tail!
    return ← oleanPath.pathExists
  else
    let some oleanPath := Path.toBuildDir "lib" relativePath "olean" | throw $ IO.userError s!"Invalid path: {path}"
    return ← oleanPath.pathExists


unsafe def main (args : List String) : IO Unit := do
  match args with
  | [] =>
    let cwd ← IO.currentDir
    println! "Extracting data at {cwd}"
    let _ ← System.FilePath.walkDir cwd fun dir => do
      for p in ← System.FilePath.readDir dir do
        if ← shouldProcess p.path then
          let _ ← IO.asTask $ IO.Process.run
            (if cwd.fileName != "lean4" then
              {cmd := "lake", args := #["env", "lean", "--run", "ExtractData.lean", p.path.toString]}
            else
              {cmd := "./build/release/stage1/bin/lean", args := #["--run", "ExtractData.lean", p.path.toString]})
          println! p.path
      pure true
  | path :: _ =>
    processFile (← Path.toAbsolute ⟨path⟩)

import Lean
import Lake


open Lean Elab System

set_option maxHeartbeats 2000000  -- 10x the default maxHeartbeats.


instance : ToJson Substring where
  toJson s := toJson s.toString

instance : ToJson String.Pos where
  toJson n := toJson n.1

deriving instance ToJson for SourceInfo
deriving instance ToJson for Syntax.Preresolved
deriving instance ToJson for Syntax
deriving instance ToJson for Position


namespace LeanDojo


/--
The trace of a tactic.
-/
structure TacticTrace where
  stateBefore: String
  stateAfter: String
  pos: String.Pos      -- Start position of the tactic.
  endPos: String.Pos   -- End position of the tactic.
deriving ToJson


/--
The trace of a premise.
-/
structure PremiseTrace where
  fullName: String            -- Fully-qualified name of the premise.
  defPos: Option Position     -- Where the premise is defined.
  defEndPos: Option Position
  modName: String             -- In which module the premise is defined.
  defPath: String             -- The path of the file where the premise is defined.
  pos: Option Position        -- Where the premise is used.
  endPos: Option Position
deriving ToJson


/--
The trace of a Lean file.
-/
structure Trace where
  commandASTs : Array Syntax    -- The ASTs of the commands in the file.
  tactics: Array TacticTrace    -- All tactics in the file.
  premises: Array PremiseTrace  -- All premises in the file.
deriving ToJson


abbrev TraceM := StateT Trace MetaM


namespace Pp


private def addLine (s : String) : String :=
  if s.isEmpty then s else s ++ "\n"


-- Similar to `Meta.ppGoal` but uses String instead of Format to make sure local declarations are separated by "\n".
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

/--
Return the path of `path` relative to `parent`.
-/
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


/--
Return if the path `path` is relative to `parent`.
-/
def isRelativeTo (path parent : FilePath) : Bool :=
  match relativeTo path parent with
  | some _ => true
  | none => false


/--
Convert the path `path` to an absolute path.
-/
def toAbsolute (path : FilePath) : IO FilePath := do
  if path.isAbsolute then
    pure path
  else
    let cwd ← IO.currentDir
    pure $ cwd / path


private def trim (path : FilePath) : FilePath :=
  assert! path.isRelative
  mkFilePath $ path.components.filter (· != ".")


def packagesDir : FilePath :=
  if Lake.defaultPackagesDir == "packages"  then
    ".lake" / Lake.defaultPackagesDir
  else
    Lake.defaultPackagesDir


def buildDir : FilePath :=
  if Lake.defaultPackagesDir.fileName == "packages" then  -- Lean >= v4.3.0-rc2
    ".lake/build"
  else  -- Lean < v4.3.0-rc2
   "build"


def libDir : FilePath := buildDir / "lib" / "lean"


/--
Convert the path of a *.lean file to its corresponding file (e.g., *.olean) in the "build" directory.
-/
def toBuildDir (subDir : FilePath) (path : FilePath) (ext : String) : Option FilePath :=
  let path' := (trim path).withExtension ext
  match relativeTo path' $ packagesDir / "lean4/src" with
  | some p =>
    match relativeTo p "lean/lake" with
    | some p' => packagesDir / "lean4/lib/lean" / p'
    | none => packagesDir / "lean4/lib" / p
  | none => match relativeTo path' packagesDir with
    | some p =>
      match p.components with
      | [] => none
      | hd :: tl => packagesDir / hd / buildDir / subDir / (mkFilePath tl)
    | none => buildDir / subDir / path'


/--
The reverse of `toBuildDir`.
-/
-- proofwidgets/build/lib/ProofWidgets/Compat.lean
-- proofwidgets/.lake/build/lib
def toSrcDir! (path : FilePath) (ext : String) : FilePath :=
  let path' := (trim path).withExtension ext
  match relativeTo path' $ packagesDir / "lean4/lib" with
  | some p =>  -- E.g., `.lake/packages/lean4/lib/lean/Init/Prelude.olean` -> `.lake/packages/lean4/src/lean/Init/Prelude.lean`
    packagesDir / "lean4/src" / p
  | none =>
    match relativeTo path' packagesDir with
    | some p =>  -- E.g., `.lake/packages/aesop/.lake/build/lib/lean/Aesop.olean`-> `.lake/packages/aesop/Aesop.lean`
      let pkgName := p.components.head!
      let sep := "build/lib/lean/"
      packagesDir / pkgName / (p.toString.splitOn sep |>.tail!.head!)
    | none =>
      -- E.g., `.lake/build/lib/lean/Mathlib/LinearAlgebra/Basic.olean` -> `Mathlib/LinearAlgebra/Basic.lean`
      relativeTo path' libDir |>.get!


/--
Create all parent directories of `p` if they don't exist.
-/
def makeParentDirs (p : FilePath) : IO Unit := do
  let some parent := p.parent | throw $ IO.userError s!"Unable to get the parent of {p}"
  IO.FS.createDirAll parent


/--
Return the *.lean file corresponding to a module name.
-/
def findLean (mod : Name) : IO FilePath := do
  let modStr := mod.toString
  if modStr.startsWith "«lake-packages»." then
    return FilePath.mk (modStr.replace "«lake-packages»" "lake-packages" |>.replace "." "/") |>.withExtension "lean"
  if modStr.startsWith "«.lake»." then
    return FilePath.mk (modStr.replace "«.lake»" ".lake" |>.replace "." "/") |>.withExtension "lean"
  if modStr == "Lake" then
    return packagesDir / "lean4/src/lean/lake/Lake.lean"
  let olean ← findOLean mod
  -- Remove a "build/lib/lean/" substring from the path.
  let lean := olean.toString.replace ".lake/build/lib/lean/" ""
    |>.replace "build/lib/lean/" "" |>.replace "lib/lean/Lake/" "lib/lean/lake/Lake/"
  let mut path := FilePath.mk lean |>.withExtension "lean"
  let leanLib ← getLibDir (← getBuildDir)
  if let some p := relativeTo path leanLib then
    path := packagesDir / "lean4/src/lean" / p
  assert! ← path.pathExists
  return path

end Path


namespace Traversal


/--
Extract tactic information from `TacticInfo` in `InfoTree`.
-/
private def visitTacticInfo (ctx : ContextInfo) (ti : TacticInfo) (parent : InfoTree) : TraceM Unit := do
  match ti.stx.getKind with
  | ``Lean.Parser.Term.byTactic =>
    match ti.stx with
    | .node _ _ #[.atom _ "by", .node _ ``Lean.Parser.Tactic.tacticSeq _] => pure ()
    | _ => assert! false

  | ``Lean.Parser.Tactic.tacticSeq =>
    match ti.stx with
    | .node _ _ #[.node _ ``Lean.Parser.Tactic.tacticSeq1Indented _] => pure ()
    | .node _ _ #[.node _ ``Lean.Parser.Tactic.tacticSeqBracketed _] => pure ()
    | _ => assert! false

  | _ => pure ()

  match parent with
  | .node (Info.ofTacticInfo i) _ =>
    match i.stx.getKind with
    | ``Lean.Parser.Tactic.tacticSeq1Indented | ``Lean.Parser.Tactic.tacticSeqBracketed | ``Lean.Parser.Tactic.rewriteSeq =>
      let ctxBefore := { ctx with mctx := ti.mctxBefore }
      let ctxAfter := { ctx with mctx := ti.mctxAfter }
      let stateBefore ← Pp.ppGoals ctxBefore ti.goalsBefore
      let stateAfter ← Pp.ppGoals ctxAfter ti.goalsAfter
      if stateBefore == "no goals" || stateBefore == stateAfter then
        pure ()
      else
        let some posBefore := ti.stx.getPos? true | pure ()
        let some posAfter := ti.stx.getTailPos? true | pure ()
        match ti.stx with
        | .node _ _ _ =>
          modify fun trace => {
            trace with tactics := trace.tactics.push {
              stateBefore := stateBefore,
              stateAfter := stateAfter,
              pos := posBefore,
              endPos := posAfter,
             }
          }
        | _ => pure ()
    | _ => pure ()
  | _ => pure ()


/--
Extract premise information from `TermInfo` in `InfoTree`.
-/
private def visitTermInfo (ti : TermInfo) (env : Environment) : TraceM Unit := do
  let some fullName := ti.expr.constName? | return ()
  let fileMap ← getFileMap

  let posBefore := match ti.toElabInfo.stx.getPos? with
    | some posInfo => fileMap.toPosition posInfo
    | none => none

  let posAfter := match ti.toElabInfo.stx.getTailPos? with
    | some posInfo => fileMap.toPosition posInfo
    | none => none

  let decRanges ← withEnv env $ findDeclarationRanges? fullName
  let defPos := decRanges >>= fun (decR : DeclarationRanges) => decR.selectionRange.pos
  let defEndPos := decRanges >>= fun (decR : DeclarationRanges) => decR.selectionRange.endPos

  let modName :=
  if let some modIdx := env.const2ModIdx.get? fullName then
    env.header.moduleNames[modIdx.toNat]!
  else
    env.header.mainModule

  let mut defPath := toString $ ← Path.findLean modName
  while defPath.startsWith "./" do
    defPath := defPath.drop 2
  if defPath.startsWith "/lake/" then
    defPath := ".lake/" ++ (defPath.drop 6)

  if defPos != posBefore ∧ defEndPos != posAfter then  -- Don't include defintions as premises.
    modify fun trace => {
        trace with premises := trace.premises.push {
          fullName := toString fullName,
          defPos := defPos,
          defEndPos := defEndPos,
          defPath := defPath,
          modName := toString modName,
          pos := posBefore,
          endPos := posAfter,
        }
    }


private def visitInfo (ctx : ContextInfo) (i : Info) (parent : InfoTree) (env : Environment) : TraceM Unit := do
  match i with
  | .ofTacticInfo ti => visitTacticInfo ctx ti parent
  | .ofTermInfo ti => visitTermInfo ti env
  | _ => pure ()


private partial def traverseTree (ctx: ContextInfo) (tree : InfoTree)
(parent : InfoTree) (env : Environment) : TraceM Unit := do
  match tree with
  | .context ctx' t =>
    match ctx'.mergeIntoOuter? ctx with
    | some ctx' => traverseTree ctx' t tree env
    | none => panic! "fail to synthesis contextInfo when traversing infoTree"
  | .node i children =>
    visitInfo ctx i parent env
    for x in children do
      traverseTree ctx x tree env
  | _ => pure ()


private def traverseTopLevelTree (tree : InfoTree) (env : Environment) : TraceM Unit := do
  match tree with
  | .context ctx t =>
    match ctx.mergeIntoOuter? none with
    | some ctx => traverseTree ctx t tree env
    | none => panic! "fail to synthesis contextInfo for top-level infoTree"
  | _ => pure ()


/--
Process an array of `InfoTree` (one for each top-level command in the file).
-/
def traverseForest (trees : Array InfoTree) (env : Environment) : TraceM Trace := do
  for t in trees do
    traverseTopLevelTree t env
  get


end Traversal


open Traversal


def getImports (header: TSyntax `Lean.Parser.Module.header) : IO String := do
  -- Similar to `lean --deps` in Lean 3.
  let mut s := ""

  for dep in headerToImports header do
    -- let oleanPath ← findOLean dep.module
    let leanPath ← Path.findLean dep.module
    s := s ++ "\n" ++ leanPath.toString
    /-
    if oleanPath.isRelative then
      let leanPath := Path.toSrcDir! oleanPath "lean"
      assert! ← leanPath.pathExists
      s := s ++ "\n" ++ leanPath.toString
    else if ¬(oleanPath.toString.endsWith "/lib/lean/Init.olean") then
      let mut p := (Path.packagesDir / "lean4").toString ++ FilePath.pathSeparator.toString
      let mut found := false
      for c in (oleanPath.withExtension "lean").components do
        if c == "lib" then
          found := true
          p := p ++ "src"
          continue
        if found then
          p := p ++ FilePath.pathSeparator.toString ++ c
      p := p.replace "/lean4/src/lean/Lake" "/lean4/src/lean/lake/Lake"
      assert! ← FilePath.mk p |>.pathExists
      s := s ++ "\n" ++ p
  -/

  return s.trim


/--
Trace a *.lean file.
-/
unsafe def processFile (path : FilePath) : IO Unit := do
  println! path
  let input ← IO.FS.readFile path
  enableInitializersExecution
  let inputCtx := Parser.mkInputContext input path.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let (env, messages) ← processHeader header {} messages inputCtx

  if messages.hasErrors then
    for msg in messages.toList do
      if msg.severity == .error then
        println! "ERROR: {← msg.toString}"
    throw $ IO.userError "Errors during import; aborting"

  let env := env.setMainModule (← moduleNameOfFileName path none)
  let commandState := { Command.mkState env messages {} with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState
  let env' := s.commandState.env
  let commands := s.commands.pop -- Remove EOI command.
  let trees := s.commandState.infoState.trees.toArray

  let traceM := (traverseForest trees env').run' ⟨#[header] ++ commands, #[], #[]⟩
  let (trace, _) ← traceM.run'.toIO {fileName := s!"{path}", fileMap := FileMap.ofString input} {env := env}

  let cwd ← IO.currentDir
  assert! cwd.fileName != "lean4"

  let some relativePath := Path.relativeTo path cwd | throw $ IO.userError s!"Invalid path: {path}"
  let json_path := Path.toBuildDir "ir" relativePath "ast.json" |>.get!
  Path.makeParentDirs json_path
  IO.FS.writeFile json_path (toJson trace).pretty

  let dep_path := Path.toBuildDir "ir" relativePath "dep_paths" |>.get!
  Path.makeParentDirs dep_path
  IO.FS.writeFile dep_path (← getImports header)


end LeanDojo


open LeanDojo

/--
Whether a *.lean file should be traced.
-/
def shouldProcess (path : FilePath) (noDeps : Bool) : IO Bool := do
  if (← path.isDir) ∨ path.extension != "lean" then
    return false

  let cwd ← IO.currentDir
  let some relativePath := Path.relativeTo path cwd |
    throw $ IO.userError s!"Invalid path: {path}"

  if noDeps ∧ Path.isRelativeTo relativePath Path.packagesDir then
    return false

  let some oleanPath := Path.toBuildDir "lib/lean" relativePath "olean" |
    throw $ IO.userError s!"Invalid path: {path}"
  return ← oleanPath.pathExists


/--
Trace all *.lean files in the current directory whose corresponding *.olean file exists.
-/
def processAllFiles (noDeps : Bool) : IO Unit := do
    let cwd ← IO.currentDir
    assert! cwd.fileName != "lean4"
    println! "Extracting data at {cwd}"

    let mut tasks := #[]
    for path in ← System.FilePath.walkDir cwd do
      if ← shouldProcess path noDeps then
        let t ← IO.asTask $ IO.Process.run
          {cmd := "lake", args := #["env", "lean", "--run", "ExtractData.lean", path.toString]}
        tasks := tasks.push (t, path)

    for (t, path) in tasks do
      match ← IO.wait t with
      | Except.error _ =>
        println! s!"WARNING: Failed to process {path}"
        pure ()
        -- throw e
      | Except.ok _ => pure ()


unsafe def main (args : List String) : IO Unit := do
  match args with
  | ["noDeps"] => processAllFiles (noDeps := true)
  | [path] => processFile (← Path.toAbsolute ⟨path⟩)
  | [] => processAllFiles (noDeps := false)
  | _ => throw $ IO.userError "Invalid arguments"

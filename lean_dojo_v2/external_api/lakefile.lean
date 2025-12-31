import Lake
open Lake DSL

package external_api where

@[default_target]
lean_lib ExternalAPI where

lean_lib «LeanCopilot» where
  srcDir := "."

require batteries from git "https://github.com/leanprover-community/batteries.git" @ "main"

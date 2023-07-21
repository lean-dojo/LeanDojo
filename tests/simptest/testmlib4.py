from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "5a919533f110b7d76410134a237ee374f24eaaad")
trace(repo, dst_dir="traced_mathlib4")
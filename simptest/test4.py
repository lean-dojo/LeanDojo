from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "7d711f6da4584ecb7d4f057715e1f72ba175c910")
trace(repo, dst_dir="traced_lean4-example")
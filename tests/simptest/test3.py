from lean_dojo import LeanGitRepo, trace

repo = LeanGitRepo("https://github.com/yangky11/lean-example", "5a0360e49946815cb53132638ccdd46fb1859e2a")
trace(repo, dst_dir="traced_lean-example")
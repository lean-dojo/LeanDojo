# test for the class `LeanGitRepo`
from lean_dojo import LeanGitRepo

def test_lean_git_repo(lean4_example_url, example_commit_hash):
    repo = LeanGitRepo(lean4_example_url, example_commit_hash)
    assert repo.url == lean4_example_url
    assert repo.commit == example_commit_hash
    assert repo.exists()
    assert repo.name == "lean4-example"
    assert repo.commit_url == f"{lean4_example_url}/tree/{example_commit_hash}"
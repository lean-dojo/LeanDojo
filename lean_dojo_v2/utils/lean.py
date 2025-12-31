"""
Utility functions for Lean version management and compatibility checking.
"""

import re
from pathlib import Path

from lean_dojo_v2.utils.constants import LEAN4_BUILD_DIR, LEAN4_PACKAGES_DIR

# Regex pattern for parsing Lean 4 toolchain versions
_LEAN4_VERSION_REGEX = re.compile(
    r"leanprover/lean4:(?P<version>v\d+\.\d+\.\d+(?:-rc\d+)?)"
)


def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    return m["version"]


def is_supported_version(v: str) -> bool:
    """
    Check if ``v`` is at least `v4.3.0` and at most `v4.30.0`.
    Note: Lean versions are generally not backwards-compatible. Also, the Lean FRO
    keeps bumping the default versions of repos to the latest version, which is
    not necessarily the latest stable version. So, we need to be careful about
    what we choose to support.
    """
    max_version = 30
    min_version = 3
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if (
        major < 4
        or (major == 4 and minor < min_version)
        or (major == 4 and minor > max_version)
        or (major == 4 and minor == max_version and patch > 1)
    ):
        return False
    if (
        major > 4
        or (major == 4 and minor > min_version)
        or (major == 4 and minor == min_version and patch > 0)
    ):
        return True


def _from_lean_path(root_dir: Path, path: Path, repo, ext: str) -> Path:
    assert path.suffix == ".lean"
    if path.is_absolute():
        path = path.relative_to(root_dir)

    assert root_dir.name != "lean4"
    if path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/src/lean/lake"):
        # E.g., "lake-packages/lean4/src/lean/lake/Lake/CLI/Error.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/src/lean/lake")
        return LEAN4_PACKAGES_DIR / "lean4/lib/lean" / p.with_suffix(ext)
    elif path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/src"):
        # E.g., "lake-packages/lean4/src/lean/Init.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/src").with_suffix(ext)
        return LEAN4_PACKAGES_DIR / "lean4/lib" / p
    elif path.is_relative_to(LEAN4_PACKAGES_DIR):
        # E.g., "lake-packages/std/Std.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR).with_suffix(ext)
        repo_name = p.parts[0]
        return (
            LEAN4_PACKAGES_DIR
            / repo_name
            / LEAN4_BUILD_DIR
            / "ir"
            / p.relative_to(repo_name)
        )
    else:
        # E.g., "Mathlib/LinearAlgebra/Basics.lean"
        return LEAN4_BUILD_DIR / "ir" / path.with_suffix(ext)


def to_xml_path(root_dir: Path, path: Path, repo) -> Path:
    return _from_lean_path(root_dir, path, repo, ext=".trace.xml")


def to_dep_path(root_dir: Path, path: Path, repo) -> Path:
    return _from_lean_path(root_dir, path, repo, ext=".dep_paths")


def to_json_path(root_dir: Path, path: Path, repo) -> Path:
    return _from_lean_path(root_dir, path, repo, ext=".ast.json")


def to_lean_path(root_dir: Path, path: Path, repo) -> Path:
    if path.is_absolute():
        path = path.relative_to(root_dir)

    if path.suffix in (".xml", ".json"):
        path = path.with_suffix("").with_suffix(".lean")
    else:
        assert path.suffix == ".dep_paths"
        path = path.with_suffix(".lean")

    assert root_dir.name != "lean4"
    if path == LEAN4_PACKAGES_DIR / "lean4/lib/lean/Lake.lean":
        return LEAN4_PACKAGES_DIR / "lean4/src/lean/lake/Lake.lean"
    elif path == LEAN4_PACKAGES_DIR / "lean4/lib/lean/LakeMain.lean":
        return LEAN4_PACKAGES_DIR / "lean4/src/lean/lake/LakeMain.lean"
    elif path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/lib/lean/Lake"):
        # E.g., "lake-packages/lean4/lib/lean/Lake/Util/List.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/lib/lean/Lake")
        return LEAN4_PACKAGES_DIR / "lean4/src/lean/lake/Lake" / p
    elif path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/lib"):
        # E.g., "lake-packages/lean4/lib/lean/Init.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/lib")
        return LEAN4_PACKAGES_DIR / "lean4/src" / p
    elif path.is_relative_to(LEAN4_PACKAGES_DIR):
        # E.g., "lake-packages/std/build/ir/Std.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR)
        repo_name = p.parts[0]
        return (
            LEAN4_PACKAGES_DIR
            / repo_name
            / p.relative_to(Path(repo_name) / LEAN4_BUILD_DIR / "ir")
        )
    else:
        # E.g., ".lake/build/ir/Mathlib/LinearAlgebra/Basics.lean" or "build/ir/Mathlib/LinearAlgebra/Basics.lean"
        assert path.is_relative_to(LEAN4_BUILD_DIR / "ir"), path
        return path.relative_to(LEAN4_BUILD_DIR / "ir")
    assert major == 4 and minor == min_version and patch == 0
    return True

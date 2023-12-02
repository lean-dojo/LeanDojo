"""This module define classes for repos, files, and theorems in Lean. 
Objects of these classes contain only surface information, without extracting any trace.
"""
import re
import os
import toml
import urllib
import webbrowser
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from github.Repository import Repository
from typing import List, Dict, Any, Generator, Union, Optional, Tuple
from urllib import error

from ..utils import (
    execute,
    read_url,
    url_exists,
    url_to_repo,
    normalize_url,
    get_repo_info,
    working_directory,
    get_latest_commit,
)
from ..constants import (
    LEAN3_URL,
    LEAN4_URL,
    LEAN4_REPO,
    LEAN4_NIGHTLY_REPO,
    LEAN3_PACKAGES_DIR,
    LEAN4_PACKAGES_DIR,
    LEAN4_PACKAGES_DIR_OLD,
    LEAN4_BUILD_DIR,
    LEAN_BUILD_DIR_OLD,
)


def _to_commit_hash(repo: Repository, label: str) -> str:
    """Convert a tag or branch to a commit hash."""
    for branch in repo.get_branches():
        if branch.name == label:
            return branch.commit.sha

    for tag in repo.get_tags():
        if tag.name == label:
            return tag.commit.sha

    raise ValueError(f"Invalid tag or branch: `{label}` for {repo}")


@dataclass(eq=True, unsafe_hash=True)
class Pos:
    """Position in source files.

    We use 1-index to keep it consistent with code editors such as Visual Studio Code.
    """

    line_nb: int
    """Line number
    """

    column_nb: int
    """Column number
    """

    @classmethod
    def from_str(cls, s: str) -> "Pos":
        """Construct a :class:`Pos` object from its string representation, e.g., :code:`"(323, 1109)"`."""
        assert s.startswith("(") and s.endswith(
            ")"
        ), f"Invalid string representation of a position: {s}"
        line, column = s[1:-1].split(",")
        line_nb = int(line)
        column_nb = int(column)
        return cls(line_nb, column_nb)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.line_nb
        yield self.column_nb

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __lt__(self, other):
        return self.line_nb < other.line_nb or (
            self.line_nb == other.line_nb and self.column_nb < other.column_nb
        )

    def __le__(self, other):
        return self < other or self == other


@dataclass(frozen=True)
class LeanFile:
    """A Lean source file (:file:`*.lean`)."""

    root_dir: Path = field(repr=False)
    """Root directory of the traced repo this :class:`LeanFile` object belongs to.

    ``root_dir`` must be an absolute path, e.g., :file:`/home/kaiyu/traced_lean-example/lean-example`
    """

    path: Path
    """Relative path w.r.t. ``root_dir``
    
    E.g., :file:`lean-example/src/example.lean`
    """

    uses_lean4: bool
    """Whether this file is written in Lean 4."""

    code: List[str] = field(init=False, repr=False)
    """Raw source code as a list of lines."""

    endwith_newline: bool = field(init=False, repr=False)
    """Whether the last line ends with a newline."""

    num_bytes: List[int] = field(init=False, repr=False)
    """The number of UTF-8 bytes of each line, including newlines.
    """

    def __post_init__(self) -> None:
        assert (
            self.root_dir.is_absolute()
        ), f"Root directory must be an absolute path: {self.root_dir}"
        assert self.path.suffix == ".lean", f"File extension must be .lean: {self.path}"
        assert not self.path.is_absolute(), f"Path must be a relative path: {self.path}"

        code = []
        endwith_newline = None
        num_bytes = []

        for line in self.abs_path.open("rb"):
            if b"\r\n" in line:
                raise RuntimeError(
                    f"{self.abs_path} contains Windows-style line endings. This is discouraged (see https://github.com/leanprover-community/mathlib4/pull/6506)."
                )
            if line.endswith(b"\n"):
                endwith_newline = True
                line = line[:-1]
            else:
                endwith_newline = False
            code.append(line.decode("utf-8"))
            num_bytes.append(len(line) + 1)

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "endwith_newline", endwith_newline)
        object.__setattr__(self, "num_bytes", num_bytes)

    @property
    def abs_path(self) -> Path:
        """Absolute path of a :class:`LeanFile` object.

        E.g., :file:`/home/kaiyu/traced_lean-example/lean-example/src/example.lean`
        """
        return self.root_dir / self.path

    @property
    def num_lines(self) -> int:
        """Number of lines in a source file."""
        return len(self.code)

    def num_columns(self, line_nb: int) -> int:
        """Number of columns in a source file."""
        return len(self.get_line(line_nb))

    def start_pos(self, zero_indexed: bool = False) -> Pos:
        """Return the start position of a source file.

        Args:
            zero_indexed (bool, optional): Whether to use 0-index instead of 1-index. Defaults to False.

        Returns:
            Pos: A :class:`Pos` object representing the start of this file.
        """
        if zero_indexed:
            return Pos(0, 0)
        else:
            return Pos(1, 1)

    def end_pos(self, zero_indexed: bool = False) -> Pos:
        """Return the end position of a source file.

        Args:
            zero_indexed (bool, optional): Whether to use 0-index instead of 1-index. Defaults to False.

        Returns:
            Pos: A :class:`Pos` object representing the end of this file.
        """
        # Line and column numbers are 1-indexed by default.
        line_nb = self.num_lines - 1
        column_nb = len(self.code[-1])
        if not zero_indexed:
            line_nb += 1
            column_nb += 1
        return Pos(line_nb, column_nb)

    def convert_pos(self, byte_idx: int) -> Pos:
        """Convert a byte index (:code:`String.Pos` in Lean 4) to a :class:`Pos` object."""
        n = 0
        for i, num_bytes in enumerate(self.num_bytes, start=1):
            n += num_bytes
            if n > byte_idx:
                line_byte_idx = byte_idx - (n - num_bytes)
                if line_byte_idx == 0:
                    return Pos(i, 1)

                line = self.get_line(i)
                m = 0

                for j, c in enumerate(line, start=1):
                    m += len(c.encode("utf-8"))
                    if m >= line_byte_idx:
                        return Pos(i, j + 1)

        raise ValueError(f"Invalid byte index {byte_idx} in {self.path}.")

    def offset(self, pos: Pos, delta: int) -> Pos:
        """Off set a position by a given number."""
        line_nb, column_nb = pos
        num_columns = len(self.get_line(line_nb)) - column_nb + 1
        if delta <= num_columns:
            return Pos(line_nb, column_nb + delta)
        delta_left = delta - num_columns - 1

        for i in range(line_nb, self.num_lines):
            line = self.code[i]
            l = len(line)
            if delta_left <= l:
                return Pos(i + 1, delta_left + 1)
            delta_left -= l + 1

        if delta_left == 0 and self.endwith_newline:
            return Pos(self.num_lines + 1, 1)

        raise ValueError(f"Invalid offset {delta} in {self.path}: {pos}.")

    def get_line(self, line_nb: int) -> str:
        """Return a given line of the source file.

        Args:
            line_nb (int): Line number (1-indexed).
        """
        return self.code[line_nb - 1]

    def __getitem__(self, key) -> str:
        """Return a code segment given its start/end positions.

        This enables ``lean_file[start:end]``.

        Args:
            key (slice): A slice of two :class:`Pos` objects for the start/end of the code segment.
        """
        assert isinstance(key, slice) and key.step is None
        if key.start is None:
            start_line = start_column = 1
        else:
            start_line, start_column = key.start
        if key.stop is None:
            end_line = self.num_lines
            end_column = 1 + len(self.get_line(end_line))
        else:
            end_line, end_column = key.stop
        if start_line == end_line:
            assert start_column <= end_column
            return self.get_line(start_line)[start_column - 1 : end_column - 1]
        else:
            assert start_line < end_line
            code_slice = [self.code[start_line - 1][start_column - 1 :]]
            for line_nb in range(start_line + 1, end_line):
                code_slice.append(self.get_line(line_nb))
            code_slice.append(self.get_line(end_line)[: end_column - 1])
            return "\n".join(code_slice)


_COMMIT_REGEX = re.compile(r"[0-9a-z]+")
_LEAN3_VERSION_REGEX = re.compile(
    r"leanprover-community/lean:(?P<version>\d+\.\d+\.\d+)"
)
_LEAN4_VERSION_REGEX = re.compile(r"leanprover/lean4:(?P<version>.+?)")


def get_lean3_version_from_config(config: Dict[str, Any]) -> str:
    """Return the required Lean version given a ``leanpkg.toml`` config."""
    m = _LEAN3_VERSION_REGEX.fullmatch(config["package"]["lean_version"])
    assert m is not None, "Invalid config."
    return f"v{m['version']}"


def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    return m["version"]


def get_lean4_commit_from_config(config_dict: Dict[str, Any]) -> str:
    """Return the required Lean commit given a ``lean-toolchain`` config."""
    assert "content" in config_dict, "config_dict must have a 'content' field"
    config = config_dict["content"].strip()
    prefix = "leanprover/lean4:"

    if config == f"{prefix}nightly":
        latest_tag = LEAN4_NIGHTLY_REPO.get_tags()[0]
        return latest_tag.commit.sha

    assert config.startswith(prefix), f"Invalid Lean 4 version: {config}"
    version = config[len(prefix) :]

    if version.startswith("nightly"):
        return _to_commit_hash(LEAN4_NIGHTLY_REPO, version)
    else:
        return _to_commit_hash(LEAN4_REPO, version)


URL = TAG = COMMIT = str


@dataclass(frozen=True)
class RepoInfoCache:
    """To minize the number of network requests, we cache and re-use the info
    of all repos, assuming it does not change during the execution of LeanDojo."""

    tag2commit: Dict[Tuple[URL, TAG], COMMIT] = field(default_factory=dict)
    uses_lean3: Dict[Tuple[URL, COMMIT], bool] = field(default_factory=dict)
    uses_lean4: Dict[Tuple[URL, COMMIT], bool] = field(default_factory=dict)
    lean_version: Dict[Tuple[URL, COMMIT], str] = field(default_factory=dict)


info_cache = RepoInfoCache()


_GIT_REQUIREMENT_REGEX = re.compile(
    r"require\s+(?P<name>\S+)\s+from\s+git\s+\"(?P<url>.+?)\"(\s+@\s+\"(?P<rev>\S+)\")?"
)


def is_new_version(v) -> bool:
    """Check if ``v`` is at least `v4.3.0-rc2`."""
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3):
        return False
    if (
        major > 4
        or (major == 4 and minor > 3)
        or (major == 4 and minor == 3 and patch > 0)
    ):
        return True
    assert major == 4 and minor == 3 and patch == 0
    if "4.3.0-rc" in v:
        rc = int(v.split("-")[1][2:])
    return rc >= 2


@dataclass(frozen=True)
class LeanGitRepo:
    """Git repo of a Lean project."""

    url: str
    """The repo's Github URL.

    Note that we only support Github as of now.
    """

    commit: str
    """The repo's commit hash.
    
    You can also use tags such as ``v3.5.0``. They will be converted to commit hashes.
    """

    repo: Repository = field(init=False, repr=False)
    """A :class:`github.Repository` object.
    """

    uses_lean3: bool = field(init=False, repr=False)
    """Whether the repo uses Lean 3.
    """

    uses_lean4: bool = field(init=False, repr=False)
    """Whether the repo uses Lean 4.
    """

    lean_version: str = field(init=False, repr=False)
    """Required Lean version.
    """

    def __post_init__(self) -> None:
        if "github.com" not in self.url:
            raise ValueError(f"{self.url} is not a Github URL")
        if not self.url.startswith("https://"):
            raise ValueError(f"{self.url} is not a valid URL")
        object.__setattr__(self, "url", normalize_url(self.url))
        object.__setattr__(self, "repo", url_to_repo(self.url))

        # Convert tags or branches to commit hashes
        if not (len(self.commit) == 40 and _COMMIT_REGEX.fullmatch(self.commit)):
            if (self.url, self.commit) in info_cache.tag2commit:
                commit = info_cache.tag2commit[(self.url, self.commit)]
            else:
                logger.debug(f"Querying the commit hash for {self.name} {self.commit}")
                commit = _to_commit_hash(self.repo, self.commit)
                assert _COMMIT_REGEX.fullmatch(commit), f"Invalid commit hash: {commit}"
                info_cache.tag2commit[(self.url, self.commit)] = commit
            object.__setattr__(self, "commit", commit)

        # Determine whether the repo uses Lean 3 or Lean 4
        if (self.url, self.commit) in info_cache.uses_lean3:
            uses_lean3 = info_cache.uses_lean3[(self.url, self.commit)]
        else:
            uses_lean3 = self.is_lean3 or url_exists(
                self._get_config_url("leanpkg.toml")
            )
        info_cache.uses_lean3[(self.url, self.commit)] = uses_lean3
        object.__setattr__(self, "uses_lean3", uses_lean3)

        if (self.url, self.commit) in info_cache.uses_lean4:
            uses_lean4 = info_cache.uses_lean4[(self.url, self.commit)]
        else:
            uses_lean4 = self.is_lean4 or url_exists(
                self._get_config_url("lean-toolchain")
            )
        info_cache.uses_lean4[(self.url, self.commit)] = uses_lean4
        object.__setattr__(self, "uses_lean4", uses_lean4)

        assert uses_lean3 ^ uses_lean4

        # Determine the required Lean version, e.g., ``v3.50.3``.
        if (self.url, self.commit) in info_cache.lean_version:
            lean_version = info_cache.lean_version[(self.url, self.commit)]
        elif self.is_lean:
            lean_version = self.commit
        elif uses_lean3:
            config = self.get_config("leanpkg.toml")
            lean_version = get_lean3_version_from_config(config)
        else:
            config = self.get_config("lean-toolchain")
            lean_version = get_lean4_commit_from_config(config)
        info_cache.lean_version[(self.url, self.commit)] = lean_version
        object.__setattr__(self, "lean_version", lean_version)

    @classmethod
    def from_path(cls, path: Path) -> "LeanGitRepo":
        """Construct a :class:`LeanGitRepo` object from the path to a local Git repo."""
        url, commit = get_repo_info(path)
        return cls(url, commit)

    @property
    def name(self) -> str:
        return self.repo.name

    @property
    def is_lean4(self) -> bool:
        return self.url == LEAN4_URL

    @property
    def is_lean3(self) -> bool:
        return self.url == LEAN3_URL

    @property
    def is_lean(self) -> bool:
        """Check if the repo is Lean itself."""
        return self.is_lean3 or self.is_lean4

    @property
    def commit_url(self) -> str:
        return os.path.join(self.url, f"tree/{self.commit}")

    def show(self) -> None:
        """Show the repo in the default browser."""
        webbrowser.open(self.commit_url)

    def exists(self) -> bool:
        return url_exists(self.commit_url)

    def clone_and_checkout(self) -> None:
        """Clone the repo to the current working directory and checkout a specific commit."""
        logger.debug(f"Cloning {self}")
        execute(f"git clone -n --recursive {self.url}", capture_output=True)
        with working_directory(self.name):
            execute(
                f"git checkout {self.commit} && git submodule update --recursive",
                capture_output=True,
            )

    def get_packages_dir(self) -> Path:
        """Return the path to the directory where Lean packages are stored."""
        if self.uses_lean3:
            return LEAN3_PACKAGES_DIR
        else:
            toolchain = self.get_config("lean-toolchain")
            v = get_lean4_version_from_config(toolchain["content"])
            if is_new_version(v):
                return LEAN4_PACKAGES_DIR
            else:
                return LEAN4_PACKAGES_DIR_OLD

    def get_build_dir(self) -> Path:
        if self.uses_lean3:
            return LEAN_BUILD_DIR_OLD
        else:
            toolchain = self.get_config("lean-toolchain")
            v = get_lean4_version_from_config(toolchain["content"])
            if is_new_version(v):
                return LEAN4_BUILD_DIR
            else:
                return LEAN_BUILD_DIR_OLD

    def get_dependencies(
        self, path: Union[str, Path, None] = None
    ) -> Dict[str, "LeanGitRepo"]:
        """Return the dependencies required by the target repo.

        Args:
            path (Union[str, Path, None], optional): Root directory of the repo if it is on the disk.

        Returns:
            Dict[str, :class:`LeanGitRepo`]: A dictionary mapping the name of each
            dependency to its :class:`LeanGitRepo` object.
        """
        if self.is_lean:
            return {}
        elif self.uses_lean3:
            return self._get_lean3_dependencies(path)
        else:
            return self._get_lean4_dependencies(path)

    def _get_lean3_dependencies(
        self, path: Union[str, Path, None] = None, parents: List[str] = []
    ) -> Dict[str, "LeanGitRepo"]:
        logger.debug(f"Querying the dependencies of {self}")
        if path is None:
            config = self.get_config("leanpkg.toml")
        else:
            config = toml.load(Path(path) / "leanpkg.toml")

        deps = {"lean": LeanGitRepo(LEAN3_URL, get_lean3_version_from_config(config))}

        if "dependencies" in config:
            for _, v in config["dependencies"].items():
                r = LeanGitRepo(url=v["git"], commit=v["rev"])
                assert r not in parents, f"Circular dependency: {r}"
                deps[r.name] = r
                for dd in r._get_lean3_dependencies(None, [r.name] + parents).values():
                    deps[dd.name] = dd

        return deps

    def _parse_lakefile_dependencies(
        self, lakefile: str
    ) -> List[Tuple[str, "LeanGitRepo"]]:
        _LOCAL_REQUIREMENT_REGEX = r"require \S+ from \""
        if re.search(_LOCAL_REQUIREMENT_REGEX, lakefile):
            raise ValueError("Local dependencies are not supported.")

        deps = []

        for m in _GIT_REQUIREMENT_REGEX.finditer(lakefile):
            url = m["url"]
            if url.endswith(".git"):
                url = url[:-4]
            if url.startswith("git@"):
                url = "https://" + url[4:].replace(":", "/")

            rev = m["rev"]
            if rev is None:
                commit = get_latest_commit(url)
            elif len(rev) == 40 and _COMMIT_REGEX.fullmatch(rev):
                commit = rev
            else:
                logger.debug(f"Querying the commit hash for {url} {rev}")
                commit = _to_commit_hash(url_to_repo(url), rev)
                assert _COMMIT_REGEX.fullmatch(commit)

            deps.append((m["name"], LeanGitRepo(url, commit)))

        return deps

    def _get_lean4_dependencies(
        self, path: Union[str, Path, None] = None, parents: List[str] = []
    ) -> Dict[str, "LeanGitRepo"]:
        logger.debug(f"Querying the dependencies of {self}")
        if path is None:
            lakefile = self.get_config("lakefile.lean")
            toolchain = self.get_config("lean-toolchain")
        else:
            lakefile = {"content": (Path(path) / "lakefile.lean").open().read()}
            toolchain = {"content": (Path(path) / "lean-toolchain").open().read()}

        commit = get_lean4_commit_from_config(toolchain)
        deps = {"lean4": LeanGitRepo(LEAN4_URL, commit)}

        for name, repo in self._parse_lakefile_dependencies(lakefile["content"]):
            if name not in deps:
                deps[name] = repo
            for dd_name, dd_repo in repo._get_lean4_dependencies(
                None, [name] + parents
            ).items():
                deps[dd_name] = dd_repo

        return deps

    def get_license(self) -> Optional[str]:
        """Return the content of the ``LICENSE`` file."""
        assert "github.com" in self.url, f"Unsupported URL: {self.url}"
        url = self.url.replace("github.com", "raw.githubusercontent.com")
        license_url = f"{url}/{self.commit}/LICENSE"
        try:
            return read_url(license_url)
        except error.HTTPError:
            return None

    def _get_config_url(self, filename: str) -> str:
        assert "github.com" in self.url, f"Unsupported URL: {self.url}"
        url = self.url.replace("github.com", "raw.githubusercontent.com")
        return f"{url}/{self.commit}/{filename}"

    def get_config(self, filename: str) -> Dict[str, Any]:
        """Return the repo's ``leanpkg.toml`` config."""
        config_url = self._get_config_url(filename)
        content = read_url(config_url)
        if filename == "leanpkg.toml":
            return toml.loads(content)
        elif filename in ("lean-toolchain", "lakefile.lean"):
            return {"content": content}
        else:
            raise ValueError(f"Unsupported config file: {filename}")


@dataclass(frozen=True)
class Theorem:
    """Theorem in Lean.

    Theorems are named constants of type :code:`Prop`. They are typically defined
    using the keywords :code:`theorem` or :code:`lemma`, but it's possible to use other
    keywords such as :code:`def` or :code:`instance`
    """

    repo: LeanGitRepo
    """Lean repo the theorem comes from.
    """

    file_path: Path
    """Lean source file the theorem comes from.
    """

    full_name: str
    """Fully qualified name of the theorem.
    """

    def __post_init__(self) -> None:
        if isinstance(self.file_path, str):
            object.__setattr__(self, "file_path", Path(self.file_path))
        assert (
            self.file_path.suffix == ".lean"
        ), f"File extension must be .lean: {self.file_path}"

"""This module define classes for repos, files, and theorems in Lean.
Objects of these classes contain only surface information, without extracting any trace.
"""

import re
import os
import json
import toml
import time
import urllib
import shutil
import tempfile
import webbrowser
from enum import Enum
from pathlib import Path
from loguru import logger
from functools import cache
from git import Repo, BadName
from github import Github, Auth
from dataclasses import dataclass, field
from github.Repository import Repository
from github.GithubException import GithubException
from typing import List, Dict, Any, Generator, Union, Optional, Tuple, Iterator

from ..utils import (
    read_url,
    url_exists,
    working_directory,
    is_git_repo,
)
from .cache import cache as repo_cache
from ..constants import TMP_DIR, LEAN4_URL


GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN", None)
"""GiHub personal access token is optional. 
If provided, it can increase the rate limit for GitHub API calls.
"""

if GITHUB_ACCESS_TOKEN:
    logger.debug("Using GitHub personal access token for authentication")
    GITHUB = Github(auth=Auth.Token(GITHUB_ACCESS_TOKEN))
    GITHUB.get_user().login
else:
    logger.debug(
        "Using GitHub without authentication. Don't be surprised if you hit the API rate limit."
    )
    GITHUB = Github()

LEAN4_REPO = None
"""The GitHub Repo for Lean 4 itself."""

LEAN4_NIGHTLY_REPO = None
"""The GitHub Repo for Lean 4 nightly releases."""

_URL_REGEX = re.compile(r"(?P<url>.*?)/*")

_SSH_TO_HTTPS_REGEX = re.compile(r"git@github\.com:(?P<user>.+)/(?P<repo>.+?)(\.git)?")

REPO_CACHE_PREFIX = "repos"


class RepoType(Enum):
    GITHUB = 0
    REMOTE = 1  # Remote but not GitHub.
    LOCAL = 2


def normalize_url(url: str, repo_type: RepoType = RepoType.GITHUB) -> str:
    if repo_type == RepoType.LOCAL:  # Convert to absolute path if local.
        return os.path.abspath(url)
    # Remove trailing `/`.
    url = _URL_REGEX.fullmatch(url)["url"]  # type: ignore
    return ssh_to_https(url)


def ssh_to_https(url: str) -> str:
    m = _SSH_TO_HTTPS_REGEX.fullmatch(url)
    return f"https://github.com/{m.group('user')}/{m.group('repo')}" if m else url


def get_repo_type(url: str) -> Optional[RepoType]:
    """Get the type of the repository.

    Args:
        url (str): The URL of the repository.
    Returns:
        Optional[str]: The type of the repository (None if the repo cannot be found).
    """
    url = ssh_to_https(url)
    parsed_url = urllib.parse.urlparse(url)  # type: ignore
    if parsed_url.scheme in ["http", "https"]:
        # Case 1 - GitHub URL.
        if "github.com" in url:
            if not url.startswith("https://"):
                logger.warning(f"{url} should start with https://")
                return None
            else:
                return RepoType.GITHUB
        # Case 2 - remote URL.
        elif url_exists(url):  # Not check whether it is a git URL
            return RepoType.REMOTE
    # Case 3 - local path
    elif is_git_repo(Path(parsed_url.path)):
        return RepoType.LOCAL
    logger.warning(f"{url} is not a valid URL")
    return None


def _split_git_url(url: str) -> Tuple[str, str]:
    """Split a Git URL into user name and repo name."""
    if url.endswith("/"):
        url = url[:-1]
        assert not url.endswith("/"), f"Unexpected URL: {url}"
    fields = url.split("/")
    user_name = fields[-2]
    repo_name = fields[-1]
    return user_name, repo_name


def _format_cache_dirname(url: str, commit: str) -> str:
    user_name, repo_name = _split_git_url(url)
    repo_type = get_repo_type(url)
    assert repo_type is not None, f"Invalid url {url}"
    if repo_type == RepoType.GITHUB:
        return f"{user_name}-{repo_name}-{commit}"
    else:  # git repo
        return f"gitpython-{repo_name}-{commit}"


@cache
def url_to_repo(
    url: str,
    num_retries: int = 2,
    repo_type: Optional[RepoType] = None,
    tmp_dir: Optional[Path] = None,
) -> Union[Repo, Repository]:
    """Convert a URL to a Repo object.

    Args:
        url (str): The URL of the repository.
        num_retries (int): Number of retries in case of failure.
        repo_type (Optional[RepoType]): The type of the repository. Defaults to None.
        tmp_dir (Optional[Path]): The temporary directory to clone the repo to. Defaults to None.

    Returns:
        Repo: A Git Repo object.
    """
    url = normalize_url(url)
    backoff = 1
    if tmp_dir is None:
        tmp_dir = (TMP_DIR or Path("/tmp")) / next(tempfile._get_candidate_names())  # type: ignore
    repo_type = repo_type or get_repo_type(url)
    assert repo_type is not None, f"Invalid url {url}"
    while True:
        try:
            if repo_type == RepoType.GITHUB:
                return GITHUB.get_repo("/".join(url.split("/")[-2:]))
            with working_directory(tmp_dir):
                repo_name = os.path.basename(url)
                if repo_type == RepoType.LOCAL:
                    assert is_git_repo(url), f"Local path {url} is not a git repo"
                    shutil.copytree(url, repo_name)
                    return Repo(repo_name)
                else:
                    return Repo.clone_from(url, repo_name)
        except Exception as ex:
            if num_retries <= 0:
                raise ex
            num_retries -= 1
            logger.debug(f'url_to_repo("{url}") failed. Retrying...')
            time.sleep(backoff)
            backoff *= 2


@cache
def get_latest_commit(url: str) -> str:
    """Get the hash of the latest commit of the Git repo at ``url``."""
    repo = url_to_repo(url)
    if isinstance(repo, Repository):
        return repo.get_branch(repo.default_branch).commit.sha
    else:
        return repo.head.commit.hexsha


def cleanse_string(s: Union[str, Path]) -> str:
    """Replace : and / with _ in a string."""
    return str(s).replace("/", "_").replace(":", "_")


@cache
def _to_commit_hash(repo: Union[Repository, Repo], label: str) -> str:
    """Convert a tag or branch to a commit hash."""
    if isinstance(repo, Repository):  # GitHub repository
        logger.debug(f"Querying the commit hash for {repo.name} {label}")
        try:
            return repo.get_commit(label).sha
        except GithubException as ex:
            raise ValueError(f"Invalid tag or branch: `{label}` for {repo.name}")
    else:  # Local or remote Git repository
        assert isinstance(repo, Repo)
        logger.debug(
            f"Querying the commit hash for {repo.working_dir} repository {label}"
        )
        try:
            # Resolve the label to a commit hash
            return repo.commit(label).hexsha
        except Exception as ex:
            raise ValueError(f"Error converting ref to commit hash: {ex}")


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

    @property
    def start_pos(self) -> Pos:
        """Return the start position of a source file.

        Returns:
            Pos: A :class:`Pos` object representing the start of this file.
        """
        return Pos(1, 1)

    @property
    def end_pos(self) -> Pos:
        """Return the end position of a source file.

        Returns:
            Pos: A :class:`Pos` object representing the end of this file.
        """
        # Line and column numbers are 1-indexed by default.
        if self.is_empty():
            return self.start_pos
        line_nb = self.num_lines
        column_nb = 1 + len(self.code[-1])
        return Pos(line_nb, column_nb)

    def is_empty(self) -> bool:
        return len(self.code) == 0

    def convert_pos(self, byte_idx: int) -> Pos:
        """Convert a byte index (:code:`String.Pos` in Lean 4) to a :class:`Pos` object."""
        n = 0
        for i, num_bytes in enumerate(self.num_bytes, start=1):
            n += num_bytes
            if n == byte_idx and i == self.num_lines:
                byte_idx -= 1
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
_LEAN4_VERSION_REGEX = re.compile(r"leanprover/lean4:(?P<version>.+?)")


def is_commit_hash(s: str):
    """Check if a string is a valid commit hash."""
    return len(s) == 40 and _COMMIT_REGEX.fullmatch(s)


def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    v = m["version"]
    if not v.startswith("v") and v[0].isnumeric():
        v = "v" + v
    return v


def get_lean4_commit_from_config(config_dict: Dict[str, Any]) -> str:
    """Return the required Lean commit given a ``lean-toolchain`` config."""
    global LEAN4_REPO
    if LEAN4_REPO is None:
        LEAN4_REPO = GITHUB.get_repo("leanprover/lean4")
    assert "content" in config_dict, "config_dict must have a 'content' field"
    version = get_lean4_version_from_config(config_dict["content"].strip())
    if version.startswith("nightly-"):
        global LEAN4_NIGHTLY_REPO
        if LEAN4_NIGHTLY_REPO is None:
            LEAN4_NIGHTLY_REPO = GITHUB.get_repo("leanprover/lean4-nightly")
        return _to_commit_hash(LEAN4_NIGHTLY_REPO, version)
    else:
        return _to_commit_hash(LEAN4_REPO, version)


URL = str
TAG = str
COMMIT = str


@dataclass(frozen=True)
class RepoInfoCache:
    """To minize the number of network requests, we cache and re-use the info
    of all repos, assuming it does not change during the execution of LeanDojo."""

    tag2commit: Dict[Tuple[URL, TAG], COMMIT] = field(default_factory=dict)
    lean_version: Dict[Tuple[URL, COMMIT], str] = field(default_factory=dict)


info_cache = RepoInfoCache()


_LAKEFILE_LEAN_GIT_REQUIREMENT_REGEX = re.compile(
    r"require\s+(?P<name>\S+)\s+from\s+git\s+\"(?P<url>.+?)\"(\s+@\s+\"(?P<rev>\S+)\")?"
)

_LAKEFILE_LEAN_LOCAL_REQUIREMENT_REGEX = re.compile(r"require \S+ from \"")

_LAKEFILE_TOML_REQUIREMENT_REGEX = re.compile(r"(?<=\[\[require\]\]).+(?=\n\n)")


def is_supported_version(v) -> bool:
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
    else:
        return True


@dataclass(frozen=True)
class LeanGitRepo:
    """Git repo of a Lean project."""

    url: str
    """The repo's URL.

    It can be a GitHub URL that starts with https:// or git@github.com, a local path, or any other valid Git URL.
    """

    commit: str
    """The repo's commit hash.
    
    You can also use tags such as ``v3.5.0``. They will be converted to commit hashes.
    """

    repo: Union[Repository, Repo] = field(init=False, repr=False)
    """A :class:`github.Repository` object for GitHub repos or
    a :class:`git.Repo` object for local or remote Git repos.
    """

    lean_version: str = field(init=False, repr=False)
    """Required Lean version.
    """

    repo_type: RepoType = field(init=False, repr=False)
    """Type of the repo. It can be ``GITHUB``, ``LOCAL`` or ``REMOTE``.
    """

    def __post_init__(self) -> None:
        repo_type = get_repo_type(self.url)
        if repo_type is None:
            raise ValueError(f"{self.url} is not a valid URL")
        object.__setattr__(self, "repo_type", repo_type)
        object.__setattr__(self, "url", normalize_url(self.url, repo_type=repo_type))
        # set repo and commit
        if repo_type == RepoType.GITHUB:
            repo = url_to_repo(self.url, repo_type=repo_type)
        else:
            # get repo from cache
            rel_cache_dir = lambda url, commit: Path(
                f"{REPO_CACHE_PREFIX}/{_format_cache_dirname(url, commit)}/{self.name}"
            )
            cache_repo_dir = repo_cache.get(rel_cache_dir(self.url, self.commit))
            if cache_repo_dir is None:
                with working_directory() as tmp_dir:
                    repo = url_to_repo(self.url, repo_type=repo_type, tmp_dir=tmp_dir)
                    commit = _to_commit_hash(repo, self.commit)
                    cache_repo_dir = repo_cache.store(
                        repo.working_dir, rel_cache_dir(self.url, commit)
                    )
            repo = Repo(cache_repo_dir)
        # Convert tags or branches to commit hashes
        if not is_commit_hash(self.commit):
            if (self.url, self.commit) in info_cache.tag2commit:
                commit = info_cache.tag2commit[(self.url, self.commit)]
            else:
                commit = _to_commit_hash(repo, self.commit)
                assert is_commit_hash(commit), f"Invalid commit hash: {commit}"
                info_cache.tag2commit[(self.url, commit)] = commit
            object.__setattr__(self, "commit", commit)
        object.__setattr__(self, "repo", repo)

        # Determine the required Lean version.
        if (self.url, self.commit) in info_cache.lean_version:
            lean_version = info_cache.lean_version[(self.url, self.commit)]
        if self.is_lean4:
            lean_version = "latest"  # lean4 itself
        else:
            config = self.get_config("lean-toolchain")
            lean_version = get_lean4_version_from_config(config["content"])
            if not is_supported_version(lean_version):
                logger.warning(
                    f"{self} relies on an unsupported Lean version: {lean_version}"
                )
        info_cache.lean_version[(self.url, self.commit)] = lean_version
        object.__setattr__(self, "lean_version", lean_version)

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "LeanGitRepo":
        """Construct a :class:`LeanGitRepo` object from the path to a local Git repo."""
        commit = Repo(path).head.commit.hexsha
        return cls(str(path), commit)

    @property
    def name(self) -> str:
        return os.path.basename(self.url)

    @property
    def is_lean4(self) -> bool:
        return self.url == LEAN4_URL

    @property
    def commit_url(self) -> str:
        return f"{self.url}/tree/{self.commit}"

    def get_cache_dirname(self) -> Path:
        """Return the formatted cache directory name"""
        assert is_commit_hash(self.commit), f"Invalid commit hash: {self.commit}"
        return Path(_format_cache_dirname(self.url, self.commit))

    def show(self) -> None:
        """Show the repo in the default browser."""
        webbrowser.open(self.commit_url)

    def exists(self) -> bool:
        if self.repo_type != RepoType.GITHUB:
            repo = self.repo  # git repo
            try:
                repo.commit(self.commit)
                return repo.head.commit.hexsha == self.commit
            except BadName:
                logger.warning(
                    f"Commit {self.commit} does not exist in this repository."
                )
                return False
        else:
            return url_exists(self.commit_url)

    def clone_and_checkout(self) -> None:
        """Clone the repo to the current working directory and checkout a specific commit."""
        logger.debug(f"Cloning {self}")
        repo = Repo.clone_from(self.url, Path(self.name), no_checkout=True)
        repo.git.checkout(self.commit)
        repo.submodule_update(init=True, recursive=True)

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
        logger.debug(f"Querying the dependencies of {self}")

        toolchain = (
            self.get_config("lean-toolchain")
            if path is None
            else {"content": (Path(path) / "lean-toolchain").open().read()}
        )
        commit = get_lean4_commit_from_config(toolchain)
        deps = {"lean4": LeanGitRepo(LEAN4_URL, commit)}

        try:
            lake_manifest = (
                self.get_config("lake-manifest.json", num_retries=0)
                if path is None
                else json.load((Path(path) / "lake-manifest.json").open())
            )
            for pkg in lake_manifest["packages"]:
                deps[pkg["name"]] = LeanGitRepo(pkg["url"], pkg["rev"])
        except Exception:
            for name, repo in self._parse_lakefile_dependencies(path):
                if name not in deps:
                    deps[name] = repo
                for dd_name, dd_repo in repo.get_dependencies().items():
                    deps[dd_name] = dd_repo

        return deps

    def _parse_lakefile_dependencies(
        self, path: Union[str, Path, None]
    ) -> List[Tuple[str, "LeanGitRepo"]]:
        if self.uses_lakefile_lean():
            return self._parse_lakefile_lean_dependencies(path)
        else:
            return self._parse_lakefile_toml_dependencies(path)

    def _parse_lakefile_lean_dependencies(
        self, path: Union[str, Path, None]
    ) -> List[Tuple[str, "LeanGitRepo"]]:
        lakefile = (
            self.get_config("lakefile.lean")["content"]
            if path is None
            else (Path(path) / "lakefile.lean").open().read()
        )

        if _LAKEFILE_LEAN_LOCAL_REQUIREMENT_REGEX.search(lakefile):
            raise ValueError("Local dependencies are not supported.")

        return self._parse_deps(_LAKEFILE_LEAN_GIT_REQUIREMENT_REGEX.finditer(lakefile))

    def _parse_deps(
        self, matches: Union[Iterator[re.Match[str]], Dict[str, str]]
    ) -> List[Tuple[str, "LeanGitRepo"]]:
        deps = []

        for m in matches:
            url = m["url"]  # type: ignore
            if url.endswith(".git"):
                url = url[:-4]
            if url.startswith("git@"):
                url = "https://" + url[4:].replace(":", "/")
            try:
                rev = m["rev"]  # type: ignore
            except KeyError:
                rev = None
            if rev is None:
                commit = get_latest_commit(url)
            elif len(rev) == 40 and _COMMIT_REGEX.fullmatch(rev):
                commit = rev
            else:
                try:
                    commit = _to_commit_hash(url_to_repo(url), rev)
                except ValueError:
                    commit = get_latest_commit(url)
                assert _COMMIT_REGEX.fullmatch(commit)

            deps.append((m["name"], LeanGitRepo(url, commit)))  # type: ignore

        return deps

    def _parse_lakefile_toml_dependencies(
        self, path: Union[str, Path, None]
    ) -> List[Tuple[str, "LeanGitRepo"]]:
        lakefile = (
            self.get_config("lakefile.toml")
            if path is None
            else (Path(path) / "lakefile.toml").open().read()
        )
        # Parsing worked
        if isinstance(lakefile, dict) and "require" in lakefile:
            matches = lakefile["require"]
        else:
            if "content" in lakefile:
                lakefile = lakefile["content"]
            matches = []
            for req in _LAKEFILE_TOML_REQUIREMENT_REGEX.finditer(lakefile):
                match = {}
                for line in req.group().strip().splitlines():
                    key, value = line.split("=")
                    match[key.strip()] = value.strip()
                matches.append(match)
        for match in matches:
            if "path" in match:
                raise ValueError("Local dependencies are not supported.")
            if "git" in match:
                match["url"] = match["git"]
                del match["git"]

        return self._parse_deps(matches)

    def get_license(self) -> Optional[str]:
        """Return the content of the ``LICENSE`` file."""
        if self.repo_type == RepoType.GITHUB:
            assert "github.com" in self.url, f"Unsupported URL: {self.url}"
            url = self.url.replace("github.com", "raw.githubusercontent.com")
            license_url = f"{url}/{self.commit}/LICENSE"
            try:
                return read_url(license_url)
            except urllib.error.HTTPError:  # type: ignore
                return None
        else:
            license_path = Path(self.repo.working_dir) / "LICENSE"
            if license_path.exists():
                return license_path.open("r").read()
            else:
                return None

    def _get_config_url(self, filename: str) -> str:
        assert self.repo_type == RepoType.GITHUB
        assert "github.com" in self.url, f"Unsupported URL: {self.url}"
        url = self.url.replace("github.com", "raw.githubusercontent.com")
        return f"{url}/{self.commit}/{filename}"

    def get_config(self, filename: str, num_retries: int = 2) -> Dict[str, Any]:
        """Return the repo's files."""
        if self.repo_type == RepoType.GITHUB:
            config_url = self._get_config_url(filename)
            content = read_url(config_url, num_retries)
        else:
            working_dir = self.repo.working_dir
            with open(os.path.join(working_dir, filename), "r") as f:
                content = f.read()
        if filename.endswith(".toml"):
            return toml.loads(content)
        elif filename.endswith(".json"):
            return json.loads(content)
        else:
            return {"content": content}

    def uses_lakefile_lean(self) -> bool:
        """Check if the repo uses a ``lakefile.lean``."""
        if self.repo_type == RepoType.GITHUB:
            url = self._get_config_url("lakefile.lean")
            return url_exists(url)
        else:
            lakefile_path = Path(self.repo.working_dir) / "lakefile.lean"
            return lakefile_path.exists()

    def uses_lakefile_toml(self) -> bool:
        """Check if the repo uses a ``lakefile.toml``."""
        if self.repo_type == RepoType.GITHUB:
            url = self._get_config_url("lakefile.toml")
            return url_exists(url)
        else:
            lakefile_path = Path(self.repo.working_dir) / "lakefile.toml"
            return lakefile_path.exists()


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

    @property
    def uid(self) -> str:
        """Unique identifier of the theorem."""
        return f"{cleanse_string(self.repo.url)}@{cleanse_string(self.repo.commit)}:{cleanse_string(self.file_path.__str__())}:{cleanse_string(self.full_name)}"

    @property
    def uhash(self) -> str:
        """Unique hash of the theorem."""
        return str(hash(self.uid) ** 2)

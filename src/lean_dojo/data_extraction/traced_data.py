"""This module defines traced repos/files/theorems.
"""
import re
import os
import ray
import json
import random
import itertools
import webbrowser
import networkx as nx
from tqdm import tqdm
from lxml import etree
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Generator, Union

from ..utils import (
    is_git_repo,
    compute_md5,
    ray_actor_pool,
    normalize_spaces,
    to_lean_path,
    to_dep_path,
    to_json_path,
    to_xml_path,
)
from .ast.lean3.node import *
from .ast.lean4.node import *
from ..constants import NUM_WORKERS
from ..constants import LOAD_USED_PACKAGES_ONLY
from .lean import LeanFile, LeanGitRepo, Theorem, Pos
from .ast.lean3.expr import Expr, ConstExpr, parse_exprs_forest


@dataclass(frozen=True)
class Comment:
    """A comment in a Lean file."""

    start: Pos
    end: Pos
    text: str

    def __post_init__(self) -> None:
        assert isinstance(self.start, Pos)
        assert isinstance(self.end, Pos)
        assert self.start <= self.end
        assert isinstance(self.text, str)

    @classmethod
    def from_lean3_data(cls, d: Dict[str, Any], lean_file: LeanFile) -> "Comment":
        start, end = parse_pos(d)
        prefix = lean_file[:start].strip()
        suffix = lean_file[end:].strip()
        if prefix.endswith("--"):
            start.column_nb -= 2
        else:
            assert prefix.endswith("/-") and suffix.startswith("-/")
            start.column_nb -= 2
            end.column_nb += 2
        return Comment(start, end, d["text"])

    def to_xml(self, parent: etree.Element) -> None:
        tree = etree.SubElement(parent, self.__class__.__name__)
        tree.set("start", str(self.start))
        tree.set("end", str(self.end))
        tree.set("text", self.text)

    @classmethod
    def from_xml(cls, tree: etree.Element) -> "Comment":
        start = Pos.from_str(tree.attrib["start"])
        end = Pos.from_str(tree.attrib["end"])
        text = tree.attrib["text"]
        return cls(start, end, text)


def _collect_lean4_comments(ast: FileNode4) -> List[Comment]:
    comments = []

    def _callback(node, _):
        nonlocal comments
        if isinstance(node, CommandModuledocNode4) or isinstance(
            node, CommandDoccommentNode4
        ):
            comments.append(Comment(node.start, node.end, node.comment))
        elif is_leaf(node) and node.trailing.strip().startswith("--"):
            num_spaces = node.trailing.index("--")
            text = node.trailing[num_spaces:]
            start = node.lean_file.offset(node.end, num_spaces)
            end = node.lean_file.offset(start, len(text))
            comments.append(Comment(start, end, text))

    ast.traverse_preorder(_callback, node_cls=None)
    return comments


_SINGLE_LINE_COMMENT_REGEX = r"--.*?(\n|$)"
_MULTI_LINE_COMMENT_REGEX = r"/-.*?(-/|$)"
_COMMENT_REGEX = re.compile(
    f"{_SINGLE_LINE_COMMENT_REGEX}|{_MULTI_LINE_COMMENT_REGEX}", re.DOTALL
)


def get_code_without_comments(
    lean_file: LeanFile, start: Pos, end: Pos, comments: List[Comment]
) -> str:
    """Return the code in ``lean_file`` from ``start`` to ``end`` with comments removed.

    Args:
        lean_file (LeanFile): The lean source file.
        start (Pos): The start position.
        end (Pos): The end position.
        comments (List[Comment]): A list of :class:`Comment` objects.

    Returns:
        str: Human-written code with comments removed.
    """
    base = start
    code_segs = []

    for c in comments:
        if base <= c.start and c.end <= end:
            code_segs.append(lean_file[base : c.start])
            base = c.end

    code_segs.append(lean_file[base:end])
    code = "".join(code_segs)

    code = _COMMENT_REGEX.sub("", code)
    assert "--" not in code and "/-" not in code

    return code.strip()


@dataclass(frozen=True)
class TracedTactic:
    """A traced tactic is a tactic annotated with additional information including
    its AST and the states before/after the tactic.
    """

    ast: Union[TacticNode, Node4] = field(repr=False)
    """AST of the tactic.
    """

    traced_theorem: Optional["TracedTheorem"] = field(
        default=None, repr=False, compare=False
    )
    """The traced theorem this tactic belongs to.
    """

    def __getstate__(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "traced_theorem"}
        d["traced_theorem"] = None  # Avoid serializing the traced theorem.
        return d

    @property
    def tactic(self) -> str:
        """The raw tactic string."""
        return self.ast.tactic

    @property
    def state_before(self) -> str:
        """Pretty-printed state before applying the tactic."""
        assert self.ast.state_before is not None
        return self.ast.state_before

    @property
    def state_after(self) -> str:
        """Pretty-printed state after applying the tactic."""
        assert self.ast.state_after is not None
        return self.ast.state_after

    @property
    def start(self) -> Pos:
        """Start position in :file:`*.lean` file."""
        return self.ast.start

    @property
    def end(self) -> Pos:
        """End position in :file:`*.lean` file."""
        return self.ast.end

    @property
    def uses_lean3(self) -> bool:
        """Whether the tactic is in Lean 3."""
        return isinstance(self.ast, TacticNode)

    @property
    def uses_lean4(self) -> bool:
        """Wehther the tactic is in Lean 4."""
        return not self.uses_lean3

    def to_string(self) -> str:
        return f"{self.__class__.__name__}(tactic={self.tactic}, state_before={self.state_before}, state_after={self.state_after})"

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def get_annotated_tactic(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Return the tactic annotated with premise information.

        Premises in the tactic are marked by ``<a> ... </a>``. For example,
        :code:`rw [add_comm b]` contains a premise :code:`add_comm` and therefore
        becomes :code:`rw [<a>add_comm</a> b]`. In addition, the function returns
        the provenance (full name, file path, line/column numbers) of all premises.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: The first return value is the tactic string marked by ``<a> ... </a>``. The second return value is a list of provenances.
        """
        assert self.traced_theorem != None
        lean_file = self.traced_theorem.traced_file.lean_file
        annot_tac = []
        provenances = []
        cur = self.start

        if self.uses_lean3:

            def _callback3(node: IdentNode, _) -> None:
                nonlocal cur

                if node.full_name is not None:
                    if node.def_path is None or node.def_pos is None:
                        logger.warning(f"Unable to locate {node.full_name}")
                    else:
                        annot_tac.append(lean_file[cur : node.start])
                        annot_tac.append(
                            "<a>" + lean_file[node.start : node.end] + "</a>"
                        )
                        prov = {"full_name": node.full_name}
                        prov["def_path"] = node.def_path
                        prov["def_pos"] = list(node.def_pos)
                        provenances.append(prov)
                        cur = node.end

            self.ast.traverse_preorder(_callback3, IdentNode)
            annot_tac.append(lean_file[cur : self.end])

            return "".join(annot_tac), provenances
        else:
            assert self.uses_lean4

            def _callback4(node: IdentNode4, _):
                nonlocal cur

                if (
                    node.full_name is not None
                    and node.mod_name is not None
                    and node.def_start is not None
                    and node.def_end is not None
                ):
                    if cur <= node.start:
                        annot_tac.append(lean_file[cur : node.start])
                        annot_tac.append(
                            "<a>" + lean_file[node.start : node.end] + "</a>"
                        )
                        prov = {"full_name": node.full_name}
                        prov["def_path"] = node.def_path
                        prov["def_pos"] = list(node.def_start)
                        prov["def_end_pos"] = list(node.def_end)
                        provenances.append(prov)
                        cur = node.end

            self.ast.traverse_preorder(_callback4, IdentNode4)
            annot_tac.append(lean_file[cur : self.end])

            return "".join(annot_tac), provenances


@dataclass(frozen=True)
class TracedTheorem:
    """A traced theorem is a theorem with additional information such as the AST."""

    root_dir: Path = field(repr=False)
    """Root directory of the corresponding traced repo.
    """

    theorem: Theorem
    """The corresponding :class:`Theorem` object.
    """

    ast: Union[TheoremNode, CommandTheoremNode4] = field(repr=False, compare=False)
    """AST of the theorem.
    """

    comments: List[Comment] = field(repr=False, compare=False)
    """All comments in the theorem/proof.
    """

    traced_file: Optional["TracedFile"] = field(default=None, repr=False, compare=False)
    """The traced file this theorem belongs to.
    """

    def __post_init__(self) -> None:
        assert (
            self.root_dir.is_absolute() and self.root_dir == self.traced_file.root_dir
        )

    def __getstate__(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "traced_file"}
        d["traced_file"] = None
        return d

    @property
    def start(self) -> Pos:
        """Start position in :file:`*.lean` file."""
        return self.ast.start

    @property
    def end(self) -> Pos:
        """End position in :file:`*.lean` file."""
        return self.ast.end

    @property
    def repo(self) -> LeanGitRepo:
        """The Lean repo this theorem belongs to."""
        return self.theorem.repo

    @property
    def uses_lean3(self) -> bool:
        """Whether the theorem is in Lean 3."""
        return isinstance(self.ast, TheoremNode)

    @property
    def uses_lean4(self) -> bool:
        """Whether the theorem is in Lean 4."""
        return not self.uses_lean3

    @property
    def file_path(self) -> Path:
        """The theorem's file path (relative to the root directory)."""
        return self.theorem.file_path

    @property
    def traced_repo(self) -> "TracedRepo":
        """The traced repo this theorem belongs to."""
        if self.traced_file is None:
            return None
        else:
            return self.traced_file.traced_repo

    @property
    def is_private(self) -> bool:
        """Check if the theorem is private."""
        return self.ast.is_private()

    def show(self) -> None:
        """Show the theorem in the default browser."""
        url = os.path.join(
            self.repo.url,
            "blob",
            self.repo.commit,
            self.file_path,
            f"#L{self.start.line_nb}-L{self.end.line_nb}",
        )
        webbrowser.open(url)

    def has_tactic_proof(self) -> bool:
        """Check if the theorem has a tactic-style proof."""
        return self.ast.has_tactic_proof()

    def get_proof_node(self) -> Node:
        """Return the AST of the theorem's proof."""
        return self.ast.get_proof_node()

    def locate_proof(self) -> Tuple[Pos, Pos]:
        """Return the start/end positions of the proof."""
        start, end = self.get_proof_node().get_closure()
        if end < self.end:
            end = self.end
        return start, end

    def get_tactic_proof(self) -> Optional[str]:
        """Return the tactic-style proof (if any)."""
        if not self.has_tactic_proof():
            return None
        node = self.get_proof_node()
        start, end = node.get_closure()
        proof = get_code_without_comments(node.lean_file, start, end, self.comments)
        if not re.match(r"^(by|begin)\s", proof):
            return None
        else:
            return proof

    def get_theorem_statement(self) -> str:
        """Return the theorem statement."""
        proof_start, _ = self.locate_proof()
        return get_code_without_comments(
            self.traced_file.lean_file, self.ast.start, proof_start, self.comments
        )

    def get_single_tactic_proof(self) -> Optional[str]:
        """Wrap the proof into a single (potentially very long) tactic."""
        if not self.has_tactic_proof():
            return None
        node = self.get_proof_node()
        start, end = node.get_closure()
        proof = get_code_without_comments(node.lean_file, start, end, self.comments)

        if self.uses_lean4:
            # logger.warning(proof)
            raise NotImplementedError
            assert (
                isinstance(node.children[0], AtomNode4) and node.children[0].val == "by"
            )
            assert proof.startswith("by")
            proof = proof[len("by") :].strip()

        else:
            proof = normalize_spaces(proof)
            assert len(node.children) == 1
            child = node.children[0]

            if isinstance(child, BeginNode):
                if not (proof.startswith("begin ") and proof.endswith(" end")):
                    logger.warning(f"Found an ill-formed proof {proof} for {self}.")
                    return None
                proof = proof[len("begin") : -len("end")].strip()
            else:
                assert isinstance(child, ByNode)
                assert proof.startswith("by")
                proof = proof[len("by") :].strip()

            if not (proof.startswith("{") and proof.endswith("}")):
                proof = "{ " + proof + " }"

        return proof

    def get_namespaces(self) -> Tuple[List[str], List[str]]:
        """Return the namespaces that the theorem is located in,
        as well as the namespaces that are merely open.
        """
        assert self.traced_file is not None
        return self.traced_file.get_namespaces(self)

    def get_premise_full_names(self) -> List[str]:
        """Return the fully qualified names of all premises used in the proof."""
        names = []

        def _callback(node: Union[IdentNode, IdentNode4], parents: List[Node]):
            if node.full_name is not None:
                names.append(node.full_name)

        node_cls = IdentNode if self.uses_lean3 else IdentNode4
        self.ast.traverse_preorder(_callback, node_cls)

        return names

    def get_traced_tactics(self) -> List[TracedTactic]:
        """Return a list of traced tactics in the proof."""
        if self.uses_lean3:
            tacs = self._get_traced_tactics_lean3()
        else:
            tacs = self._get_traced_tactics_lean4()

        # Deduplicate.
        signatures = set()
        tacs_dedup = []
        for t in tacs:
            sig = (t.state_before, t.tactic, t.state_after)
            if sig not in signatures:
                signatures.add(sig)
                tacs_dedup.append(t)
        return tacs_dedup

    def _get_traced_tactics_lean3(self) -> List[TracedTactic]:
        tacs = []

        def _callback(node: TacticNode, _):
            if node.state_before is not None:
                # Tactics outside theorem/lemma definitions are not recorded.
                tacs.append(TracedTactic(node, self))

        self.ast.traverse_preorder(_callback, TacticNode)
        return tacs

    def _get_traced_tactics_lean4(self) -> List[TracedTactic]:
        tacs = []

        def _callback(node, _):
            if type(node) in (
                TacticTacticseq1IndentedNode4,
                TacticTacticseqbracketedNode4,
            ):
                for tac_node in node.get_tactic_nodes():
                    if (
                        hasattr(tac_node, "state_before")
                        and tac_node.state_before is not None
                    ):
                        # Tactics outside theorem/lemma definitions are not recorded.
                        tacs.append(TracedTactic(tac_node, self))

        self.ast.traverse_preorder(_callback, node_cls=None)
        return tacs

    def get_num_tactics(self) -> int:
        """Return the number of tactics in the proof."""
        return len(self.get_traced_tactics())


_TAG_INDEX_REGEX = re.compile(r"(?P<key>\S+)\[(?P<idx>\d+)\]$")


def _qualify_name(name: str, prefix: str) -> str:
    """Qualify a name with a prefix."""
    if name.startswith("_root_."):
        return name[len("_root_.") :]
    elif prefix == "":
        return name
    else:
        return f"{prefix}.{name}"


def _in_tactic(_: IdentNode, parents: List[Node]) -> bool:
    """Check if the current node is in a tactic."""
    return any(isinstance(p, TacticNode) for p in parents)


def _fix_indentation(tac: str, indent: int) -> str:
    """Fix the indentation of a tactic."""
    lines = tac.splitlines()
    if len(lines) == 1:
        return tac
    else:
        lines_new = [lines[0]]
        for l in lines[1:]:
            for i in range(len(l)):
                if l[i] != " " or i >= indent:
                    lines_new.append(l[i:])
                    break

        return "\n".join(lines_new)


@dataclass(eq=False)
class TracedFile:
    """A traced file is a Lean source file annotated with syntactic/semantic information
    such as tactic states, Lean expressions, and abstract syntax trees (ASTs).
    """

    root_dir: Path
    """Root directory (in absolute path) of the corresponding traced repo.
    """

    repo: LeanGitRepo
    """The Lean repo this traced file belongs to.
    """

    lean_file: LeanFile
    """Lean source file of this traced file.
    """

    ast: Union[FileNode, FileNode4] = field(repr=False)
    """Abstract syntax tree (AST) of the entire :code:`*.lean` file.
    
    AST nodes are defined in :ref:`lean_dojo.data_extraction.ast.lean3.node` and :ref:`lean_dojo.data_extraction.ast.lean4.node`. 
    Some AST nodes are annotated with semantic information. E.g., :class:`lean_dojo.data_extraction.ast.lean3.node.TacticNode` 
    includes the states before/after the tactic.
    """

    exprs: List[Expr] = field(repr=False)
    """All Lean expressions in the :code:`*.lean` file.
    """

    comments: List[Comment] = field(repr=False)
    """All comments in the :code:`*.lean` file.
    """

    traced_repo: Optional["TracedRepo"] = field(default=None, repr=False)
    """The traced repo this traced file belongs to.
    
    Note that ``traced_repo`` will become None after the traced file is serialized/deserialized on its own.
    """

    def __post_init__(self) -> None:
        assert self.root_dir.is_absolute(), f"{self.root_dir} is not an absolute path"

    def __getstate__(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "traced_repo"}
        d["traced_repo"] = None
        return d

    @property
    def path(self) -> Path:
        """Path of the :file:`*.lean` file relative to the root directory."""
        return self.lean_file.path

    @property
    def abs_path(self) -> Path:
        """Absolute path of the :code:`*.lean` file."""
        return self.root_dir / self.path

    @property
    def uses_lean4(self) -> bool:
        """Whether the file uses Lean 4."""
        return self.lean_file.uses_lean4

    @property
    def uses_lean3(self) -> bool:
        """Whether the file uses Lean 3."""
        return not self.lean_file.uses_lean4

    @property
    def has_prelude(self) -> bool:
        """Check whether the file starts with :code:``prelude``.

        :code:``prelude`` instructs Lean NOT to include its built-in library automatically.
        """
        result = False

        def _callback(node: Union[PreludeNode, ModulePreludeNode4], _: List[Node]):
            nonlocal result
            result = True
            return True  # Stop traversing.

        node_cls = PreludeNode if self.uses_lean3 else ModulePreludeNode4
        self.ast.traverse_preorder(_callback, node_cls)
        return result

    @classmethod
    def from_traced_file(
        cls, root_dir: Union[str, Path], json_path: Path, repo: LeanGitRepo
    ) -> "TracedFile":
        """Construct a :class:`TracedFile` object by parsing a :file:`*.ast.json` file
        produced by :code:`lean --ast --tsast --tspp` (Lean 3) or :file:`ExtractData.lean` (Lean 4).

        Args:
            root_dir (Union[str, Path]): Root directory of the traced repo.
            json_path (Path): Path of the :file:`*.ast.json` file relative to ``root_dir``.
        """
        root_dir = Path(root_dir)
        root_dir = root_dir.resolve()
        if not json_path.is_absolute():
            json_path = root_dir / json_path
        if not json_path.exists():
            raise FileNotFoundError(f"{json_path} does not exist")
        assert json_path.suffixes == [
            ".ast",
            ".json",
        ], f"{json_path} is not a *.ast.json file"

        if repo.uses_lean3:
            return cls._from_lean3_traced_file(root_dir, json_path, repo)
        else:
            assert repo.uses_lean4, f"repo {repo} uses neither Lean 3 nor Lean 4"
            return cls._from_lean4_traced_file(root_dir, json_path, repo)

    @classmethod
    def _from_lean3_traced_file(
        cls, root_dir: Path, json_path: Path, repo: LeanGitRepo
    ) -> "TracedFile":
        lean_path = json_path.with_suffix("").with_suffix(".lean").relative_to(root_dir)
        lean_file = LeanFile(root_dir, lean_path, uses_lean4=False)
        data = json.load(json_path.open())

        exprs, expr_tags_map = parse_exprs_forest(data)
        data["expr_tags_map"] = expr_tags_map

        assert data["ast"][0] is None
        assert "end" not in data["ast"][1]
        end_pos = lean_file.end_pos(zero_indexed=True)
        data["ast"][1]["end"] = [end_pos.line_nb, end_pos.column_nb]

        data["module_paths"] = []
        for line in json_path.with_suffix("").with_suffix(".dep_paths").open():
            line = line.strip()
            if line == "":
                break
            data["module_paths"].append(line)

        ast = FileNode.from_data(data, 1, lean_file)
        TracedFile._post_process_lean3(ast, exprs, lean_file)

        comments = [Comment.from_lean3_data(d, lean_file) for d in data["comments"]]

        return cls(root_dir, repo, lean_file, ast, exprs, comments)

    @classmethod
    def _from_lean4_traced_file(
        cls, root_dir: Path, json_path: Path, repo: LeanGitRepo
    ) -> "TracedFile":
        lean_path = to_lean_path(root_dir, json_path, repo)
        lean_file = LeanFile(root_dir, lean_path, uses_lean4=True)

        data = json.load(json_path.open())

        data["module_paths"] = []
        for line in (
            json_path.with_suffix("").with_suffix("").with_suffix(".dep_paths").open()
        ):
            line = line.strip()
            if line == "":
                break
            data["module_paths"].append(line)

        ast = FileNode4.from_data(data, lean_file)
        comments = _collect_lean4_comments(ast)
        TracedFile._post_process_lean4(
            ast,
            lean_file,
            data["tactics"],
            data["premises"],
            data["module_paths"],
            comments,
        )

        return cls(root_dir, repo, lean_file, ast, None, comments)

    @classmethod
    def _post_process_lean3(
        cls, ast: FileNode, exprs: List[Expr], lean_file: LeanFile
    ) -> None:
        assert all(i == int(e.tags) for i, e in enumerate(exprs))
        inside_sections_namespaces = []

        def _callback(node: Node, parents: List[Node]):
            if isinstance(node, NamespaceNode) or isinstance(node, SectionNode):
                inside_sections_namespaces.append(node)
            elif isinstance(node, EndNode):
                begin_node = inside_sections_namespaces.pop()
                assert begin_node.name == node.name, "Namespace or section mismatch"
            elif is_potential_premise_lean3(node):
                prefix = ".".join(
                    ns.name
                    for ns in inside_sections_namespaces
                    if isinstance(ns, NamespaceNode)
                )
                if is_mutual_lean3(node):
                    full_name = [_qualify_name(name, prefix) for name in node.name]
                    object.__setattr__(node, "full_name", full_name)
                elif node.name is not None:
                    full_name = _qualify_name(node.name, prefix)
                    object.__setattr__(node, "full_name", full_name)
            elif (
                isinstance(node, IdentNode)
                and _in_tactic(node, parents)
                and node.expr is not None
            ):
                fields = node.expr.split(".")
                e = exprs[int(fields[0])]
                for tag in fields[1:]:
                    m = _TAG_INDEX_REGEX.fullmatch(tag)
                    if m is not None:
                        e = e.__getattribute__(m["key"])[int(m["idx"])]
                    else:
                        e = e.__getattribute__(tag)

                if isinstance(e, ConstExpr):
                    if e.def_pos is None:
                        pass
                    elif e.def_path is None:
                        e.def_path = str(lean_file.path)
                    object.__setattr__(node, "full_name", e.full_name)
                    object.__setattr__(node, "def_path", e.def_path)
                    object.__setattr__(node, "def_pos", e.def_pos)

        ast.traverse_preorder(_callback, node_cls=None)

    @classmethod
    def _post_process_lean4(
        cls,
        ast: FileNode4,
        lean_file: LeanFile,
        tactics_data: List[Dict[str, Any]],
        premises_data: List[Dict[str, Any]],
        imports_data: List[str],
        comments: List[Comment],
    ) -> None:
        pos2tactics = {}
        for t in tactics_data:
            start = lean_file.convert_pos(t["pos"])
            end = lean_file.convert_pos(t["endPos"])
            pos2tactics[(start, end)] = t

        pos2premises = {}
        for p in premises_data:
            if (
                p is None
                or p["pos"] is None
                or p["endPos"] is None
                or p["fullName"] is None
                or p["fullName"] == "[anonymous]"
            ):
                continue
            start_line_nb, start_column_nb = p["pos"]["line"], p["pos"]["column"]
            end_line_nb, end_column_nb = p["endPos"]["line"], p["endPos"]["column"]
            start = Pos(line_nb=start_line_nb, column_nb=start_column_nb + 1)
            end = Pos(line_nb=end_line_nb, column_nb=end_column_nb + 1)
            pos2premises[(start, end)] = p

        inside_sections_namespaces = []

        def _callback(node: Node4, _):
            if (
                type(node)
                in (
                    CommandNamespaceNode4,
                    CommandSectionNode4,
                    CommandNoncomputablesectionNode4,
                )
                and node.name is not None
            ):
                inside_sections_namespaces.append(node)
            elif (
                isinstance(node, CommandEndNode4)
                and node.name is not None
                and len(inside_sections_namespaces) > 0
            ):
                inside_sections_namespaces.pop()
            elif is_potential_premise_lean4(node):
                prefix = ".".join(
                    ns.name
                    for ns in inside_sections_namespaces
                    if isinstance(ns, CommandNamespaceNode4)
                )
                full_name = (
                    [_qualify_name(name, prefix) for name in node.name]
                    if is_mutual_lean4(node)
                    else _qualify_name(node.name, prefix)
                )
                object.__setattr__(node, "full_name", full_name)
                if isinstance(node, CommandDeclarationNode4) and node.is_theorem:
                    object.__setattr__(node.get_theorem_node(), "full_name", full_name)
            elif type(node) in (
                TacticTacticseq1IndentedNode4,
                TacticTacticseqbracketedNode4,
            ):
                for tac_node in node.get_tactic_nodes():
                    assert type(tac_node) in (OtherNode4, TacticTacticseqbracketedNode4)
                    if (tac_node.start, tac_node.end) not in pos2tactics:
                        continue
                    t = pos2tactics[(tac_node.start, tac_node.end)]
                    tac = get_code_without_comments(
                        lean_file, tac_node.start, tac_node.end, comments
                    )
                    tac = _fix_indentation(tac, tac_node.start.column_nb - 1)
                    object.__setattr__(tac_node, "state_before", t["stateBefore"])
                    object.__setattr__(tac_node, "state_after", t["stateAfter"])
                    object.__setattr__(tac_node, "tactic", tac)
            elif isinstance(node, IdentNode4):
                start, end = node.get_closure()
                if (start, end) in pos2premises:
                    assert start is not None
                    assert end is not None
                    p = pos2premises[(start, end)]
                    prem = get_code_without_comments(lean_file, start, end, comments)
                    prem = _fix_indentation(prem, start.column_nb - 1)
                    if p["fullName"] is not None:
                        object.__setattr__(node, "full_name", p["fullName"])
                    if p["modName"] is not None:
                        object.__setattr__(node, "mod_name", p["modName"])
                    if p["defPath"] is not None:
                        object.__setattr__(node, "def_path", p["defPath"])
                    if p["defPos"] is not None and p["defEndPos"] is not None:
                        def_start_line_nb, def_start_column_nb = (
                            p["defPos"]["line"],
                            p["defPos"]["column"],
                        )
                        def_end_line_nb, def_end_column_nb = (
                            p["defEndPos"]["line"],
                            p["defEndPos"]["column"],
                        )
                        def_start = Pos(
                            line_nb=def_start_line_nb, column_nb=def_start_column_nb + 1
                        )
                        def_end = Pos(
                            line_nb=def_end_line_nb, column_nb=def_end_column_nb + 1
                        )
                        object.__setattr__(node, "def_start", def_start)
                        object.__setattr__(node, "def_end", def_end)
            elif type(node) in (ModuleImportNode4,):
                node_module_name = object.__getattribute__(node, "module")
                if node_module_name is not None:
                    suffix = node_module_name.replace(".", "/")
                    for import_line in imports_data:
                        if import_line.endswith(
                            suffix + ".lean"
                        ) or import_line.endswith(suffix + "/default.lean"):
                            object.__setattr__(node, "path", Path(import_line))

        ast.traverse_preorder(_callback, node_cls=None)

    def get_namespaces(self, thm: TracedTheorem) -> Tuple[List[str], List[str]]:
        """Return the namespaces that ``thm`` is located in,
        as well as the namespaces that are merely open.
        """
        if self.uses_lean4:
            raise NotImplementedError

        inside_sections_namespaces = []
        open_namespaces = set()
        found = False

        def _callback(node: Node, parents: List[Node]):
            nonlocal found
            if found:
                return
            elif isinstance(node, NamespaceNode) or isinstance(node, SectionNode):
                inside_sections_namespaces.append(node)
            elif isinstance(node, EndNode):
                begin_node = inside_sections_namespaces.pop()
                assert begin_node.name == node.name, "Namespace or section mismatch"
            elif isinstance(node, OpenNode):
                open_namespaces.update(node.namespaces)
            elif node is thm.ast:
                found = True
                return True

        self.ast.traverse_preorder(_callback, node_cls=None)
        inside_namespaces = [
            node.name
            for node in inside_sections_namespaces
            if isinstance(node, NamespaceNode)
        ]

        return inside_namespaces, list(open_namespaces)

    def check_sanity(self) -> None:
        """Perform some basic sanity checks.

        The function raises exceptions in case of unsuccessful checks.
        """
        assert isinstance(self.root_dir, Path)
        assert isinstance(self.lean_file, LeanFile)
        if self.uses_lean3:
            assert isinstance(self.ast, FileNode)
            assert isinstance(self.exprs, list) and all(
                isinstance(e, Expr) for e in self.exprs
            )
        else:
            isinstance(self.ast, FileNode4)

        assert self.lean_file.root_dir == self.root_dir

        for t in self.get_traced_theorems():
            assert str(self.lean_file.path).endswith(str(t.theorem.file_path))
            assert t.traced_file is None or t.traced_file is self

    def traverse_preorder(self, callback, node_cls: Optional[type] = None):
        """Traverse the AST in preorder.

        Args:
            callback (function): Callback function for visiting AST nodes.
            node_cls (Optional[type], optional): Restrict the application of
                ``callback`` to only nodes of type ``node_cls``.
                Defaults to None, which means applying ``callback`` to all.
        """
        self.ast.traverse_preorder(callback, node_cls)

    def _get_repo_and_relative_path(self) -> Tuple[LeanGitRepo, Path]:
        """Return the repo this file belongs to, as well as the file's path relative to it."""
        packages_dir = self.traced_repo.repo.get_packages_dir()

        if self.path.is_relative_to(packages_dir):
            # The theorem belongs to one of the dependencies.
            p = self.path.relative_to(packages_dir)
            name = p.parts[0]
            repo = self.traced_repo.dependencies[name]
            return repo, p.relative_to(name)
        else:
            # The theorem belongs to the traced repo itself.
            return self.traced_repo.repo, self.path

    def get_traced_theorem(
        self, thm_or_name: Union[Theorem, str]
    ) -> Optional[TracedTheorem]:
        """Return a :class:`TracedTheorem` object given an :class:`Theorem` object
        or its fully-qualified name."""
        if isinstance(thm_or_name, Theorem):
            thm = thm_or_name
        else:
            repo, path = self._get_repo_and_relative_path()
            thm = Theorem(repo, path, thm_or_name)
        result = None
        private_result = None

        def _callback(node: Union[TheoremNode, CommandTheoremNode4], _) -> None:
            nonlocal result, private_result
            if node.full_name == thm.full_name:
                comments = self._filter_comments(node.start, node.end)
                t = TracedTheorem(self.root_dir, thm, node, comments, self)
                if t.is_private:
                    private_result = t
                else:
                    result = t

        node_cls = TheoremNode if self.uses_lean3 else CommandTheoremNode4
        self.ast.traverse_preorder(_callback, node_cls)

        # Prioritize non-private theorems.
        if result is None:
            result = private_result
        return result

    def get_traced_theorems(self) -> List[TracedTheorem]:
        """Return a list of traced theorem in this traced file."""
        traced_theorems = []

        def _callback(node: Union[TheoremNode, CommandTheoremNode4], _) -> None:
            repo, path = self._get_repo_and_relative_path()
            thm = Theorem(repo, path, node.full_name)
            comments = self._filter_comments(node.start, node.end)
            traced_theorems.append(
                TracedTheorem(self.root_dir, thm, node, comments, self)
            )
            # No need to traverse the subtree since theorems cannot be nested.
            return True

        node_cls = TheoremNode if self.uses_lean3 else CommandTheoremNode4
        self.traverse_preorder(_callback, node_cls)
        return traced_theorems

    def _filter_comments(self, start: Pos, end: Pos) -> List[Comment]:
        """Return a list of comments that are contained in the given range."""
        comments = []
        for c in self.comments:
            if c.start < start:
                assert c.end <= start
            elif c.start < end:
                assert c.end <= end
                comments.append(c)
        return comments

    def get_direct_dependencies(self, repo: LeanGitRepo) -> List[Tuple[str, Path]]:
        """Return the names and paths of all modules imported by the current :file:`*.lean` file."""
        deps = set()

        if not self.has_prelude:  # Add the prelude as a dependency.
            packages_dir = repo.get_packages_dir()
            if self.uses_lean3:
                init_lean = Path("library/init/default.lean")
                if self.root_dir.name == "lean":
                    deps.add(("init", init_lean))
                else:
                    deps.add(("init", packages_dir / "lean" / init_lean))
            else:
                assert self.uses_lean4
                init_lean = Path("src/lean/Init.lean")
                if self.root_dir.name == "lean4":
                    deps.add(("Init", init_lean))
                else:
                    deps.add(("Init", packages_dir / "lean4" / init_lean))

        def _callback(node: Union[ModuleNode, ModuleImportNode4], _) -> None:
            if node.module is not None and node.path is not None:
                deps.add((node.module, node.path))

        node_cls = ModuleNode if self.uses_lean3 else ModuleImportNode4
        self.traverse_preorder(_callback, node_cls)
        return list(deps)

    def get_premise_definitions(self) -> List[Dict[str, Any]]:
        """Return all theorems and definitions defined in the current file that
        can be potentially used as premises.

        Returns:
            List[Dict[str, Any]]: _description_
        """
        if self.uses_lean3:
            results = []

            def _callback3(node: Node, _) -> None:
                if is_potential_premise_lean3(node):
                    start, end = node.get_closure()
                    if isinstance(node, TheoremNode):
                        # We assume theorems are defined using keywords "theorem"
                        # or "lemma" but not, e.g., "def".
                        proof_start, _ = node.get_proof_node().get_closure()
                        code = get_code_without_comments(
                            self.lean_file, start, proof_start, self.comments
                        )
                        if code.endswith(":="):
                            code = code[:-2].strip()
                    else:
                        code = get_code_without_comments(
                            self.lean_file, start, end, self.comments
                        )
                    # TODO: For alias, restate_axiom, etc., the code is not very informative.
                    if is_mutual_lean3(node):
                        for s in node.full_name:
                            results.append(
                                {
                                    "full_name": s,
                                    "code": code,
                                    "start": list(start),
                                    "end": list(end),
                                    "kind": node.kind(),
                                }
                            )
                    elif node.name is not None and not node.name.startswith("user__"):
                        results.append(
                            {
                                "full_name": node.full_name,
                                "code": code,
                                "start": list(start),
                                "end": list(end),
                                "kind": node.kind(),
                            }
                        )

            self.traverse_preorder(_callback3, node_cls=None)
            return results
        elif self.uses_lean4:
            results = []

            def _callback4(node: Node4, _) -> None:
                if is_potential_premise_lean4(node):
                    start, end = node.get_closure()
                    if isinstance(node, CommandDeclarationNode4) and node.is_theorem:
                        # We assume theorems are defined using keywords "theorem"
                        # or "lemma" but not, e.g., "def".
                        proof_start, _ = (
                            node.get_theorem_node().get_proof_node().get_closure()
                        )
                        code = get_code_without_comments(
                            self.lean_file, start, proof_start, self.comments
                        )
                        if code.endswith(":="):
                            code = code[:-2].strip()
                    else:
                        code = get_code_without_comments(
                            self.lean_file, start, end, self.comments
                        )
                    # TODO: For alias, restate_axiom, etc., the code is not very informative.
                    if is_mutual_lean4(node):
                        for s in node.full_name:
                            results.append(
                                {
                                    "full_name": s,
                                    "code": code,
                                    "start": list(start),
                                    "end": list(end),
                                    "kind": node.kind(),
                                }
                            )
                    else:
                        results.append(
                            {
                                "full_name": node.full_name,
                                "code": code,
                                "start": list(start),
                                "end": list(end),
                                "kind": node.kind(),
                            }
                        )

            self.traverse_preorder(_callback4, node_cls=None)
            return results

    def to_xml(self) -> str:
        """Serialize a :class:`TracedFile` object to XML."""
        tree = etree.Element(self.__class__.__name__)

        tree.set("path", str(self.path))
        tree.set("md5", compute_md5(self.abs_path))

        self.ast.to_xml(tree)

        if self.uses_lean3:
            if self.exprs is not None:
                exprs_node = etree.SubElement(tree, "Exprs")
                for e in self.exprs:
                    e.to_xml(exprs_node)

        if self.comments is not None:
            comments_node = etree.SubElement(tree, "Comments")
            for c in self.comments:
                c.to_xml(comments_node)

        return etree.tostring(tree, encoding="utf-8", pretty_print=True).decode()

    @classmethod
    def from_xml(
        cls,
        root_dir: Union[str, Path],
        path: Union[str, Path],
        repo: LeanGitRepo,
    ) -> "TracedFile":
        """Load a :class:`TracedFile` object from its :file:`*.trace.xml` file.

        Args:
            root_dir (Union[str, Path]): Root directory of the traced repo.
            path (Union[str, Path]): Path of the :file:`*.trace.xml` file relative to ``root_dir``.
            repo (LeanGitRepo): The repo to which the traced file belongs.
        """
        root_dir = Path(root_dir)
        path = Path(path)
        assert path.suffixes == [".trace", ".xml"]
        lean_path = to_lean_path(root_dir, path, repo)
        lean_file = LeanFile(root_dir, lean_path, repo.uses_lean4)

        tree = etree.parse(path).getroot()
        assert tree.tag == "TracedFile"
        assert tree.attrib["path"] == str(lean_path)
        assert tree.attrib["md5"] == compute_md5(lean_file.abs_path)

        if repo.uses_lean4:
            ast_tree, comments_tree = list(tree)
            ast = FileNode4.from_xml(ast_tree, lean_file)
            exprs = None  # TODO
            comments = [Comment.from_xml(c) for c in comments_tree]
        else:
            ast_tree, exprs_tree, comments_tree = list(tree)
            ast = FileNode.from_xml(ast_tree, lean_file)
            exprs = [Expr.from_xml(e) for e in exprs_tree]
            comments = [Comment.from_xml(c) for c in comments_tree]

        return cls(root_dir, repo, lean_file, ast, exprs, comments)


def _save_xml_to_disk(tf: TracedFile) -> None:
    xml_path = tf.root_dir / to_xml_path(tf.root_dir, tf.path, tf.repo)
    with xml_path.open("wt") as oup:
        oup.write(tf.to_xml())


def _build_dependency_graph(
    seed_files: List[TracedFile], root_dir: Path, repo: LeanGitRepo
) -> nx.DiGraph:
    G = nx.DiGraph()

    for tf in seed_files:
        tf_path_str = str(tf.path)
        assert not G.has_node(tf_path_str)
        G.add_node(tf_path_str, traced_file=tf)

    traced_files = seed_files.copy()
    i = 0

    while i < len(traced_files):
        tf = traced_files[i]
        tf_path_str = str(tf.path)

        for dep_module, dep_path in tf.get_direct_dependencies(repo):
            dep_path_str = str(dep_path)
            if not G.has_node(dep_path_str):
                json_path = to_json_path(root_dir, dep_path, repo)
                tf_dep = TracedFile.from_traced_file(root_dir, json_path, repo)
                G.add_node(dep_path_str, traced_file=tf_dep)
                traced_files.append(tf_dep)

            G.add_edge(tf_path_str, dep_path_str, module=dep_module)

        i += 1

    assert nx.is_directed_acyclic_graph(G)
    return G


@ray.remote
class _TracedRepoHelper:
    """
    Helper class serving as Ray actor.
    """

    def __init__(self, root_dir: Path, repo: LeanGitRepo) -> None:
        self.root_dir = root_dir
        self.repo = repo

    def parse_traced_file(self, path: Path) -> TracedFile:
        return TracedFile.from_traced_file(self.root_dir, path, self.repo)

    def save_xml_to_disk(self, tf: TracedFile) -> None:
        return _save_xml_to_disk(tf)

    def load_xml_from_disk(self, path: Path) -> TracedFile:
        return TracedFile.from_xml(self.root_dir, path, self.repo)


@dataclass(frozen=True, eq=False)
class TracedRepo:
    """A traced repo is a Lean repo of traced files and additional information, such as
    other repos it depends on, as well as the dependency graph between files.
    """

    repo: LeanGitRepo
    """The corresponding Lean repo.
    """

    dependencies: Dict[str, LeanGitRepo]
    """Dictionary mapping the name of each dependency to a :class:`LeanGitRepo` object.
    """

    root_dir: Path
    """Root directory of the traced repo.
    """

    traced_files_graph: nx.DiGraph = field(repr=False)
    """Dependency graph between files in the repo.
    
    The graph is a DAG, and there is an edge from file :file:`X` to file :file:`Y`
    if and only if :file:`X` imports :file:`Y`
    """

    def __post_init__(self) -> None:
        assert self.root_dir.is_absolute()

    def __setstate__(self, state) -> None:
        object.__setattr__(self, "__dict__", state)
        self._update_traced_files()

    @property
    def name(self) -> str:
        """Name of the repo."""
        return self.repo.name

    @property
    def uses_lean3(self) -> bool:
        """Whether the repo uses Lean 3."""
        return self.repo.uses_lean3

    @property
    def uses_lean4(self) -> bool:
        """Whether the repo uses Lean 4."""
        return self.repo.uses_lean4

    def show(self) -> None:
        """Show the repo in the default browser."""
        self.repo.show()

    def check_sanity(self) -> None:
        """Perform some basic sanity checks.

        The function raises exceptions in case of unsuccessful checks.
        """
        logger.debug(f"Checking the sanity of {self}")
        assert isinstance(self.repo, LeanGitRepo)
        assert isinstance(self.dependencies, dict)
        for k, v in self.dependencies.items():
            assert isinstance(k, str) and isinstance(v, LeanGitRepo)
        assert isinstance(self.root_dir, Path)
        assert isinstance(self.traced_files_graph, nx.DiGraph)

        assert self.repo not in self.dependencies.values()
        assert len(self.dependencies) == 0 or not self.repo.is_lean

        json_files = {
            p.relative_to(self.root_dir) for p in self.root_dir.glob("**/*.ast.json")
        }
        lean_files = {
            p.relative_to(self.root_dir) for p in self.root_dir.glob("**/*.lean")
        }
        xml_files = {
            p.relative_to(self.root_dir) for p in self.root_dir.glob("**/*.trace.xml")
        }
        path_files = {
            p.relative_to(self.root_dir) for p in self.root_dir.glob("**/*.dep_paths")
        }

        if not LOAD_USED_PACKAGES_ONLY:
            assert len(json_files) == self.traced_files_graph.number_of_nodes()

        for path_str, tf_node in self.traced_files_graph.nodes.items():
            tf = tf_node["traced_file"]
            path = Path(path_str)
            tf.check_sanity()
            assert tf.path == path and tf.root_dir == self.root_dir
            assert tf.traced_repo is None or tf.traced_repo is self
            assert path in lean_files
            assert (
                to_dep_path(self.root_dir, path, self.repo) in path_files
            ), to_dep_path(self.root_dir, path, self.repo)
            assert (
                to_json_path(self.root_dir, path, self.repo) in json_files
            ), to_json_path(self.root_dir, path, self.repo)
            if len(xml_files) > 0:
                assert (
                    to_xml_path(self.root_dir, path, self.repo) in xml_files
                ), to_xml_path(self.root_dir, path, self.repo)

    @classmethod
    def from_traced_files(cls, root_dir: Union[str, Path]) -> None:
        """Construct a :class:`TracedRepo` object by parsing :file:`*.ast.json` and :file:`*.path` files
           produced by :code:`lean --ast --tsast --tspp` (Lean 3) or :file:`ExtractData.lean` (Lean 4).

        Args:
            root_dir (Union[str, Path]): Root directory of the traced repo.
        """
        root_dir = Path(root_dir).resolve()
        if not is_git_repo(root_dir):
            raise RuntimeError(f"{root_dir} is not a Git repo.")
        repo = LeanGitRepo.from_path(root_dir)

        json_paths = list(root_dir.glob("**/*.ast.json"))
        random.shuffle(json_paths)
        logger.debug(
            f"Parsing {len(json_paths)} *.ast.json files in {root_dir} with {NUM_WORKERS} workers"
        )

        if NUM_WORKERS <= 1:
            traced_files = [
                TracedFile.from_traced_file(root_dir, path, repo)
                for path in tqdm(json_paths)
            ]
        else:
            with ray_actor_pool(_TracedRepoHelper, root_dir, repo) as pool:
                traced_files = list(
                    tqdm(
                        pool.map_unordered(
                            lambda a, p: a.parse_traced_file.remote(p), json_paths
                        ),
                        total=len(json_paths),
                    )
                )

        dependencies = repo.get_dependencies(root_dir)
        traced_files_graph = _build_dependency_graph(traced_files, root_dir, repo)
        traced_repo = cls(repo, dependencies, root_dir, traced_files_graph)
        traced_repo._update_traced_files()
        return traced_repo

    @property
    def traced_files(self) -> Generator[TracedFile, None, None]:
        """Return an iterator of traced files from their dependency graph."""
        for _, tf_node in self.traced_files_graph.nodes.items():
            yield tf_node["traced_file"]

    @property
    def num_traced_files(self) -> int:
        """Number of traced files in the repo."""
        return len(self.traced_files_graph)

    def get_traced_file(self, path: Union[str, Path]) -> TracedFile:
        """Return a traced file by its path."""
        return self.traced_files_graph.nodes[str(path)]["traced_file"]

    def _update_traced_files(self) -> None:
        for tf in self.traced_files:
            tf.traced_repo = self

    def save_to_disk(self) -> None:
        """Save all traced files in the repo to the disk as :file:`*.trace.xml` files."""
        logger.debug(
            f"Saving {self.num_traced_files} traced XML files to {self.root_dir} with {NUM_WORKERS} workers"
        )
        if NUM_WORKERS <= 1:
            for tf in tqdm(self.traced_files, total=self.num_traced_files):
                _save_xml_to_disk(tf)
        else:
            with ray_actor_pool(_TracedRepoHelper, self.root_dir, self.repo) as pool:
                list(
                    tqdm(
                        pool.map_unordered(
                            lambda a, tf: a.save_xml_to_disk.remote(tf),
                            self.traced_files,
                        ),
                        total=self.num_traced_files,
                    )
                )

    @classmethod
    def load_from_disk(cls, root_dir: Union[str, Path]) -> "TracedRepo":
        """Load a traced repo from :file:`*.trace.xml` files."""
        root_dir = Path(root_dir).resolve()
        if not is_git_repo(root_dir):
            raise RuntimeError(f"{root_dir} is not a Git repo.")
        repo = LeanGitRepo.from_path(root_dir)

        xml_paths = list(root_dir.glob("**/*.trace.xml"))
        logger.debug(
            f"Loading {len(xml_paths)} traced XML files from {root_dir} with {NUM_WORKERS} workers"
        )

        # Start from files in the target repo as seeds.
        # Only load dependency files that are actually used.
        if LOAD_USED_PACKAGES_ONLY:
            xml_paths = [
                p
                for p in xml_paths
                if not "lake-packages/" in str(p) and not ".lake/packages" in str(p)
            ]

        if NUM_WORKERS <= 1:
            traced_files = [
                TracedFile.from_xml(root_dir, path, repo) for path in tqdm(xml_paths)
            ]
        else:
            with ray_actor_pool(_TracedRepoHelper, root_dir, repo) as pool:
                traced_files = list(
                    tqdm(
                        pool.map_unordered(
                            lambda a, path: a.load_xml_from_disk.remote(path), xml_paths
                        ),
                        total=len(xml_paths),
                    )
                )

        dependencies = repo.get_dependencies(root_dir)
        traced_files_graph = _build_dependency_graph(traced_files, root_dir, repo)
        traced_repo = cls(repo, dependencies, root_dir, traced_files_graph)
        traced_repo._update_traced_files()
        return traced_repo

    def get_traced_theorems(self) -> List[TracedTheorem]:
        """Return all traced theorems in the repo."""
        return list(
            itertools.chain.from_iterable(
                tf.get_traced_theorems() for tf in self.traced_files
            )
        )

    def get_traced_theorem(self, thm: Theorem) -> Optional[TracedTheorem]:
        """Return a :class:`TracedTheorem` object corresponding to ``thm``"""
        if thm.repo == self.repo:
            path = Path(thm.repo.name) / thm.file_path
        else:
            assert thm.repo in self.dependencies.values()
            packages_dir = thm.repo.get_packages_dir()
            path = Path(self.name) / packages_dir / thm.repo.name / thm.file_path
        return self.get_traced_file(path).get_traced_theorem(thm.full_name)

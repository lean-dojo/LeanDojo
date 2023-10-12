import re
from lxml import etree
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, unescape
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

from ....utils import (
    camel_case,
    is_optional_type,
    remove_optional_type,
    parse_int_list,
    parse_str_list,
)
from .ast_utils import parse_pos
from ...lean import Pos, LeanFile


EXPR_TAGS = str


@dataclass(frozen=True)
class Node:
    lean_file: LeanFile
    start: Pos
    end: Pos
    children: List["Node"] = field(repr=False)
    id: str

    @classmethod
    def from_data(cls, data: Dict[str, Any], idx: int, lean_file: LeanFile) -> "Node":
        subcls = cls._kind_to_node_type(data["ast"][idx]["kind"])
        return subcls.from_data(data, idx, lean_file)  # type: ignore

    @classmethod
    def _kind_to_node_type(cls, kind: str) -> type:
        if kind == "{":
            return LeftCurlyBracketNode
        elif kind == "(":
            return LeftParenthesisNode
        elif kind == "<|>":
            return OrElseNode
        elif kind == "class_inductive":
            return ClassInductiveNode
        elif kind == "user_command":
            return UserCommandNode
        elif kind.isalpha():
            cls_name = camel_case(kind) + "Node"
            gbs = globals()
            if cls_name in gbs:
                return gbs[cls_name]  # type: ignore
            else:
                logger.warning(kind)
                return OtherNode
        else:
            return OtherNode

    @classmethod
    def kind(cls: type) -> str:
        return cls.__name__[:-4].lower()

    @classmethod
    def _get_expr(cls, data: Dict[str, Any], idx: int) -> Optional[EXPR_TAGS]:
        if "expr" in data["ast"][idx]:
            return data["expr_tags_map"][data["ast"][idx]["expr"]]  # type: ignore
        else:
            return None

    @property
    def is_buggy(self) -> bool:
        """See :ref:`limitations`."""
        return False

    def traverse_preorder(
        self,
        callback: Callable[["Node", List["Node"]], Any],
        node_cls: Optional[type],
        parents: List["Node"] = [],
    ) -> None:
        if node_cls is None or isinstance(self, node_cls):
            if callback(self, parents):
                return
        for child in self.children:
            child.traverse_preorder(callback, node_cls, parents + [self])

    @property
    def text(self) -> str:
        return self.lean_file[self.start : self.end]  # type: ignore

    def to_xml(self, parent: etree.Element) -> None:
        tree = etree.SubElement(parent, self.__class__.__name__)

        for k in self.__dataclass_fields__:
            if k in ("lean_file", "children"):
                continue
            v = getattr(self, k)
            if v is not None:
                v = escape(str(v), entities={'"': "&quot;"})
                tree.set(k, v)

        for child in self.children:
            child.to_xml(tree)

    @classmethod
    def from_xml(cls, tree: etree.Element, lean_file: LeanFile) -> "Node":
        subcls = globals()[tree.tag]
        start = Pos.from_str(tree.attrib["start"])
        end = Pos.from_str(tree.attrib["end"])
        children = [Node.from_xml(subtree, lean_file) for subtree in tree]
        kwargs = {}

        for field in subcls.__dataclass_fields__.values():
            if field.name in ("lean_file", "start", "end", "children"):
                continue
            v = tree.attrib.get(field.name, None)
            if v is None:
                kwargs[field.name] = None
                continue

            assert isinstance(v, str)
            v = unescape(v, entities={"&quot;": '"'})
            tp = (
                remove_optional_type(field.type)
                if is_optional_type(field.type)
                else field.type
            )
            if tp is Pos:
                kwargs[field.name] = Pos.from_str(v)
            elif tp is Path:
                kwargs[field.name] = Path(v)
            elif tp is List[int]:
                kwargs[field.name] = parse_int_list(v)
            elif tp is List[str]:
                kwargs[field.name] = parse_str_list(v)
            else:
                kwargs[field.name] = v  # type: ignore

        return subcls(lean_file, start, end, children, **kwargs)  # type: ignore

    def get_closure(self) -> Tuple[Pos, Pos]:
        """Return the earliest start position and latest end position of all nodes in the subtree."""
        start = self.start
        end = self.end

        def _callback(node: "Node", parents: List["Node"]) -> None:
            nonlocal start
            nonlocal end
            if node.is_buggy or type(node) in (DocNode, MdocNode):
                return True
            start = min(start, node.start)
            end = max(end, node.end)

        self.traverse_preorder(_callback, node_cls=None)
        return start, end


def _parse_children(data: Dict[str, Any], idx: int, lean_file: LeanFile) -> List[Node]:
    if "children" in data["ast"][idx]:
        return [
            Node.from_data(data, j, lean_file)
            for j in data["ast"][idx]["children"]
            if data["ast"][j] is not None
        ]
    else:
        return []


@dataclass(frozen=True)
class FileNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "FileNode":
        assert data["ast"][idx]["kind"] == "file"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class CommandsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "CommandsNode":
        assert data["ast"][idx]["kind"] == "commands"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class CommandNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "CommandNode":
        assert data["ast"][idx]["kind"] == "command"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))

    @property
    def is_buggy(self) -> bool:
        """See :ref:`limitations`."""
        return (
            len(self.children) == 1
            and isinstance(self.children[0], DefinitionNode)
            and self.children[0].is_buggy
        )


@dataclass(frozen=True)
class LocalNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "LocalNode":
        assert data["ast"][idx]["kind"] == "local"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ArgsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ArgsNode":
        assert data["ast"][idx]["kind"] == "args"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class SymNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SymNode":
        assert data["ast"][idx]["kind"] == "sym"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class QuotedNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "QuotedNode":
        assert data["ast"][idx]["kind"] == "quoted"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class PrefixNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PrefixNode":
        assert data["ast"][idx]["kind"] == "prefix"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class InfixNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "InfixNode":
        assert data["ast"][idx]["kind"] == "infix"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class InfixrNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "InfixrNode":
        assert data["ast"][idx]["kind"] == "infixr"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class VarNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "VarNode":
        assert data["ast"][idx]["kind"] == "var"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class NamespaceNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "NamespaceNode":
        assert data["ast"][idx]["kind"] == "namespace"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        assert len(children) == 1 and isinstance(children[0], IdentNode)
        name = children[0].ident
        return cls(lean_file, start, end, children, str(idx), name)


@dataclass(frozen=True)
class SectionNode(Node):
    name: Optional[str]
    end_node_id: Optional[int] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SectionNode":
        assert data["ast"][idx]["kind"] == "section"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        if len(children) == 0:
            name = None
        else:
            assert len(children) == 1 and isinstance(children[0], IdentNode)
            name = children[0].ident
        return cls(lean_file, start, end, children, str(idx), name)


@dataclass(frozen=True)
class OpenNode(Node):
    namespaces: List[str]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "OpenNode":
        assert data["ast"][idx]["kind"] == "open"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        namespaces = cls._get_namespaces(children)
        return cls(lean_file, start, end, children, str(idx), namespaces)

    @classmethod
    def _get_namespaces(cls, children) -> List[str]:
        namespaces = []
        for child in children:
            assert isinstance(child, GroupNode)
            if len(child.children) == 1:
                gchild = child.children[0]
                assert isinstance(gchild, IdentNode)
                namespaces.append(gchild.ident)
        return namespaces


@dataclass(frozen=True)
class VariablesNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "VariablesNode":
        assert data["ast"][idx]["kind"] == "variables"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class VariableNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "VariableNode":
        assert data["ast"][idx]["kind"] == "variable"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class MaxNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "MaxNode":
        assert data["ast"][idx]["kind"] == "max"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class CollectionNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "CollectionNode":
        assert data["ast"][idx]["kind"] == "collection"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ChoicesNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ChoicesNode":
        assert data["ast"][idx]["kind"] == "choices"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ChoiceNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ChoiceNode":
        assert data["ast"][idx]["kind"] == "choice"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ImportsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ImportsNode":
        assert data["ast"][idx]["kind"] == "imports"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ImportNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ImportNode":
        assert data["ast"][idx]["kind"] == "import"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ExportNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ExportNode":
        assert data["ast"][idx]["kind"] == "export"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ModuleNode(Node):
    module: str
    path: Path

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ModuleNode":
        assert data["ast"][idx]["kind"] == "module"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        if isinstance(data["ast"][idx]["value"], str):
            module = data["ast"][idx]["value"]
            suffix = data["ast"][idx]["value"]
        else:
            assert isinstance(data["ast"][idx]["value"], list)
            module = ".".join(data["ast"][idx]["value"])
            suffix = "/".join(data["ast"][idx]["value"])
        for p in data["module_paths"]:
            if p.endswith(suffix + ".lean") or p.endswith(suffix + "/default.lean"):
                return cls(lean_file, start, end, children, str(idx), module, Path(p))
        else:
            # Relative imports end with `_\d` and are not recorded in `*.dep_paths`.
            m = re.fullmatch(r"(?P<module>.+)_(?P<n>\d+)", module)
            if m is not None:
                n = int(m["n"])
                path = lean_file.path.parents[n] / (m["module"] + ".lean")
                return cls(lean_file, start, end, children, str(idx), module, path)
            else:
                raise ValueError(f"Unable to find the path of {module}")


@dataclass(frozen=True)
class ExplicitNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ExplicitNode":
        assert data["ast"][idx]["kind"] == "explicit"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class TheoremNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "TheoremNode":
        assert data["ast"][idx]["kind"] == "theorem"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        else:
            assert (
                isinstance(children[0], ModifiersNode)
                and isinstance(children[1], LevelsNode)
                and isinstance(children[2], IdentNode)
            )
            return children[2]

    def get_proof_node(self) -> Node:
        if (
            isinstance(self.children[-1], OtherNode)
            and self.children[-1].kind == "using_well_founded"
        ):
            return self.children[-2]
        else:
            return self.children[-1]

    def has_tactic_proof(self) -> bool:
        node = self.get_proof_node()
        return (
            isinstance(node, NotationNode)
            and len(node.children) == 1
            and type(node.children[0]) in (ByNode, BeginNode)
        )

    def is_private(self) -> bool:
        return isinstance(self.children[0], ModifiersNode) and any(
            isinstance(gchild, PrivateNode) for gchild in self.children[0].children
        )


@dataclass(frozen=True)
class DefinitionNode(Node):
    """
    A definition or a mutual recursion definition.
    """

    name: Union[str, List[str]]
    full_name: Optional[Union[str, List[str]]] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "DefinitionNode":
        assert data["ast"][idx]["kind"] == "definition"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        node = cls._get_ident_node(children)
        name: Union[str, List[str]]
        if isinstance(node, IdentNode):
            name = node.ident
        else:
            assert isinstance(node, MutualsNode)
            name = node.names
        return cls(lean_file, start, end, children, str(idx), name)

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.name, str)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> Union["IdentNode", "MutualsNode"]:
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        elif isinstance(children[2], IdentNode):
            assert isinstance(children[0], ModifiersNode) and isinstance(
                children[1], LevelsNode
            )
            return children[2]
        else:
            # Mutual recursion (https://leanprover.github.io/theorem_proving_in_lean/induction_and_recursion.html#mutual-recursion).
            assert isinstance(children[2], MutualsNode)
            return children[2]

    @property
    def is_buggy(self) -> bool:
        """See :ref:`limitations`."""
        return (
            self.start.line_nb == self.end.line_nb == 1
            and isinstance(self.name, str)
            and self.name.startswith("user__.")
        )


@dataclass(frozen=True)
class IdentNode(Node):
    ident: str
    expr: Optional[EXPR_TAGS]
    full_name: Optional[str] = None
    def_path: Optional[str] = None
    def_pos: Optional[Pos] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IdentNode":
        assert data["ast"][idx]["kind"] == "ident"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        assert children == []
        ident = data["ast"][idx]["value"]
        if isinstance(ident, list):
            ident = ".".join(ident)
        expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx), ident, expr)


@dataclass(frozen=True)
class LhsNode(Node):
    # expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "LhsNode":
        assert data["ast"][idx]["kind"] == "lhs"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        # expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx))  # , expr)


@dataclass(frozen=True)
class ParseNode(Node):
    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ParseNode":
        assert data["ast"][idx]["kind"] == "parse"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx), expr)


@dataclass(frozen=True)
class VarsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "VarsNode":
        assert data["ast"][idx]["kind"] == "vars"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class TokenNode(Node):
    token: str

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "TokenNode":
        assert data["ast"][idx]["kind"] == "token"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        assert children == []
        return cls(lean_file, start, end, children, str(idx), data["ast"][idx]["value"])


def get_tactic_data(data: Dict[str, Any], ast_idx: int) -> Optional[Dict[str, Any]]:
    if "tactics" not in data:
        return None

    for t in data["tactics"]:
        if t["ast"] == ast_idx:
            return t  # type: ignore

    return None


@dataclass(frozen=True)
class TacticNode(Node):
    # Currently unable to trace tactics in `example`, `instance`, etc.
    tactic: str
    state_before: Optional[str]
    state_after: Optional[str]
    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "TacticNode":
        assert data["ast"][idx]["kind"] == "tactic"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)

        raw_tactic = lean_file[start:end]  # type: ignore
        tactic = " ".join(_.strip() for _ in raw_tactic.splitlines())

        tac_data = get_tactic_data(data, idx)
        if tac_data is None:
            # Tactics outside theorem/lemma definitions are not recorded.
            state_before = state_after = None
        else:
            state_before_data = data["states"][tac_data["start"]]
            state_after_data = data["states"][tac_data["end"]]
            state_before = state_before_data["pp"]
            state_after = state_after_data["pp"]

        expr = cls._get_expr(data, idx)
        return cls(
            lean_file,
            start,
            end,
            children,
            str(idx),
            tactic,
            state_before,
            state_after,
            expr,
        )


@dataclass(frozen=True)
class SorryNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SorryNode":
        assert data["ast"][idx]["kind"] == "sorry"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class IntrosNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IntrosNode":
        assert data["ast"][idx]["kind"] == "intros"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class IntroNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IntroNode":
        assert data["ast"][idx]["kind"] == "intro"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class EqnNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "EqnNode":
        assert data["ast"][idx]["kind"] == "eqn"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ConstNode(Node):
    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ConstNode":
        assert data["ast"][idx]["kind"] == "const"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx), expr)


@dataclass(frozen=True)
class ConstantNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ConstantNode":
        assert data["ast"][idx]["kind"] == "constant"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        else:
            assert (
                isinstance(children[0], ModifiersNode)
                and isinstance(children[1], LevelsNode)
                and isinstance(children[2], IdentNode)
            )
            return children[2]


@dataclass(frozen=True)
class ConstantsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ConstantsNode":
        assert data["ast"][idx]["kind"] == "constants"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class AxiomNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AxiomNode":
        assert data["ast"][idx]["kind"] == "axiom"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        else:
            assert (
                isinstance(children[0], ModifiersNode)
                and isinstance(children[1], LevelsNode)
                and isinstance(children[2], IdentNode)
            )
            return children[2]


@dataclass(frozen=True)
class LevelsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "LevelsNode":
        assert data["ast"][idx]["kind"] == "levels"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ParamNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ParamNode":
        assert data["ast"][idx]["kind"] == "param"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class StringNode(Node):
    "String literal"
    string: str

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "StringNode":
        assert data["ast"][idx]["kind"] == "string"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx), data["ast"][idx]["value"])


@dataclass(frozen=True)
class IfNode(Node):
    "`if ... then ... else ...` term"

    @classmethod
    def from_data(cls, data: Dict[str, Any], idx: int, lean_file: LeanFile) -> "IfNode":
        assert data["ast"][idx]["kind"] == "if"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class MatchNode(Node):
    "`match ... end` term"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "MatchNode":
        assert data["ast"][idx]["kind"] == "match"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class LetNode(Node):
    "`let ... in ...` term"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "LetNode":
        assert data["ast"][idx]["kind"] == "let"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ShowNode(Node):
    "`show ..., from ...` term"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ShowNode":
        assert data["ast"][idx]["kind"] == "show"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class FromNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "FromNode":
        assert data["ast"][idx]["kind"] == "from"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class HaveNode(Node):
    "`have ... from ... by ...`"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "HaveNode":
        assert data["ast"][idx]["kind"] == "have"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class AssumeNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AssumeNode":
        assert data["ast"][idx]["kind"] == "assume"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class SubtypeNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SubtypeNode":
        assert data["ast"][idx]["kind"] == "subtype"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ExtendsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ExtendsNode":
        assert data["ast"][idx]["kind"] == "extends"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ParentNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ParentNode":
        assert data["ast"][idx]["kind"] == "parent"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class TupleNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "TupleNode":
        assert data["ast"][idx]["kind"] == "tuple"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class PatNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PatNode":
        assert data["ast"][idx]["kind"] == "pat"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ClassNode(Node):
    "Type class"

    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ClassNode":
        assert data["ast"][idx]["kind"] == "class"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        else:
            assert (
                isinstance(children[0], ModifiersNode)
                and isinstance(children[1], LevelsNode)
                and isinstance(children[2], IdentNode)
            )
            return children[2]


@dataclass(frozen=True)
class ClassInductiveNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ClassInductiveNode":
        assert data["ast"][idx]["kind"] == "class_inductive"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert isinstance(children[0], ModifiersNode)
            return children[1]
        else:
            raise Exception("Unexpected children: " + str(children))


@dataclass(frozen=True)
class ModifiersNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ModifiersNode":
        assert data["ast"][idx]["kind"] == "modifiers"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class NatNode(Node):
    "Natural number literal, e.g., `1`"
    nat: int
    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "NatNode":
        assert data["ast"][idx]["kind"] == "nat"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        assert children == []
        expr = cls._get_expr(data, idx)
        return cls(
            lean_file,
            start,
            end,
            children,
            str(idx),
            data["ast"][idx]["value"],
            expr,
        )


@dataclass(frozen=True)
class FieldsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "FieldsNode":
        assert data["ast"][idx]["kind"] == "fields"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ExprNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ExprNode":
        assert data["ast"][idx]["kind"] == "expr"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class EndNode(Node):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "EndNode":
        assert data["ast"][idx]["kind"] == "end"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        if len(children) == 0:
            name = None
        else:
            assert len(children) == 1 and isinstance(children[0], IdentNode)
            name = children[0].ident
        return cls(lean_file, start, end, children, str(idx), name)


@dataclass(frozen=True)
class GroupNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "GroupNode":
        assert data["ast"][idx]["kind"] == "group"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class AppNode(Node):
    # expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AppNode":
        assert data["ast"][idx]["kind"] == "app"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        # expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx))  # , expr)


@dataclass(frozen=True)
class BindersNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "BindersNode":
        assert data["ast"][idx]["kind"] == "binders"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class BeginNode(Node):
    "`begin` ... `end` proof block"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "BeginNode":
        assert data["ast"][idx]["kind"] == "begin"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class AttrsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AttrsNode":
        assert data["ast"][idx]["kind"] == "attrs"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class InstanceNode(Node):
    name: Optional[str]
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "InstanceNode":
        assert data["ast"][idx]["kind"] == "instance"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        ident_node = cls._get_ident_node(children)
        if ident_node is not None:
            name = ident_node.ident
        else:
            name = None
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> Optional["IdentNode"]:
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[0], ModifiersNode) and isinstance(
            children[1], IdentNode
        ):
            return children[1]
        else:
            return None


@dataclass(frozen=True)
class FunNode(Node):
    "Lambda function"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "FunNode":
        assert data["ast"][idx]["kind"] == "fun"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class SourcesNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SourcesNode":
        assert data["ast"][idx]["kind"] == "sources"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class StructinstNode(Node):
    "Creating a structure object using `{ ... }`, e.g., `{foo := 1, bar := 2}`"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "StructinstNode":
        assert data["ast"][idx]["kind"] == "structinst"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class InductiveNode(Node):
    name: Union[str, List[str]]
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "InductiveNode":
        assert data["ast"][idx]["kind"] == "inductive"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        node = cls._get_ident_node(children)
        name: Union[str, List[str]]
        if isinstance(node, IdentNode):
            name = node.ident
        else:
            assert isinstance(node, MutualsNode)
            name = node.names
        return cls(lean_file, start, end, children, str(idx), name)

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.name, str)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> Union["IdentNode", "MutualsNode"]:
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        elif isinstance(children[2], IdentNode):
            assert isinstance(children[0], ModifiersNode) and isinstance(
                children[1], LevelsNode
            )
            return children[2]
        else:
            # Mutual recursion (https://leanprover.github.io/theorem_proving_in_lean/induction_and_recursion.html#mutual-recursion).
            assert isinstance(children[2], MutualsNode)
            return children[2]


@dataclass(frozen=True)
class StructureNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "StructureNode":
        assert data["ast"][idx]["kind"] == "structure"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        else:
            assert (
                isinstance(children[0], ModifiersNode)
                and isinstance(children[1], LevelsNode)
                and isinstance(children[2], IdentNode)
            )
            return children[2]


@dataclass(frozen=True)
class NoncomputableNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "NoncomputableNode":
        assert data["ast"][idx]["kind"] == "noncomputable"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class IncludeNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IncludeNode":
        assert data["ast"][idx]["kind"] == "include"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class OmitNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "OmitNode":
        assert data["ast"][idx]["kind"] == "omit"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class EqnsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "EqnsNode":
        assert data["ast"][idx]["kind"] == "eqns"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ByNode(Node):
    "`by` ... proof block"

    @classmethod
    def from_data(cls, data: Dict[str, Any], idx: int, lean_file: LeanFile) -> "ByNode":
        assert data["ast"][idx]["kind"] == "by"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class FieldNode(Node):
    "Accessing the field of a structure, e.g., `b.size`"

    # expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "FieldNode":
        assert data["ast"][idx]["kind"] == "field"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        # expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx))  # , expr)


@dataclass(frozen=True)
class TypeNode(Node):
    # expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "TypeNode":
        assert data["ast"][idx]["kind"] == "Type"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        # expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx))  # , expr)


@dataclass(frozen=True)
class SortNode(Node):
    # expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SortNode":
        assert data["ast"][idx]["kind"] == "Sort"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        # expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx))  # , expr)


@dataclass(frozen=True)
class AttrNode(Node):
    attr: str

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AttrNode":
        assert data["ast"][idx]["kind"] == "attr"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx), data["ast"][idx]["value"])


@dataclass(frozen=True)
class AttributeNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AttributeNode":
        assert data["ast"][idx]["kind"] == "attribute"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ExprsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ExprsNode":
        assert data["ast"][idx]["kind"] == "exprs"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class DocNode(Node):
    "Doc string"
    doc: str

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "DocNode":
        assert data["ast"][idx]["kind"] == "doc"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx), data["ast"][idx]["value"])


@dataclass(frozen=True)
class MdocNode(Node):
    "Markdown doc string"
    mdoc: str

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "MdocNode":
        assert data["ast"][idx]["kind"] == "mdoc"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx), data["ast"][idx]["value"])


@dataclass(frozen=True)
class UniversesNode(Node):
    "Universes declaration, e.g., `universes u v w`"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "UniversesNode":
        assert data["ast"][idx]["kind"] == "universes"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class MetaNode(Node):
    "`meta` keyword for metaprogramming"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "MetaNode":
        assert data["ast"][idx]["kind"] == "meta"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ProtectedNode(Node):
    "`protected` keyword"

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ProtectedNode":
        assert data["ast"][idx]["kind"] == "protected"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class NotationNode(Node):
    # expr: Optional[EXPR_TAGS]
    value: Optional[str]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "NotationNode":
        assert data["ast"][idx]["kind"] == "notation"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        # expr = cls._get_expr(data, idx)
        if "value" in data["ast"][idx]:
            value = data["ast"][idx]["value"]
        else:
            value = None
        return cls(lean_file, start, end, children, str(idx), value)  # , expr)


@dataclass(frozen=True)
class AbbreviationNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "AbbreviationNode":
        assert data["ast"][idx]["kind"] == "abbreviation"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        name = cls._get_ident_node(children).ident
        return cls(lean_file, start, end, children, str(idx), name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], IdentNode):
            return children[0]
        elif isinstance(children[1], IdentNode):
            assert type(children[0]) in (ModifiersNode, LevelsNode)
            return children[1]
        else:
            assert (
                isinstance(children[0], ModifiersNode)
                and isinstance(children[1], LevelsNode)
                and isinstance(children[2], IdentNode)
            )
            return children[2]


@dataclass(frozen=True)
class PriorityNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PriorityNode":
        assert data["ast"][idx]["kind"] == "priority"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class TheoryNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "TheoryNode":
        assert data["ast"][idx]["kind"] == "theory"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class IndsNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IndsNode":
        assert data["ast"][idx]["kind"] == "inds"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class StepNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "StepNode":
        assert data["ast"][idx]["kind"] == "step"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ScopedNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ScopedNode":
        assert data["ast"][idx]["kind"] == "scoped"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class RenamingNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "RenamingNode":
        assert data["ast"][idx]["kind"] == "renaming"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class EvalNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "EvalNode":
        assert data["ast"][idx]["kind"] == "eval"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ParametersNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ParametersNode":
        assert data["ast"][idx]["kind"] == "parameters"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class FoldrNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "FoldrNode":
        assert data["ast"][idx]["kind"] == "foldr"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class SufficesNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SufficesNode":
        assert data["ast"][idx]["kind"] == "suffices"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class BoolNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "BoolNode":
        assert data["ast"][idx]["kind"] == "bool"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class MutualNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "MutualNode":
        assert data["ast"][idx]["kind"] == "mutual"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class DecimalNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "DecimalNode":
        assert data["ast"][idx]["kind"] == "decimal"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class PrivateNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PrivateNode":
        assert data["ast"][idx]["kind"] == "private"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ScopeNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ScopeNode":
        assert data["ast"][idx]["kind"] == "scope"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class SepNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "SepNode":
        assert data["ast"][idx]["kind"] == "sep"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class BodiesNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "BodiesNode":
        assert data["ast"][idx]["kind"] == "bodies"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class PostfixNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PostfixNode":
        assert data["ast"][idx]["kind"] == "postfix"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class UniverseNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "UniverseNode":
        assert data["ast"][idx]["kind"] == "universe"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class IndNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IndNode":
        assert data["ast"][idx]["kind"] == "ind"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class HideNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "HideNode":
        assert data["ast"][idx]["kind"] == "hide"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class DoNode(Node):
    @classmethod
    def from_data(cls, data: Dict[str, Any], idx: int, lean_file: LeanFile) -> "DoNode":
        assert data["ast"][idx]["kind"] == "do"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class MkNode(Node):
    @classmethod
    def from_data(cls, data: Dict[str, Any], idx: int, lean_file: LeanFile) -> "MkNode":
        assert data["ast"][idx]["kind"] == "mk"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ImaxNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ImaxNode":
        assert data["ast"][idx]["kind"] == "imax"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class HidingNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "HidingNode":
        assert data["ast"][idx]["kind"] == "hiding"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class CalcNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "CalcNode":
        assert data["ast"][idx]["kind"] == "calc"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class InfixlNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "InfixlNode":
        assert data["ast"][idx]["kind"] == "infixl"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class BodyNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "BodyNode":
        assert data["ast"][idx]["kind"] == "body"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class CharNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "CharNode":
        assert data["ast"][idx]["kind"] == "char"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ParameterNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ParameterNode":
        assert data["ast"][idx]["kind"] == "parameter"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class MutualsNode(Node):
    names: List[str]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "MutualsNode":
        assert data["ast"][idx]["kind"] == "mutuals"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        assert len(children) > 1 and all(isinstance(_, IdentNode) for _ in children)
        names = [_.ident for _ in children]
        return cls(
            lean_file,
            start,
            end,
            children,
            str(idx),
            names,
        )


@dataclass(frozen=True)
class ExampleNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ExampleNode":
        assert data["ast"][idx]["kind"] == "example"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class IndicesNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "IndicesNode":
        assert data["ast"][idx]["kind"] == "indices"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class PreludeNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PreludeNode":
        assert data["ast"][idx]["kind"] == "prelude"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class PrecedenceNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "PrecedenceNode":
        assert data["ast"][idx]["kind"] == "precedence"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class ReserveNode(Node):
    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "ReserveNode":
        assert data["ast"][idx]["kind"] == "reserve"
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        return cls(lean_file, start, end, children, str(idx))


@dataclass(frozen=True)
class LeftCurlyBracketNode(Node):
    "`{`"

    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "LeftCurlyBracketNode":
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx), expr)

    @classmethod
    def kind(cls) -> str:
        return "{"


@dataclass(frozen=True)
class LeftParenthesisNode(Node):
    "`(`"

    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "LeftParenthesisNode":
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx), expr)

    @classmethod
    def kind(cls) -> str:
        return "("


@dataclass(frozen=True)
class OrElseNode(Node):
    "`<|>`"

    expr: Optional[EXPR_TAGS]

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "OrElseNode":
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        expr = cls._get_expr(data, idx)
        return cls(lean_file, start, end, children, str(idx), expr)

    @classmethod
    def kind(cls) -> str:
        return "<|>"


@dataclass(frozen=True)
class UserCommandNode(Node):
    value: str
    name: Optional[str]
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "OtherNode":
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        value = data["ast"][idx]["value"]
        if value in ("alias", "restate_axiom"):
            name = cls._get_ident_node(children).ident
        else:
            name = None
        return cls(lean_file, start, end, children, str(idx), value, name)

    @classmethod
    def _get_ident_node(cls, children: List[Node]) -> "IdentNode":
        if isinstance(children[0], ParseNode):
            assert isinstance(children[0].children[0], IdentNode)
            return children[0].children[0]
        else:
            assert isinstance(children[0], ModifiersNode)
            assert isinstance(children[1], ParseNode)
            assert isinstance(children[1].children[0], IdentNode)
            return children[1].children[0]


@dataclass(frozen=True)
class OtherNode(Node):
    kind: str  # type: ignore

    @classmethod
    def from_data(
        cls, data: Dict[str, Any], idx: int, lean_file: LeanFile
    ) -> "OtherNode":
        start, end = parse_pos(data["ast"][idx])
        children = _parse_children(data, idx, lean_file)
        kind = data["ast"][idx]["kind"]
        return cls(lean_file, start, end, children, str(idx), kind)


NODES_WITH_EXPRS = {
    cls.kind()
    for cls_name, cls in globals().items()
    if re.fullmatch(r"[A-Z][A-Za-z]*Node", cls_name)
    and ("expr" in cls.__dataclass_fields__ or "pexpr" in cls.__dataclass_fields__)
}


def is_potential_premise_lean3(node: Node) -> bool:
    """Check if ``node`` is a theorem/definition that can be used as a premise."""
    if type(node) in (
        TheoremNode,
        DefinitionNode,
        InductiveNode,
        StructureNode,
        ClassNode,
        ClassInductiveNode,
        InstanceNode,
        AxiomNode,
        AbbreviationNode,
        ConstantNode,
    ):
        return True
    elif isinstance(node, UserCommandNode) and node.value in ("alias", "restate_axiom"):
        return True
    return False


def is_mutual_lean3(node: Node) -> bool:
    return type(node) in (DefinitionNode, InductiveNode) and node.is_mutual

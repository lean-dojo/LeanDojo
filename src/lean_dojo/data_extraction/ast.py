from lxml import etree
from pathlib import Path
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, unescape
from typing import List, Dict, Any, Optional, Callable, Tuple, Generator

from ..utils import (
    camel_case,
    is_optional_type,
    remove_optional_type,
    parse_int_list,
    parse_str_list,
)
from .lean import Pos, LeanFile


@dataclass(frozen=True)
class Node:
    lean_file: LeanFile
    start: Optional[Pos]
    end: Optional[Pos]
    children: List["Node"] = field(repr=False)

    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "Node":
        subcls = cls._kind_to_node_type(node_data["kind"])
        return subcls.from_data(node_data, lean_file)

    @classmethod
    def _kind_to_node_type(cls, kind: str) -> type:
        prefix = "Lean.Parser."
        if kind.startswith(prefix):
            kind = kind[len(prefix) :]
        cls_name = camel_case(kind.replace(".", "_")) + "Node"
        gbs = globals()
        if cls_name in gbs:
            return gbs[cls_name]  # type: ignore
        else:
            # logger.warning(kind)
            return OtherNode

    @classmethod
    def kind(cls: type) -> str:
        return cls.__name__[:-4].lower()

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

    def traverse_postorder(
        self,
        callback: Callable[["Node", List[Any]], Any],
    ) -> Any:
        return callback(
            self, [child.traverse_postorder(callback) for child in self.children]
        )

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
        start = Pos.from_str(tree.attrib["start"]) if "start" in tree.attrib else None
        end = Pos.from_str(tree.attrib["end"]) if "end" in tree.attrib else None
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
        return self.start, self.end


def _parse_pos(info: Dict[str, Any], lean_file: LeanFile) -> Pos:
    if "synthetic" in info and not info["synthetic"]["canonical"]:
        return None

    if (
        "original" in info
    ):  # | original (leading : Substring) (pos : String.Pos) (trailing : Substring) (endPos : String.Pos)
        start, end = info["original"]["pos"], info["original"]["endPos"]
    else:
        assert (
            "synthetic" in info
        )  # | synthetic (pos : String.Pos) (endPos : String.Pos) (canonical := false)
        start, end = info["synthetic"]["pos"], info["synthetic"]["endPos"]

    start = lean_file.convert_pos(start)
    end = lean_file.convert_pos(end)

    return start, end


@dataclass(frozen=True)
class AtomNode(Node):
    leading: str
    trailing: str
    val: str

    @classmethod
    def from_data(
        cls, atom_data: Dict[str, Any], lean_file: LeanFile
    ) -> Optional["AtomNode"]:
        info = atom_data["info"]
        start, end = _parse_pos(info, lean_file)

        if "original" in info:
            leading = info["original"]["leading"]
            trailing = info["original"]["trailing"]
        else:
            assert "synthetic" in info
            leading = info["synthetic"]["leading"]
            trailing = info["synthetic"]["trailing"]

        return cls(lean_file, start, end, [], leading, trailing, atom_data["val"])


@dataclass(frozen=True)
class IdentNode(Node):
    leading: str
    trailing: str
    raw_val: str
    val: str

    full_name: Optional[str] = None
    mod_name: Optional[str] = None
    def_path: Optional[str] = None
    def_start: Optional[Pos] = None
    def_end: Optional[Pos] = None

    @classmethod
    def from_data(
        cls, ident_data: Dict[str, Any], lean_file: LeanFile
    ) -> Optional["IdentNode"]:
        info = ident_data["info"]
        start, end = _parse_pos(info, lean_file)
        assert ident_data["preresolved"] == []

        if "original" in info:
            leading = info["original"]["leading"]
            trailing = info["original"]["trailing"]
        else:
            assert "synthetic" in info
            leading = info["synthetic"]["leading"]
            trailing = info["synthetic"]["trailing"]

        return cls(
            lean_file,
            start,
            end,
            [],
            leading,
            trailing,
            ident_data["rawVal"],
            ident_data["val"],
        )

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.full_name, str)


def is_leaf(node: Node) -> bool:
    return isinstance(node, AtomNode) or isinstance(node, IdentNode)


@dataclass(frozen=True)
class FileNode(Node):
    @classmethod
    def from_data(cls, data: Dict[str, Any], lean_file: LeanFile) -> "FileNode":
        children = []

        def _get_closure(node: Node, child_spans: List[Tuple[Pos, Pos]]):
            if len(child_spans) == 0:
                return node.start, node.end

            child_starts = [s for s, _ in child_spans if s is not None]
            if len(child_starts) == 0:
                start = None
            else:
                start = min(child_starts)

            child_ends = [e for _, e in child_spans if e is not None]
            if len(child_ends) == 0:
                end = None
            else:
                end = max(child_ends)

            if node.start is None:
                object.__setattr__(node, "start", start)
            else:
                assert node.start == start

            if node.end is None:
                object.__setattr__(node, "end", end)
            else:
                assert node.end == end

            return start, end

        for i, d in enumerate(data["commandASTs"]):
            node_data = d["node"]
            if i == 0:
                assert node_data["kind"] == "Lean.Parser.Module.header"
            node = Node.from_data(node_data, lean_file)
            node.traverse_postorder(_get_closure)
            children.append(node)

        return cls(lean_file, lean_file.start_pos(), lean_file.end_pos(), children)


def _parse_children(node_data: Dict[str, Any], lean_file: LeanFile) -> List[Node]:
    children = []

    for d in node_data["args"]:
        if (
            "node" in d
        ):  # | node   (info : SourceInfo) (kind : SyntaxNodeKind) (args : Array Syntax) : Syntax
            node = Node.from_data(d["node"], lean_file)
        elif "atom" in d:  # | atom   (info : SourceInfo) (val : String) : Syntax
            node = AtomNode.from_data(d["atom"], lean_file)
        elif (
            "ident" in d
        ):  # | ident  (info : SourceInfo) (rawVal : Substring) (val : Name) (preresolved : List Syntax.Preresolved) : Syntax
            node = IdentNode.from_data(d["ident"], lean_file)
        else:
            raise ValueError(d)

        if node is not None:
            children.append(node)

    return children


@dataclass(frozen=True)
class TermAttrkindNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermAttrkindNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TermAttrkindAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermAttrkindAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class IdentAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "IdentAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)

    def get_ident(self) -> str:
        return "".join(gc.val for gc in self.children if is_leaf(gc))


@dataclass(frozen=True)
class LeanElabCommandCommandIrreducibleDefNode(Node):
    name: Optional[str]
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "LeanElabCommandCommandIrreducibleDefNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        if isinstance(children[0], CommandDeclmodifiersAntiquotNode):
            name = None
        else:
            assert isinstance(children[0], CommandDeclmodifiersNode)
            assert (
                isinstance(children[1], AtomNode)
                and children[1].val == "irreducible_def"
            )
            declid_node = children[2]
            assert isinstance(declid_node, CommandDeclidNode)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class GroupNode(Node):
    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "GroupNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class MathlibTacticLemmaNode(Node):
    name: str
    full_name: Optional[str] = None
    _is_private_decl: Optional[bool] = (
        False  # `_is_private` doesn't play well with lxml.
    )

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "MathlibTacticLemmaNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode)
        assert isinstance(children[1], GroupNode)
        assert (
            isinstance(children[1].children[0], AtomNode)
            and children[1].children[0].val == "lemma"
        )
        declid_node = children[1].children[1]
        assert isinstance(declid_node, CommandDeclidNode)
        ident_node = declid_node.children[0]
        assert isinstance(ident_node, IdentNode)
        name = ident_node.val

        return cls(lean_file, start, end, children, name)

    def is_private(self) -> bool:
        return self._is_private_decl

    def get_proof_node(self) -> Node:
        decl_val_node = self.children[1].children[3]
        if type(decl_val_node) in (
            CommandDeclvalsimpleNode,
            CommandWherestructinstNode,
        ):
            return decl_val_node.children[1]
        else:
            return decl_val_node

    def has_tactic_proof(self) -> bool:
        node = self.get_proof_node()
        return isinstance(node, TermBytacticNode)

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.name, str)


@dataclass(frozen=True)
class LemmaNode(Node):
    name: str
    full_name: Optional[str] = None
    _is_private_decl: Optional[bool] = (
        False  # `_is_private` doesn't play well with lxml.
    )

    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "LemmaNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode)
        assert isinstance(children[1], GroupNode)
        assert (
            isinstance(children[1].children[0], AtomNode)
            and children[1].children[0].val == "lemma"
        )
        declid_node = children[1].children[1]
        assert isinstance(declid_node, CommandDeclidNode)
        ident_node = declid_node.children[0]
        assert isinstance(ident_node, IdentNode)
        name = ident_node.val

        return cls(lean_file, start, end, children, name)

    def is_private(self) -> bool:
        return self._is_private_decl

    def get_proof_node(self) -> Node:
        decl_val_node = self.children[1].children[3]
        if type(decl_val_node) in (
            CommandDeclvalsimpleNode,
            CommandWherestructinstNode,
        ):
            return decl_val_node.children[1]
        else:
            return decl_val_node

    def has_tactic_proof(self) -> bool:
        node = self.get_proof_node()
        return isinstance(node, TermBytacticNode)

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.name, str)


@dataclass(frozen=True)
class CommandDeclarationNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclarationNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        if isinstance(children[0], CommandDeclmodifiersAntiquotNode):
            name = None
        else:
            assert isinstance(children[0], CommandDeclmodifiersNode)
            assert type(children[1]) in (
                CommandDefNode,
                CommandTheoremNode,
                CommandInductiveNode,
                CommandClassinductiveNode,
                CommandStructureNode,
                CommandInstanceNode,
                CommandAbbrevNode,
                CommandOpaqueNode,
                CommandAxiomNode,
                CommandExampleNode,
            )
            name = children[1].name

            if children[0].is_private():
                for child in children:
                    if isinstance(child, CommandTheoremNode):
                        object.__setattr__(child, "_is_private_decl", True)

        return cls(lean_file, start, end, children, name)

    @property
    def is_theorem(self) -> bool:
        return isinstance(self.children[1], CommandTheoremNode)

    def get_theorem_node(self) -> "CommandTheoremNode":
        assert self.is_theorem
        return self.children[1]

    @property
    def is_example(self) -> bool:
        return isinstance(self.children[1], CommandExampleNode)


@dataclass(frozen=True)
class CommandDeclmodifiersAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclmodifiersAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclmodifiersNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclmodifiersNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)

    def is_private(self) -> bool:
        result = False

        def _callback(node: CommandPrivateNode, _) -> bool:
            nonlocal result
            result = True
            return True

        self.traverse_preorder(_callback, CommandPrivateNode)
        return result


@dataclass(frozen=True)
class CommandPrivateNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandPrivateNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandOpenNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandOpenNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandOpenonlyNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandOpenonlyNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class NullNode(Node):
    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "NullNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandStructuretkNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandStructuretkNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert isinstance(children[0], AtomNode) and children[0].val == "structure"
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandClasstkNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandClasstkNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert isinstance(children[0], AtomNode) and children[0].val == "class"
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandStructureNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandStructureNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert type(children[0]) in (CommandStructuretkNode, CommandClasstkNode)
        if isinstance(children[1], CommandDeclidAntiquotNode):
            name = None
        else:
            assert isinstance(children[1], CommandDeclidNode)
            decl_id_node = children[1]
            ident_node = decl_id_node.children[0]

            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandInductiveNode(Node):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandInductiveNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode) and children[0].val == "inductive"

        if isinstance(children[1], CommandDeclidAntiquotNode):
            name = None
        else:
            assert isinstance(children[1], CommandDeclidNode)
            decl_id_node = children[1]
            ident_node = decl_id_node.children[0]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandClassinductiveNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandClassinductiveNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert (
            isinstance(children[0].children[0], AtomNode)
            and children[0].children[0].val == "class"
        )
        assert (
            isinstance(children[0].children[1], AtomNode)
            and children[0].children[1].val == "inductive"
        )

        if isinstance(children[1], CommandDeclidAntiquotNode):
            name = None
        else:
            assert isinstance(children[1], CommandDeclidNode)
            decl_id_node = children[1]
            ident_node = decl_id_node.children[0]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class TermHoleNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermHoleNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and type(children[0]) in (
            AtomNode,
            TokenAntiquotNode,
        )
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class LeanBinderidentNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "LeanBinderidentNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and type(children[0]) in (
            TermHoleNode,
            IdentNode,
            IdentAntiquotNode,
        )
        return cls(lean_file, start, end, children)

    def get_ident(self) -> Optional[str]:
        if isinstance(self.children[0], TermHoleNode):
            return None
        else:
            assert isinstance(self.children[0], IdentNode)
            return self.children[0].val


@dataclass(frozen=True)
class LeanBinderidentAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "LeanBinderidentAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)

    def get_ident(self) -> Optional[str]:
        return None


@dataclass(frozen=True)
class StdTacticAliasAliasNode(Node):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "StdTacticAliasAliasNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode)
        assert isinstance(children[1], AtomNode) and children[1].val == "alias"
        if isinstance(children[2], IdentAntiquotNode):
            name = None
        else:
            ident_node = children[2]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val
        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class StdTacticAliasAliaslrNode(Node):
    name: List[str]
    full_name: Optional[List[str]] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "StdTacticAliasAliaslrNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode)
        assert isinstance(children[1], AtomNode) and children[1].val == "alias"
        assert isinstance(children[2], AtomNode) and children[2].val == "⟨"
        assert isinstance(children[4], AtomNode) and children[4].val == ","
        assert isinstance(children[6], AtomNode) and children[6].val == "⟩"

        name = []
        assert type(children[3]) in (LeanBinderidentNode, LeanBinderidentAntiquotNode)
        name.append(children[3].get_ident())
        assert type(children[5]) in (LeanBinderidentNode, LeanBinderidentAntiquotNode)
        name.append(children[5].get_ident())
        name = [n for n in name if n is not None]

        return cls(lean_file, start, end, children, name)

    @property
    def is_mutual(self) -> bool:
        return True


@dataclass(frozen=True)
class CommandAbbrevNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandAbbrevNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode) and children[0].val == "abbrev"
        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode):
            name = None
        else:
            assert isinstance(declid_node, CommandDeclidNode)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandOpaqueNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandOpaqueNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode) and children[0].val == "opaque"
        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode):
            name = None
        else:
            assert isinstance(declid_node, CommandDeclidNode)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandAxiomNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandAxiomNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode) and children[0].val == "axiom"
        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode):
            name = None
        else:
            assert isinstance(declid_node, CommandDeclidNode)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandExampleNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandExampleNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert isinstance(children[0], AtomNode) and children[0].val == "example"
        name = None
        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandInstanceNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandInstanceNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        if isinstance(children[0], TermAttrkindAntiquotNode):
            name = None
        else:
            assert isinstance(children[0], TermAttrkindNode)
            assert isinstance(children[1], AtomNode) and children[1].val == "instance"
            if children[3].children != []:
                declid_node = children[3].children[0]
                if isinstance(declid_node, CommandDeclidNode):
                    ident_node = declid_node.children[0]
                    assert isinstance(ident_node, IdentNode)
                    name = ident_node.val
                else:
                    name = None
            else:
                name = None
        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandDefNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDefNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        if isinstance(children[0], TokenAntiquotNode) or isinstance(
            children[1], CommandDeclidAntiquotNode
        ):
            name = None
        else:
            assert isinstance(children[0], AtomNode) and children[0].val == "def"
            assert isinstance(children[1], CommandDeclidNode)
            decl_id_node = children[1]
            ident_node = decl_id_node.children[0]

            if isinstance(ident_node, IdentNode):
                name = ident_node.val
            else:
                assert isinstance(ident_node, IdentAntiquotNode)
                name = ident_node.get_ident()

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandDeclidAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclidAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclidNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclidNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclvalsimpleNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclvalsimpleNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TokenAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TokenAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclvaleqnsNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclvaleqnsNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandWherestructinstNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandWherestructinstNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclsigNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclsigNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TermExplicitbinderNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermExplicitbinderNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TermTypespecNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermTypespecNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandTheoremNode(Node):
    name: str
    full_name: Optional[str] = None
    _is_private_decl: Optional[bool] = (
        False  # `_is_private` doesn't play well with lxml.
    )

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandTheoremNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode) and children[0].val == "theorem"

        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode):
            name = None
        else:
            ident_node = declid_node.children[0]
            if isinstance(ident_node, IdentNode):
                name = ident_node.val
            else:
                assert isinstance(ident_node, IdentAntiquotNode)
                name = ident_node.get_ident()

        if not isinstance(children[1], CommandDeclidAntiquotNode):
            assert isinstance(children[2], CommandDeclsigNode)
            decl_val_node = children[3]
            assert type(decl_val_node) in (
                CommandDeclvalsimpleNode,
                CommandDeclvaleqnsNode,
                CommandWherestructinstNode,
            )

            if isinstance(decl_val_node, CommandDeclvalsimpleNode):
                assert (
                    isinstance(decl_val_node.children[0], AtomNode)
                    and decl_val_node.children[0].val == ":="
                )
            elif isinstance(decl_val_node, CommandWherestructinstNode):
                assert (
                    isinstance(decl_val_node.children[0], AtomNode)
                    and decl_val_node.children[0].val == "where"
                )

        return cls(lean_file, start, end, children, name)

    def is_private(self) -> bool:
        return self._is_private_decl

    def get_proof_node(self) -> Node:
        decl_val_node = self.children[3]
        if type(decl_val_node) in (
            CommandDeclvalsimpleNode,
            CommandWherestructinstNode,
        ):
            return decl_val_node.children[1]
        else:
            return decl_val_node

    def has_tactic_proof(self) -> bool:
        node = self.get_proof_node()
        return isinstance(node, TermBytacticNode)

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.name, str)


@dataclass(frozen=True)
class TermBytacticNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermBytacticNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TacticTacticseq1IndentedAntiquotNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseq1IndentedAntiquotNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)

    def get_tactic_nodes(
        self, atomic_only: bool = False
    ) -> Generator[Node, None, None]:
        return


@dataclass(frozen=True)
class TacticTacticseqNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseqNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and type(children[0]) in (
            TacticTacticseq1IndentedNode,
            TacticTacticseqbracketedNode,
            TacticTacticseq1IndentedAntiquotNode,
        )
        return cls(lean_file, start, end, children)

    def get_tactic_nodes(
        self, atomic_only: bool = False
    ) -> Generator[Node, None, None]:
        yield from self.children[0].get_tactic_nodes(atomic_only)


@dataclass(frozen=True)
class TacticTacticseq1IndentedNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseq1IndentedNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and isinstance(children[0], NullNode)
        return cls(lean_file, start, end, children)

    def get_tactic_nodes(
        self, atomic_only: bool = False
    ) -> Generator[Node, None, None]:
        for i, tac_node in enumerate(self.children[0].children):
            if i % 2 == 0:
                if not atomic_only or not contains_tactic(tac_node):
                    yield tac_node
            else:
                assert isinstance(tac_node, NullNode) or isinstance(tac_node, AtomNode)


@dataclass(frozen=True)
class TacticTacticseqbracketedNode(Node):
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    tactic: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseqbracketedNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 3
        return cls(lean_file, start, end, children)

    @property
    def tactic_nodes(self) -> List[Node]:
        children = self.children
        if not isinstance(children[0], AtomNode) or children[0].val != "{":
            return []

        assert isinstance(children[1], NullNode)
        assert isinstance(children[2], AtomNode) and children[2].val == "}"
        nodes = []
        for i, tac_node in enumerate(children[1].children):
            if i % 2 == 0:
                nodes.append(tac_node)
            else:
                assert isinstance(tac_node, NullNode) or isinstance(tac_node, AtomNode)
        return nodes

    def get_tactic_nodes(
        self, atomic_only: bool = False
    ) -> Generator[Node, None, None]:
        children = self.children
        if isinstance(children[0], AtomNode) and children[0].val == "{":
            assert isinstance(children[1], NullNode)
            assert isinstance(children[2], AtomNode) and children[2].val == "}"
            for i, tac_node in enumerate(children[1].children):
                if i % 2 == 0:
                    if not atomic_only or not contains_tactic(tac_node):
                        yield tac_node
                else:
                    assert isinstance(tac_node, NullNode) or isinstance(
                        tac_node, AtomNode
                    )


def contains_tactic(node: Node) -> bool:
    result = False

    def _callback(x, _) -> bool:
        if x is not node and type(x) in (
            TacticTacticseq1IndentedNode,
            TacticTacticseqbracketedNode,
        ):
            nonlocal result
            result = True
            return True

    node.traverse_preorder(_callback, node_cls=None)
    return result


@dataclass(frozen=True)
class ModuleHeaderNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModuleHeaderNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class ModulePreludeNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModulePreludeNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class ModulePreludeNode(Node):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModulePreludeNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class ModuleImportNode(Node):
    module: Optional[str]
    path: Optional[Path] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModuleImportNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode) and children[0].val == "import"
        if isinstance(children[2], IdentNode):
            module = children[2].val
        else:
            module = None

        return cls(lean_file, start, end, children, module)


@dataclass(frozen=True)
class CommandModuledocNode(Node):
    comment: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandModuledocNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 2 and all(isinstance(_, AtomNode) for _ in children)
        assert children[0].val == "/-!"
        comment = children[1].val
        return cls(lean_file, start, end, children, comment)


@dataclass(frozen=True)
class CommandDoccommentNode(Node):
    comment: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDoccommentNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 2 and all(isinstance(_, AtomNode) for _ in children)
        assert children[0].val == "/--"
        comment = children[1].val
        return cls(lean_file, start, end, children, comment)


@dataclass(frozen=True)
class CommandNamespaceNode(Node):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandNamespaceNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 2
        assert isinstance(children[0], AtomNode) and children[0].val == "namespace"
        if isinstance(children[1], IdentNode):
            name = children[1].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandSectionNode(Node):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandNamespaceNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 2
        assert isinstance(children[0], AtomNode) and children[0].val == "section"
        assert isinstance(children[1], NullNode)

        if len(children[1].children) == 1 and isinstance(
            children[1].children[0], IdentNode
        ):
            name = children[1].children[0].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandNoncomputablesectionNode(Node):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandNoncomputablesectionNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 3
        assert isinstance(children[0], AtomNode) and children[0].val == "noncomputable"
        assert isinstance(children[1], AtomNode) and children[1].val == "section"
        assert isinstance(children[2], NullNode)

        if len(children[2].children) == 1 and isinstance(
            children[2].children[0], IdentNode
        ):
            name = children[2].children[0].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandEndNode(Node):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandEndNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 2
        assert isinstance(children[0], AtomNode) and children[0].val == "end"
        assert isinstance(children[1], NullNode)

        if len(children[1].children) == 1 and isinstance(
            children[1].children[0], IdentNode
        ):
            name = children[1].children[0].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class OtherNode(Node):
    kind: str  # type: ignore
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    tactic: Optional[str] = None

    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "OtherNode":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children, node_data["kind"])


def is_potential_premise_lean4(node: Node) -> bool:
    """Check if ``node`` is a theorem/definition that can be used as a premise."""
    if (isinstance(node, CommandDeclarationNode) and not node.is_example) or type(
        node
    ) in (
        LemmaNode,
        MathlibTacticLemmaNode,
        LeanElabCommandCommandIrreducibleDefNode,
        StdTacticAliasAliasNode,
        StdTacticAliasAliaslrNode,
    ):
        return node.name is not None
    else:
        return False


def is_mutual_lean4(node: Node) -> bool:
    return (
        type(node) in (IdentNode, CommandTheoremNode, StdTacticAliasAliaslrNode)
        and node.is_mutual
    )

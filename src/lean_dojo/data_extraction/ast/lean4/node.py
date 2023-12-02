from lxml import etree
from pathlib import Path
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, unescape
from typing import List, Dict, Any, Optional, Callable, Tuple, Generator

from ....utils import (
    camel_case,
    is_optional_type,
    remove_optional_type,
    parse_int_list,
    parse_str_list,
)
from ...lean import Pos, LeanFile


@dataclass(frozen=True)
class Node4:
    lean_file: LeanFile
    start: Optional[Pos]
    end: Optional[Pos]
    children: List["Node4"] = field(repr=False)

    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "Node4":
        subcls = cls._kind_to_node_type(node_data["kind"])
        return subcls.from_data(node_data, lean_file)

    @classmethod
    def _kind_to_node_type(cls, kind: str) -> type:
        prefix = "Lean.Parser."
        if kind.startswith(prefix):
            kind = kind[len(prefix) :]
        cls_name = camel_case(kind.replace(".", "_")) + "Node4"
        gbs = globals()
        if cls_name in gbs:
            return gbs[cls_name]  # type: ignore
        else:
            # logger.warning(kind)
            return OtherNode4

    @classmethod
    def kind(cls: type) -> str:
        return cls.__name__[:-5].lower()

    def traverse_preorder(
        self,
        callback: Callable[["Node4", List["Node4"]], Any],
        node_cls: Optional[type],
        parents: List["Node4"] = [],
    ) -> None:
        if node_cls is None or isinstance(self, node_cls):
            if callback(self, parents):
                return
        for child in self.children:
            child.traverse_preorder(callback, node_cls, parents + [self])

    def traverse_postorder(
        self,
        callback: Callable[["Node4", List[Any]], Any],
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
    def from_xml(cls, tree: etree.Element, lean_file: LeanFile) -> "Node4":
        subcls = globals()[tree.tag]
        start = Pos.from_str(tree.attrib["start"]) if "start" in tree.attrib else None
        end = Pos.from_str(tree.attrib["end"]) if "end" in tree.attrib else None
        children = [Node4.from_xml(subtree, lean_file) for subtree in tree]
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
class AtomNode4(Node4):
    leading: str
    trailing: str
    val: str

    @classmethod
    def from_data(
        cls, atom_data: Dict[str, Any], lean_file: LeanFile
    ) -> Optional["AtomNode4"]:
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
class IdentNode4(Node4):
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
    ) -> Optional["IdentNode4"]:
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


def is_leaf(node: Node4) -> bool:
    return isinstance(node, AtomNode4) or isinstance(node, IdentNode4)


@dataclass(frozen=True)
class FileNode4(Node4):
    @classmethod
    def from_data(cls, data: Dict[str, Any], lean_file: LeanFile) -> "FileNode4":
        children = []

        def _get_closure(node: Node4, child_spans: List[Tuple[Pos, Pos]]):
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
            node = Node4.from_data(node_data, lean_file)
            node.traverse_postorder(_get_closure)
            children.append(node)

        return cls(lean_file, lean_file.start_pos(), lean_file.end_pos(), children)


def _parse_children(node_data: Dict[str, Any], lean_file: LeanFile) -> List[Node4]:
    children = []

    for d in node_data["args"]:
        if (
            "node" in d
        ):  # | node   (info : SourceInfo) (kind : SyntaxNodeKind) (args : Array Syntax) : Syntax
            node = Node4.from_data(d["node"], lean_file)
        elif "atom" in d:  # | atom   (info : SourceInfo) (val : String) : Syntax
            node = AtomNode4.from_data(d["atom"], lean_file)
        elif (
            "ident" in d
        ):  # | ident  (info : SourceInfo) (rawVal : Substring) (val : Name) (preresolved : List Syntax.Preresolved) : Syntax
            node = IdentNode4.from_data(d["ident"], lean_file)
        else:
            raise ValueError(d)

        if node is not None:
            children.append(node)

    return children


@dataclass(frozen=True)
class TermAttrkindNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermAttrkindNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TermAttrkindAntiquotNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermAttrkindAntiquotNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class IdentAntiquotNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "IdentAntiquotNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)

    def get_ident(self) -> str:
        return "".join(gc.val for gc in self.children if is_leaf(gc))


@dataclass(frozen=True)
class LeanElabCommandCommandIrreducibleDefNode4(Node4):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "LeanElabCommandCommandIrreducibleDefNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode4)
        assert (
            isinstance(children[1], AtomNode4) and children[1].val == "irreducible_def"
        )
        declid_node = children[2]
        assert isinstance(declid_node, CommandDeclidNode4)
        ident_node = declid_node.children[0]
        assert isinstance(ident_node, IdentNode4)
        name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class GroupNode4(Node4):
    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "GroupNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class MathlibTacticLemmaNode4(Node4):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "MathlibTacticLemmaNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode4)
        assert isinstance(children[1], GroupNode4)
        assert (
            isinstance(children[1].children[0], AtomNode4)
            and children[1].children[0].val == "lemma"
        )
        declid_node = children[1].children[1]
        assert isinstance(declid_node, CommandDeclidNode4)
        ident_node = declid_node.children[0]
        assert isinstance(ident_node, IdentNode4)
        name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandDeclarationNode4(Node4):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclarationNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        if isinstance(children[0], CommandDeclmodifiersAntiquotNode4):
            name = None
        else:
            assert isinstance(children[0], CommandDeclmodifiersNode4)
            assert type(children[1]) in (
                CommandDefNode4,
                CommandTheoremNode4,
                CommandInductiveNode4,
                CommandClassinductiveNode4,
                CommandStructureNode4,
                CommandInstanceNode4,
                CommandAbbrevNode4,
                CommandOpaqueNode4,
                CommandAxiomNode4,
                CommandExampleNode4,
            )
            name = children[1].name

            if children[0].is_private():
                for child in children:
                    if isinstance(child, CommandTheoremNode4):
                        object.__setattr__(child, "_is_private_decl", True)

        return cls(lean_file, start, end, children, name)

    @property
    def is_theorem(self) -> bool:
        return isinstance(self.children[1], CommandTheoremNode4)

    def get_theorem_node(self) -> "CommandTheoremNode4":
        assert self.is_theorem
        return self.children[1]

    @property
    def is_example(self) -> bool:
        return isinstance(self.children[1], CommandExampleNode4)


@dataclass(frozen=True)
class CommandDeclmodifiersAntiquotNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclmodifiersAntiquotNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclmodifiersNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclmodifiersNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)

    def is_private(self) -> bool:
        result = False

        def _callback(node: CommandPrivateNode4, _) -> bool:
            nonlocal result
            result = True
            return True

        self.traverse_preorder(_callback, CommandPrivateNode4)
        return result


@dataclass(frozen=True)
class CommandPrivateNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandPrivateNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandOpenNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandOpenNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandOpenonlyNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandOpenonlyNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class NullNode4(Node4):
    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "NullNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandStructuretkNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandStructuretkNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert isinstance(children[0], AtomNode4) and children[0].val == "structure"
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandClasstkNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandClasstkNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert isinstance(children[0], AtomNode4) and children[0].val == "class"
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandStructureNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandStructureNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert type(children[0]) in (CommandStructuretkNode4, CommandClasstkNode4)
        if isinstance(children[1], CommandDeclidAntiquotNode4):
            name = None
        else:
            assert isinstance(children[1], CommandDeclidNode4)
            decl_id_node = children[1]
            ident_node = decl_id_node.children[0]

            assert isinstance(ident_node, IdentNode4)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandInductiveNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandInductiveNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode4) and children[0].val == "inductive"
        assert isinstance(children[1], CommandDeclidNode4)
        decl_id_node = children[1]
        ident_node = decl_id_node.children[0]
        assert isinstance(ident_node, IdentNode4)
        name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandClassinductiveNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandClassinductiveNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert (
            isinstance(children[0].children[0], AtomNode4)
            and children[0].children[0].val == "class"
        )
        assert (
            isinstance(children[0].children[1], AtomNode4)
            and children[0].children[1].val == "inductive"
        )
        assert isinstance(children[1], CommandDeclidNode4)
        decl_id_node = children[1]
        ident_node = decl_id_node.children[0]
        assert isinstance(ident_node, IdentNode4)
        name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class TermHoleNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermHoleNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and type(children[0]) in (
            AtomNode4,
            TokenAntiquotNode4,
        )
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class LeanBinderidentNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "LeanBinderidentNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and type(children[0]) in (
            TermHoleNode4,
            IdentNode4,
            IdentAntiquotNode4,
        )
        return cls(lean_file, start, end, children)

    def get_ident(self) -> Optional[str]:
        if isinstance(self.children[0], TermHoleNode4):
            return None
        else:
            assert isinstance(self.children[0], IdentNode4)
            return self.children[0].val


@dataclass(frozen=True)
class StdTacticAliasAliasNode4(Node4):
    name: str
    full_name: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "StdTacticAliasAliasNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode4)
        assert isinstance(children[1], AtomNode4) and children[1].val == "alias"
        ident_node = children[2]
        assert isinstance(ident_node, IdentNode4)
        name = ident_node.val
        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class StdTacticAliasAliaslrNode4(Node4):
    name: List[str]
    full_name: Optional[List[str]] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "StdTacticAliasAliaslrNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], CommandDeclmodifiersNode4)
        assert isinstance(children[1], AtomNode4) and children[1].val == "alias"
        assert isinstance(children[2], AtomNode4) and children[2].val == "⟨"
        assert isinstance(children[4], AtomNode4) and children[4].val == ","
        assert isinstance(children[6], AtomNode4) and children[6].val == "⟩"

        name = []
        assert isinstance(children[3], LeanBinderidentNode4)
        name.append(children[3].get_ident())
        assert isinstance(children[5], LeanBinderidentNode4)
        name.append(children[5].get_ident())
        name = [n for n in name if n is not None]

        return cls(lean_file, start, end, children, name)

    @property
    def is_mutual(self) -> bool:
        return True


@dataclass(frozen=True)
class CommandAbbrevNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandAbbrevNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode4) and children[0].val == "abbrev"
        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode4):
            name = None
        else:
            assert isinstance(declid_node, CommandDeclidNode4)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode4)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandOpaqueNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandOpaqueNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode4) and children[0].val == "opaque"
        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode4):
            name = None
        else:
            assert isinstance(declid_node, CommandDeclidNode4)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode4)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandAxiomNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandAxiomNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode4) and children[0].val == "axiom"
        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode4):
            name = None
        else:
            assert isinstance(declid_node, CommandDeclidNode4)
            ident_node = declid_node.children[0]
            assert isinstance(ident_node, IdentNode4)
            name = ident_node.val

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandExampleNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandExampleNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert isinstance(children[0], AtomNode4) and children[0].val == "example"
        name = None
        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandInstanceNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandInstanceNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        if isinstance(children[0], TermAttrkindAntiquotNode4):
            name = None
        else:
            assert isinstance(children[0], TermAttrkindNode4)
            assert isinstance(children[1], AtomNode4) and children[1].val == "instance"
            if children[3].children != []:
                declid_node = children[3].children[0]
                if isinstance(declid_node, CommandDeclidNode4):
                    ident_node = declid_node.children[0]
                    assert isinstance(ident_node, IdentNode4)
                    name = ident_node.val
                else:
                    name = None
            else:
                name = None
        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandDefNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDefNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        if isinstance(children[0], TokenAntiquotNode4) or isinstance(
            children[1], CommandDeclidAntiquotNode4
        ):
            name = None
        else:
            assert isinstance(children[0], AtomNode4) and children[0].val == "def"
            assert isinstance(children[1], CommandDeclidNode4)
            decl_id_node = children[1]
            ident_node = decl_id_node.children[0]

            if isinstance(ident_node, IdentNode4):
                name = ident_node.val
            else:
                assert isinstance(ident_node, IdentAntiquotNode4)
                name = ident_node.get_ident()

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandDeclidAntiquotNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclidAntiquotNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclidNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclidNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclvalsimpleNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclvalsimpleNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TokenAntiquotNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TokenAntiquotNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclvaleqnsNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclvaleqnsNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandWherestructinstNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandWherestructinstNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandDeclsigNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDeclsigNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TermExplicitbinderNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermExplicitbinderNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TermTypespecNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermTypespecNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class CommandTheoremNode4(Node4):
    name: str
    full_name: Optional[str] = None
    _is_private_decl: Optional[
        bool
    ] = False  # `_is_private` doesn't play well with lxml.

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandTheoremNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode4) and children[0].val == "theorem"

        declid_node = children[1]
        if isinstance(declid_node, CommandDeclidAntiquotNode4):
            name = None
        else:
            ident_node = declid_node.children[0]
            if isinstance(ident_node, IdentNode4):
                name = ident_node.val
            else:
                assert isinstance(ident_node, IdentAntiquotNode4)
                name = ident_node.get_ident()

        assert isinstance(children[2], CommandDeclsigNode4)
        decl_val_node = children[3]
        assert type(decl_val_node) in (
            CommandDeclvalsimpleNode4,
            CommandDeclvaleqnsNode4,
            CommandWherestructinstNode4,
        )

        if isinstance(decl_val_node, CommandDeclvalsimpleNode4):
            assert (
                isinstance(decl_val_node.children[0], AtomNode4)
                and decl_val_node.children[0].val == ":="
            )
            assert isinstance(decl_val_node.children[2], NullNode4)
        elif isinstance(decl_val_node, CommandWherestructinstNode4):
            assert (
                isinstance(decl_val_node.children[0], AtomNode4)
                and decl_val_node.children[0].val == "where"
            )
            assert isinstance(decl_val_node.children[2], NullNode4)

        return cls(lean_file, start, end, children, name)

    def is_private(self) -> bool:
        return self._is_private_decl

    def get_proof_node(self) -> Node4:
        decl_val_node = self.children[3]
        if type(decl_val_node) in (
            CommandDeclvalsimpleNode4,
            CommandWherestructinstNode4,
        ):
            return decl_val_node.children[1]
        else:
            return decl_val_node

    def has_tactic_proof(self) -> bool:
        node = self.get_proof_node()
        return isinstance(node, TermBytacticNode4)

    @property
    def is_mutual(self) -> bool:
        return not isinstance(self.name, str)


@dataclass(frozen=True)
class TermBytacticNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TermBytacticNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class TacticTacticseqNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseqNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and type(children[0]) in (
            TacticTacticseq1IndentedNode4,
            TacticTacticseqbracketedNode4,
        )
        return cls(lean_file, start, end, children)

    def get_tactic_nodes(self) -> Generator[Node4, None, None]:
        yield from self.children[0].get_tactic_nodes()


@dataclass(frozen=True)
class TacticTacticseq1IndentedNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseq1IndentedNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 1 and isinstance(children[0], NullNode4)
        return cls(lean_file, start, end, children)

    def get_tactic_nodes(self) -> Generator[Node4, None, None]:
        for i, tac_node in enumerate(self.children[0].children):
            if i % 2 == 0:
                yield tac_node
            else:
                assert isinstance(tac_node, NullNode4) or isinstance(
                    tac_node, AtomNode4
                )


@dataclass(frozen=True)
class TacticTacticseqbracketedNode4(Node4):
    tactic_nodes: List[Node4]
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    tactic: Optional[str] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "TacticTacticseqbracketedNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 3
        if isinstance(children[0], AtomNode4) and children[0].val == "{":
            assert isinstance(children[1], NullNode4)
            assert isinstance(children[2], AtomNode4) and children[2].val == "}"
            tactic_nodes = []
            for i, tac_node in enumerate(children[1].children):
                if i % 2 == 0:
                    tactic_nodes.append(tac_node)
                else:
                    assert isinstance(tac_node, NullNode4) or isinstance(
                        tac_node, AtomNode4
                    )
        else:
            tactic_nodes = []

        return cls(lean_file, start, end, children, tactic_nodes)

    def get_tactic_nodes(self) -> Generator[Node4, None, None]:
        yield from self.tactic_nodes


@dataclass(frozen=True)
class ModuleHeaderNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModuleHeaderNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class ModulePreludeNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModulePreludeNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class ModulePreludeNode4(Node4):
    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModulePreludeNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children)


@dataclass(frozen=True)
class ModuleImportNode4(Node4):
    module: Optional[str]
    path: Optional[Path] = None

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "ModuleImportNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert isinstance(children[0], AtomNode4) and children[0].val == "import"
        if isinstance(children[2], IdentNode4):
            module = children[2].val
        else:
            module = None

        return cls(lean_file, start, end, children, module)


@dataclass(frozen=True)
class CommandModuledocNode4(Node4):
    comment: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandModuledocNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 2 and all(isinstance(_, AtomNode4) for _ in children)
        assert children[0].val == "/-!"
        comment = children[1].val
        return cls(lean_file, start, end, children, comment)


@dataclass(frozen=True)
class CommandDoccommentNode4(Node4):
    comment: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandDoccommentNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        assert len(children) == 2 and all(isinstance(_, AtomNode4) for _ in children)
        assert children[0].val == "/--"
        comment = children[1].val
        return cls(lean_file, start, end, children, comment)


@dataclass(frozen=True)
class CommandNamespaceNode4(Node4):
    name: str

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandNamespaceNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 2
        assert isinstance(children[0], AtomNode4) and children[0].val == "namespace"
        if isinstance(children[1], IdentNode4):
            name = children[1].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandSectionNode4(Node4):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandNamespaceNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 2
        assert isinstance(children[0], AtomNode4) and children[0].val == "section"
        assert isinstance(children[1], NullNode4)

        if len(children[1].children) == 1 and isinstance(
            children[1].children[0], IdentNode4
        ):
            name = children[1].children[0].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandNoncomputablesectionNode4(Node4):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandNoncomputablesectionNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 3
        assert isinstance(children[0], AtomNode4) and children[0].val == "noncomputable"
        assert isinstance(children[1], AtomNode4) and children[1].val == "section"
        assert isinstance(children[2], NullNode4)

        if len(children[2].children) == 1 and isinstance(
            children[2].children[0], IdentNode4
        ):
            name = children[2].children[0].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class CommandEndNode4(Node4):
    name: Optional[str]

    @classmethod
    def from_data(
        cls, node_data: Dict[str, Any], lean_file: LeanFile
    ) -> "CommandEndNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)

        assert len(children) == 2
        assert isinstance(children[0], AtomNode4) and children[0].val == "end"
        assert isinstance(children[1], NullNode4)

        if len(children[1].children) == 1 and isinstance(
            children[1].children[0], IdentNode4
        ):
            name = children[1].children[0].val
        else:
            name = None

        return cls(lean_file, start, end, children, name)


@dataclass(frozen=True)
class OtherNode4(Node4):
    kind: str  # type: ignore
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    tactic: Optional[str] = None

    @classmethod
    def from_data(cls, node_data: Dict[str, Any], lean_file: LeanFile) -> "OtherNode4":
        assert node_data["info"] == "none"
        start, end = None, None
        children = _parse_children(node_data, lean_file)
        return cls(lean_file, start, end, children, node_data["kind"])


def is_potential_premise_lean4(node: Node4) -> bool:
    """Check if ``node`` is a theorem/definition that can be used as a premise."""
    if (isinstance(node, CommandDeclarationNode4) and not node.is_example) or type(
        node
    ) in (
        MathlibTacticLemmaNode4,
        LeanElabCommandCommandIrreducibleDefNode4,
        StdTacticAliasAliasNode4,
        StdTacticAliasAliaslrNode4,
    ):
        return node.name is not None
    else:
        return False


def is_mutual_lean4(node: Node4) -> bool:
    return (
        type(node) in (IdentNode4, CommandTheoremNode4, StdTacticAliasAliaslrNode4)
        and node.is_mutual
    )

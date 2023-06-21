import sys
from pygments import highlight
from pygments.lexers import LeanLexer
from pygments.formatters import LatexFormatter


def main() -> None:
    path = sys.argv[1]
    assert path.endswith(".lean")
    code = open(path, "r").read()

    formatter = LatexFormatter(style="manni", full=True, linenos=False)
    with open(path.replace(".lean", ".tex"), "wt") as oup:
        highlight(code, LeanLexer(), formatter, outfile=oup)


if __name__ == "__main__":
    main()

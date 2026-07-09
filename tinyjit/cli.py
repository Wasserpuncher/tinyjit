"""Command line interface: ``run`` a program, or ``dump`` its machine code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import syntax
from .compiler import CompileError, compile_program
from .executor import Program, UnsupportedPlatform
from .interpreter import Interpreter
from .jit import Module


def _read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run(path: str, entry: str, args: list[int], use_interpreter: bool) -> int:
    source = _read_source(path)

    if use_interpreter:
        result = Interpreter(syntax.parse(source)).call(entry, args)
    else:
        module = Module(source)
        if entry not in module:
            raise CompileError(f"no function named {entry!r}; have {module.names()}")
        result = module[entry](*args)

    print(result)
    return 0


def dump(path: str) -> int:
    """Show the compiled machine code, grouped by function, as a hex dump."""
    functions = syntax.parse(_read_source(path))
    code, table = compile_program(functions)

    # Order the functions by their offset so the dump reads top to bottom.
    ordered = sorted(table.items(), key=lambda item: item[1][0])
    boundaries = [offset for _, (offset, _) in ordered] + [len(code)]

    for index, (name, (offset, arity)) in enumerate(ordered):
        end = boundaries[index + 1]
        print(f"; fn {name}({arity} arg{'s' if arity != 1 else ''})  "
              f"offset={offset} size={end - offset}")
        _hex_dump(code[offset:end], offset)
        print()

    print(f"; total {len(code)} bytes across {len(ordered)} function(s)")
    return 0


def _hex_dump(data: bytes, base: int) -> None:
    for row in range(0, len(data), 16):
        chunk = data[row : row + 16]
        hex_bytes = " ".join(f"{byte:02x}" for byte in chunk)
        print(f"  {base + row:08x}  {hex_bytes}")


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyjit run", description="compile a program and call one of its functions"
    )
    parser.add_argument("file")
    parser.add_argument("-e", "--entry", default="main", help="function to call (default: main)")
    parser.add_argument(
        "--interp", action="store_true",
        help="evaluate with the reference interpreter instead of native code",
    )
    parser.add_argument("args", nargs="*", type=int, help="integer arguments")
    return parser


def _dump_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyjit dump", description="show the generated machine code")
    parser.add_argument("file")
    return parser


_USAGE = "usage: tinyjit {run,dump} ...\n  run FILE [-e ENTRY] [--interp] [ARGS...]\n  dump FILE"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0 if argv else 2

    command, rest = argv[0], argv[1:]

    try:
        if command == "run":
            # parse_intermixed_args reliably matches a variable-length positional
            # (the integer arguments) even when it follows the -e/--interp
            # options. Plain parse_args does not, on Python < 3.12.
            namespace = _run_parser().parse_intermixed_args(rest)
            return run(namespace.file, namespace.entry, namespace.args, namespace.interp)
        if command == "dump":
            namespace = _dump_parser().parse_args(rest)
            return dump(namespace.file)
    except FileNotFoundError as exc:
        print(f"tinyjit: no such file: {exc.filename}", file=sys.stderr)
        return 1
    except (syntax.SyntaxError_, CompileError, UnsupportedPlatform) as exc:
        print(f"tinyjit: {exc}", file=sys.stderr)
        return 1

    print(f"tinyjit: unknown command {command!r}\n{_USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

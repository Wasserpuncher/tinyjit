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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tinyjit",
        description="A JIT compiler that turns a small language into x86-64 machine code.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="compile a program and call one of its functions")
    run_parser.add_argument("file")
    run_parser.add_argument("-e", "--entry", default="main", help="function to call (default: main)")
    run_parser.add_argument("args", nargs="*", type=int, help="integer arguments")
    run_parser.add_argument(
        "--interp", action="store_true",
        help="evaluate with the reference interpreter instead of native code",
    )

    dump_parser = sub.add_parser("dump", help="show the generated machine code")
    dump_parser.add_argument("file")

    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return run(args.file, args.entry, args.args, args.interp)
        if args.command == "dump":
            return dump(args.file)
    except FileNotFoundError as exc:
        print(f"tinyjit: no such file: {exc.filename}", file=sys.stderr)
        return 1
    except (syntax.SyntaxError_, CompileError, UnsupportedPlatform) as exc:
        print(f"tinyjit: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())

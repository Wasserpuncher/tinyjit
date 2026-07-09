"""The public API: compile tinyjit source straight to callable native code."""

from __future__ import annotations

from . import syntax
from .compiler import compile_program
from .executor import Program


class Module:
    """A compiled program. Its functions are attributes and dict items alike.

    >>> module = compile('fn add(a, b) { return a + b; }')
    >>> module['add'](2, 3)
    5
    >>> module.add(40, 2)
    42
    """

    def __init__(self, source: str):
        functions = syntax.parse(source)
        code, table = compile_program(functions)
        self._program = Program(code)  # keep the mapping alive as long as we live
        self._functions = {
            name: self._program.entry(offset, arity)
            for name, (offset, arity) in table.items()
        }
        self.machine_code = code

    def __getitem__(self, name: str):
        try:
            return self._functions[name]
        except KeyError:
            raise KeyError(f"no function named {name!r}") from None

    def __getattr__(self, name: str):
        # Only reached for names not found as normal attributes.
        functions = self.__dict__.get("_functions", {})
        if name in functions:
            return functions[name]
        raise AttributeError(name)

    def __contains__(self, name: str) -> bool:
        return name in self._functions

    def names(self) -> list[str]:
        return list(self._functions)


def compile(source: str) -> Module:
    """Compile tinyjit source into a :class:`Module` of native functions."""
    return Module(source)

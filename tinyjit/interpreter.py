"""A tree-walking reference interpreter for tinyjit.

Its only job is to define the language's semantics *independently* of the
compiler, so the two can be compared. Every arithmetic operation is reduced to
the signed 64-bit domain with two's-complement wraparound, matching exactly what
the generated x86-64 instructions do — including the awkward corners of signed
division, which is where a naive implementation and real hardware disagree.
"""

from __future__ import annotations

from . import syntax
from .syntax import (
    Assign, Binary, Call, Const, ExprStatement, Function, If, Let, Return, Unary,
    Var, While,
)

_MASK = (1 << 64) - 1
_SIGN = 1 << 63


class InterpreterError(RuntimeError):
    pass


class _ReturnSignal(Exception):
    def __init__(self, value: int):
        self.value = value


def wrap64(value: int) -> int:
    """Reduce a Python int to the signed 64-bit range, wrapping like the CPU."""
    value &= _MASK
    return value - (1 << 64) if value & _SIGN else value


def _div(a: int, b: int) -> int:
    # x86 idiv truncates toward zero and traps on /0 and INT64_MIN / -1.
    # The compiler guards both; we define the same total behaviour here.
    if b == 0:
        return 0
    if b == -1:
        return wrap64(-a)
    sign = -1 if (a < 0) != (b < 0) else 1
    return wrap64(sign * (abs(a) // abs(b)))


def _mod(a: int, b: int) -> int:
    if b == 0 or b == -1:
        return 0
    remainder = abs(a) % abs(b)
    return wrap64(-remainder if a < 0 else remainder)


_ARITH = {
    "+": lambda a, b: wrap64(a + b),
    "-": lambda a, b: wrap64(a - b),
    "*": lambda a, b: wrap64(a * b),
    "/": _div,
    "%": _mod,
    "<": lambda a, b: 1 if a < b else 0,
    "<=": lambda a, b: 1 if a <= b else 0,
    ">": lambda a, b: 1 if a > b else 0,
    ">=": lambda a, b: 1 if a >= b else 0,
    "==": lambda a, b: 1 if a == b else 0,
    "!=": lambda a, b: 1 if a != b else 0,
}


class Interpreter:
    def __init__(self, functions: list[Function]):
        self.functions = {fn.name: fn for fn in functions}
        if len(self.functions) != len(functions):
            raise InterpreterError("duplicate function definition")

    def call(self, name: str, args: list[int]) -> int:
        function = self.functions.get(name)
        if function is None:
            raise InterpreterError(f"call to undefined function {name!r}")
        if len(args) != len(function.params):
            raise InterpreterError(
                f"{name!r} expects {len(function.params)} args, got {len(args)}"
            )
        scope = {param: wrap64(value) for param, value in zip(function.params, args)}
        try:
            self._exec_block(function.body, scope)
        except _ReturnSignal as signal:
            return signal.value
        return 0  # falling off the end returns 0, matching the compiler

    def _exec_block(self, statements, scope: dict[str, int]) -> None:
        for statement in statements:
            self._exec(statement, scope)

    def _exec(self, statement, scope: dict[str, int]) -> None:
        if isinstance(statement, Let) or isinstance(statement, Assign):
            scope[statement.name] = self._eval(statement.value, scope)
        elif isinstance(statement, Return):
            raise _ReturnSignal(self._eval(statement.value, scope))
        elif isinstance(statement, If):
            if self._eval(statement.condition, scope) != 0:
                self._exec_block(statement.then_body, scope)
            else:
                self._exec_block(statement.else_body, scope)
        elif isinstance(statement, While):
            while self._eval(statement.condition, scope) != 0:
                self._exec_block(statement.body, scope)
        elif isinstance(statement, ExprStatement):
            self._eval(statement.expr, scope)
        else:
            raise InterpreterError(f"unknown statement {statement!r}")

    def _eval(self, node, scope: dict[str, int]) -> int:
        if isinstance(node, Const):
            return wrap64(node.value)
        if isinstance(node, Var):
            if node.name not in scope:
                raise InterpreterError(f"undefined variable {node.name!r}")
            return scope[node.name]
        if isinstance(node, Unary):
            operand = self._eval(node.operand, scope)
            return wrap64(-operand) if node.op == "-" else (1 if operand == 0 else 0)
        if isinstance(node, Binary):
            if node.op == "&&":
                return 1 if (self._eval(node.left, scope) != 0 and self._eval(node.right, scope) != 0) else 0
            if node.op == "||":
                return 1 if (self._eval(node.left, scope) != 0 or self._eval(node.right, scope) != 0) else 0
            left = self._eval(node.left, scope)
            right = self._eval(node.right, scope)
            return _ARITH[node.op](left, right)
        if isinstance(node, Call):
            return self.call(node.name, [self._eval(arg, scope) for arg in node.args])
        raise InterpreterError(f"unknown expression {node!r}")


def run(source: str, entry: str, args: list[int]) -> int:
    return Interpreter(syntax.parse(source)).call(entry, args)

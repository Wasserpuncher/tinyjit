"""Compile a tinyjit program to a single blob of x86-64 machine code.

The strategy is intentionally simple and easy to verify rather than fast: a
**stack machine**. Every expression leaves its result in ``rax``; to combine two
values, the left operand is spilled to a slot in the stack frame while the right
is evaluated, then reloaded. This never needs register allocation and is trivial
to get right, yet the output still runs on the bare CPU.

Frame layout, growing downward from ``rbp``::

    [rbp -  8]   local 0   (first parameter, or first `let`)
    [rbp - 16]   local 1
       ...        ...
    [rbp - 8*L]  local L-1
    [rbp - 8*(L+1)]   eval-stack slot 0
    [rbp - 8*(L+2)]   eval-stack slot 1
       ...

Because intermediate values live in these frame slots rather than on the CPU
stack, ``rsp`` never moves after the prologue. It is aligned to 16 bytes there
and stays aligned, so every ``call`` we emit already satisfies the System V ABI.

Calls between functions are emitted as relocations and patched once all
functions are laid out contiguously, so ``call rel32`` always reaches its target.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import assembler
from .assembler import ARG_REGISTERS, RAX, RCX, Assembler
from .interpreter import wrap64
from .syntax import (
    Assign, Binary, Call, Const, ExprStatement, Function, If, Let, Return, Unary,
    Var, While,
)

_COMPARISONS = {"<": "l", "<=": "le", ">": "g", ">=": "ge", "==": "e", "!=": "ne"}


class CompileError(RuntimeError):
    pass


@dataclass
class CompiledFunction:
    name: str
    arity: int
    code: bytes
    relocations: list
    offset: int = 0  # filled in by the linker


class _FunctionCompiler:
    """Emits code for one function, tracking the eval-stack depth as it goes."""

    def __init__(self, function: Function, known_functions: set[str]):
        self.function = function
        self.known_functions = known_functions
        self.asm = Assembler()
        self.depth = 0
        self.max_depth = 0
        self.label_counter = 0

        # Assign a frame slot to every parameter and every declared variable.
        self.slots: dict[str, int] = {}
        for param in function.params:
            self._declare(param)
        for name in _declared_names(function.body):
            if name not in self.slots:
                self._declare(name)

    def _declare(self, name: str) -> None:
        if name in self.slots:
            raise CompileError(f"{self.function.name!r}: variable {name!r} declared twice")
        self.slots[name] = len(self.slots)

    def _new_label(self, hint: str) -> str:
        self.label_counter += 1
        return f".{self.function.name}.{hint}{self.label_counter}"

    # --- frame slot addressing -------------------------------------------

    def _local_disp(self, name: str) -> int:
        return -8 * (self.slots[name] + 1)

    def _eval_disp(self, depth: int) -> int:
        return -8 * (len(self.slots) + depth + 1)

    def _push_acc(self) -> None:
        self.asm.store_rbp(self._eval_disp(self.depth), RAX)
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

    def _pop_to(self, reg: int) -> None:
        self.depth -= 1
        self.asm.load_rbp(reg, self._eval_disp(self.depth))

    # --- top level -------------------------------------------------------

    def compile(self) -> CompiledFunction:
        frame_position = self.asm.prologue()

        # Spill incoming register arguments into their frame slots.
        for index, param in enumerate(self.function.params):
            self.asm.store_rbp(self._local_disp(param), ARG_REGISTERS[index])

        for statement in self.function.body:
            self._stmt(statement)

        # Safety net: if control reaches the end without a `return`, return 0.
        self.asm.xor_rax_rax()
        self.asm.leave_ret()

        frame_bytes = _round_up_16(8 * (len(self.slots) + self.max_depth))
        self.asm.patch_u32(frame_position, frame_bytes)

        if self.depth != 0:  # pure self-check: the stack machine must balance
            raise CompileError(f"internal error: eval stack unbalanced by {self.depth}")

        code, relocations = self.asm.assemble()
        return CompiledFunction(self.function.name, len(self.function.params), code, relocations)

    # --- statements ------------------------------------------------------

    def _stmt(self, statement) -> None:
        if isinstance(statement, (Let, Assign)):
            self._expr(statement.value)
            self.asm.store_rbp(self._local_disp(statement.name), RAX)
        elif isinstance(statement, Return):
            self._expr(statement.value)
            self.asm.leave_ret()
        elif isinstance(statement, If):
            self._if(statement)
        elif isinstance(statement, While):
            self._while(statement)
        elif isinstance(statement, ExprStatement):
            self._expr(statement.expr)  # evaluated for effect; result discarded
        else:
            raise CompileError(f"cannot compile statement {statement!r}")

    def _if(self, node: If) -> None:
        else_label = self._new_label("else")
        end_label = self._new_label("endif")
        self._expr(node.condition)
        self.asm.test_rax_rax()
        self.asm.je(else_label)
        for statement in node.then_body:
            self._stmt(statement)
        self.asm.jmp(end_label)
        self.asm.label(else_label)
        for statement in node.else_body:
            self._stmt(statement)
        self.asm.label(end_label)

    def _while(self, node: While) -> None:
        start_label = self._new_label("while")
        end_label = self._new_label("endwhile")
        self.asm.label(start_label)
        self._expr(node.condition)
        self.asm.test_rax_rax()
        self.asm.je(end_label)
        for statement in node.body:
            self._stmt(statement)
        self.asm.jmp(start_label)
        self.asm.label(end_label)

    # --- expressions (each leaves its result in rax) ---------------------

    def _expr(self, node) -> None:
        if isinstance(node, Const):
            self.asm.mov_imm(RAX, wrap64(node.value))
        elif isinstance(node, Var):
            self._require_local(node.name)
            self.asm.load_rbp(RAX, self._local_disp(node.name))
        elif isinstance(node, Unary):
            self._unary(node)
        elif isinstance(node, Binary):
            self._binary(node)
        elif isinstance(node, Call):
            self._call(node)
        else:
            raise CompileError(f"cannot compile expression {node!r}")

    def _require_local(self, name: str) -> None:
        if name not in self.slots:
            raise CompileError(f"{self.function.name!r}: undefined variable {name!r}")

    def _unary(self, node: Unary) -> None:
        self._expr(node.operand)
        if node.op == "-":
            self.asm.neg_rax()
        else:  # logical not: rax = (rax == 0)
            self.asm.test_rax_rax()
            self.asm.setcc_al("e")
            self.asm.movzx_rax_al()

    def _binary(self, node: Binary) -> None:
        if node.op in ("&&", "||"):
            self._short_circuit(node)
            return

        self._expr(node.left)
        self._push_acc()
        self._expr(node.right)
        self.asm.mov_reg(RCX, RAX)  # right -> rcx
        self._pop_to(RAX)           # left  -> rax

        op = node.op
        if op == "+":
            self.asm.add_rax_rcx()
        elif op == "-":
            self.asm.sub_rax_rcx()
        elif op == "*":
            self.asm.imul_rax_rcx()
        elif op == "/":
            self._divmod(is_mod=False)
        elif op == "%":
            self._divmod(is_mod=True)
        elif op in _COMPARISONS:
            self.asm.cmp_rax_rcx()
            self.asm.setcc_al(_COMPARISONS[op])
            self.asm.movzx_rax_al()
        else:
            raise CompileError(f"unknown operator {op!r}")

    def _divmod(self, is_mod: bool) -> None:
        """Signed divide rax by rcx, guarding the two cases that trap the CPU.

        ``idiv`` raises a hardware exception (which would kill the Python
        process) on division by zero and on ``INT64_MIN / -1``. We branch around
        both: divide-by-zero yields 0, and the ``-1`` case is done with ``neg``
        so no ``idiv`` executes. The interpreter defines identical results, so
        the differential tests cover these corners rather than crashing on them.
        """
        zero_label = self._new_label("divzero")
        neg1_label = self._new_label("divneg1")
        done_label = self._new_label("divdone")

        self.asm.cmp_rcx_imm8(0)      # divisor == 0 ?
        self.asm.je(zero_label)
        self.asm.cmp_rcx_imm8(-1)     # divisor == -1 ?
        self.asm.je(neg1_label)

        # Normal path: rax already holds the dividend, rcx the divisor.
        self.asm.cqo()
        self.asm.idiv_rcx()
        if is_mod:
            self.asm.mov_reg(RAX, assembler.RDX)  # remainder -> rax
        self.asm.jmp(done_label)

        self.asm.label(neg1_label)
        if is_mod:
            self.asm.xor_rax_rax()    # a % -1 == 0
        else:
            self.asm.neg_rax()        # a / -1 == -a, without the idiv overflow

        self.asm.jmp(done_label)

        self.asm.label(zero_label)
        self.asm.xor_rax_rax()        # a / 0 == a % 0 == 0

        self.asm.label(done_label)

    def _short_circuit(self, node: Binary) -> None:
        end = self._new_label("boolend")
        if node.op == "&&":
            false_label = self._new_label("and_false")
            self._expr(node.left)
            self.asm.test_rax_rax()
            self.asm.je(false_label)
            self._expr(node.right)
            self.asm.test_rax_rax()
            self.asm.setcc_al("ne")
            self.asm.movzx_rax_al()
            self.asm.jmp(end)
            self.asm.label(false_label)
            self.asm.xor_rax_rax()
            self.asm.label(end)
        else:  # ||
            true_label = self._new_label("or_true")
            self._expr(node.left)
            self.asm.test_rax_rax()
            self.asm.jne(true_label)
            self._expr(node.right)
            self.asm.test_rax_rax()
            self.asm.setcc_al("ne")
            self.asm.movzx_rax_al()
            self.asm.jmp(end)
            self.asm.label(true_label)
            self.asm.mov_imm(RAX, 1)
            self.asm.label(end)

    def _call(self, node: Call) -> None:
        if node.name not in self.known_functions:
            raise CompileError(f"call to undefined function {node.name!r}")
        if len(node.args) > len(ARG_REGISTERS):
            raise CompileError(f"call to {node.name!r} has too many arguments")

        for arg in node.args:
            self._expr(arg)
            self._push_acc()
        for index in reversed(range(len(node.args))):
            self._pop_to(ARG_REGISTERS[index])
        self.asm.call_symbol(node.name)
        # result is already in rax


def _declared_names(statements) -> list[str]:
    """All variable names introduced by `let` anywhere in a function body."""
    names: list[str] = []
    for statement in statements:
        if isinstance(statement, Let):
            names.append(statement.name)
        elif isinstance(statement, If):
            names += _declared_names(statement.then_body)
            names += _declared_names(statement.else_body)
        elif isinstance(statement, While):
            names += _declared_names(statement.body)
    return names


def _round_up_16(value: int) -> int:
    return (value + 15) // 16 * 16


def compile_program(functions: list[Function]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """Compile and link all functions into one blob.

    Returns the machine-code bytes and a table mapping each function name to its
    ``(offset, arity)`` — everything :mod:`tinyjit.jit` needs to build callables.
    """
    names = {fn.name for fn in functions}
    if len(names) != len(functions):
        raise CompileError("duplicate function definition")

    compiled = [_FunctionCompiler(fn, names).compile() for fn in functions]

    # Lay the functions out back to back and record where each one starts.
    blob = bytearray()
    offsets: dict[str, int] = {}
    for function in compiled:
        function.offset = len(blob)
        offsets[function.name] = function.offset
        blob.extend(function.code)

    # Patch every inter-function call now that all offsets are known.
    for function in compiled:
        for reloc in function.relocations:
            call_site = function.offset + reloc.position
            target = offsets[reloc.symbol]
            rel = target - (call_site + 4)
            _patch_i32(blob, call_site, rel)

    table = {fn.name: (fn.offset, fn.arity) for fn in compiled}
    return bytes(blob), table


def _patch_i32(blob: bytearray, position: int, value: int) -> None:
    import struct

    struct.pack_into("<i", blob, position, value)

"""A tiny x86-64 machine-code assembler.

This emits raw bytes for the small instruction subset the compiler needs, in the
System V AMD64 calling convention (Linux / Intel macOS). It is not a general
assembler; every method here exists because some code path in ``compiler.py``
needs it, and each encoding is exercised by execution in the test suite.

Two kinds of forward reference are supported:

* **Local labels** — targets of ``jmp``/``je``/``jne`` inside one function.
  Resolved in :meth:`Assembler.assemble` once every label position is known.
* **Call symbols** — calls to *other* functions, whose final address is only
  known after all functions are laid out in one blob. These are returned as
  relocations for :mod:`tinyjit.compiler` to patch during linking.

Register numbers follow the x86 encoding: rax=0, rcx=1, rdx=2, rbx=3, rsp=4,
rbp=5, rsi=6, rdi=7, r8=8 … r15=15.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Registers we actually name.
RAX, RCX, RDX = 0, 1, 2
RSP, RBP = 4, 5
RSI, RDI = 6, 7
R8, R9 = 8, 9

# System V passes the first six integer arguments in these registers, in order.
ARG_REGISTERS = (RDI, RSI, RDX, RCX, R8, R9)

# Condition codes for the setcc/jcc families, by the low nibble of their opcode.
_SETCC = {"e": 0x94, "ne": 0x95, "l": 0x9C, "le": 0x9E, "g": 0x9F, "ge": 0x9D}


class AssemblyError(RuntimeError):
    pass


@dataclass
class _Fixup:
    position: int  # offset of the 4-byte little-endian displacement
    label: str


@dataclass
class Relocation:
    """A ``call`` whose target is another function, patched at link time."""

    position: int  # offset of the 4-byte rel32 within the final function bytes
    symbol: str


@dataclass
class Assembler:
    code: bytearray = field(default_factory=bytearray)
    _labels: dict[str, int] = field(default_factory=dict)
    _fixups: list[_Fixup] = field(default_factory=list)
    _relocations: list[Relocation] = field(default_factory=list)

    # --- primitives -------------------------------------------------------

    def emit(self, *bytes_: int) -> None:
        self.code.extend(bytes_)

    def _rex(self, reg: int, rm: int) -> int:
        """REX.W prefix, extended with .R/.B when a register needs a 4th bit."""
        rex = 0x48  # REX.W: 64-bit operand size
        if reg >= 8:
            rex |= 0x04  # REX.R
        if rm >= 8:
            rex |= 0x01  # REX.B
        return rex

    def _modrm_reg(self, reg: int, rm: int) -> int:
        """ModRM for register-direct addressing (mod=11)."""
        return 0xC0 | ((reg & 7) << 3) | (rm & 7)

    def _modrm_mem_rbp(self, reg: int) -> int:
        """ModRM for ``[rbp + disp32]`` (mod=10, rm=rbp)."""
        return 0x80 | ((reg & 7) << 3) | (RBP & 7)

    # --- labels and relocations ------------------------------------------

    def label(self, name: str) -> None:
        if name in self._labels:
            raise AssemblyError(f"duplicate label {name!r}")
        self._labels[name] = len(self.code)

    def _jump(self, opcode: bytes, label: str) -> None:
        self.code.extend(opcode)
        self._fixups.append(_Fixup(len(self.code), label))
        self.code.extend(b"\x00\x00\x00\x00")

    def jmp(self, label: str) -> None:
        self._jump(b"\xE9", label)

    def je(self, label: str) -> None:
        self._jump(b"\x0F\x84", label)

    def jne(self, label: str) -> None:
        self._jump(b"\x0F\x85", label)

    def call_symbol(self, symbol: str) -> None:
        self.emit(0xE8)
        self._relocations.append(Relocation(len(self.code), symbol))
        self.code.extend(b"\x00\x00\x00\x00")

    # --- data movement ---------------------------------------------------

    def mov_imm(self, reg: int, value: int) -> None:
        """``movabs reg, imm64`` — loads any 64-bit constant."""
        self.emit(self._rex(0, reg), 0xB8 + (reg & 7))
        self.code.extend(struct.pack("<q", _as_int64(value)))

    def mov_reg(self, dst: int, src: int) -> None:
        """``mov dst, src`` (64-bit register to register)."""
        self.emit(self._rex(src, dst), 0x89, self._modrm_reg(src, dst))

    def load_rbp(self, reg: int, disp: int) -> None:
        """``mov reg, [rbp + disp]``."""
        self.emit(self._rex(reg, RBP), 0x8B, self._modrm_mem_rbp(reg))
        self.code.extend(struct.pack("<i", disp))

    def store_rbp(self, disp: int, reg: int) -> None:
        """``mov [rbp + disp], reg``."""
        self.emit(self._rex(reg, RBP), 0x89, self._modrm_mem_rbp(reg))
        self.code.extend(struct.pack("<i", disp))

    # --- arithmetic (all operate on rax, with rcx as the second operand) --

    def add_rax_rcx(self) -> None:
        self.emit(0x48, 0x01, self._modrm_reg(RCX, RAX))

    def sub_rax_rcx(self) -> None:
        self.emit(0x48, 0x29, self._modrm_reg(RCX, RAX))

    def imul_rax_rcx(self) -> None:
        self.emit(0x48, 0x0F, 0xAF, self._modrm_reg(RAX, RCX))

    def neg_rax(self) -> None:
        self.emit(0x48, 0xF7, self._modrm_reg(3, RAX))  # F7 /3

    def cqo(self) -> None:
        """Sign-extend rax into rdx:rax, as idiv requires."""
        self.emit(0x48, 0x99)

    def idiv_rcx(self) -> None:
        self.emit(0x48, 0xF7, self._modrm_reg(7, RCX))  # F7 /7

    def xor_rax_rax(self) -> None:
        self.emit(0x48, 0x31, self._modrm_reg(RAX, RAX))

    # --- comparison and boolean ------------------------------------------

    def cmp_rax_rcx(self) -> None:
        self.emit(0x48, 0x39, self._modrm_reg(RCX, RAX))

    def cmp_rcx_imm8(self, value: int) -> None:
        """``cmp rcx, imm8`` (sign-extended)."""
        self.emit(0x48, 0x83, self._modrm_reg(7, RCX), value & 0xFF)  # 83 /7

    def test_rax_rax(self) -> None:
        self.emit(0x48, 0x85, self._modrm_reg(RAX, RAX))

    def setcc_al(self, cc: str) -> None:
        self.emit(0x0F, _SETCC[cc], 0xC0)  # setcc al

    def movzx_rax_al(self) -> None:
        self.emit(0x48, 0x0F, 0xB6, 0xC0)  # movzx rax, al

    # --- frame management ------------------------------------------------

    def prologue(self) -> int:
        """``push rbp; mov rbp, rsp; sub rsp, <placeholder>``.

        Returns the position of the 4-byte frame-size immediate, so the caller
        can patch it once the final frame size is known.
        """
        self.emit(0x55)  # push rbp
        self.emit(0x48, 0x89, 0xE5)  # mov rbp, rsp
        self.emit(0x48, 0x81, 0xEC)  # sub rsp, imm32
        position = len(self.code)
        self.code.extend(b"\x00\x00\x00\x00")
        return position

    def patch_u32(self, position: int, value: int) -> None:
        struct.pack_into("<I", self.code, position, value & 0xFFFFFFFF)

    def leave_ret(self) -> None:
        self.emit(0xC9, 0xC3)  # leave ; ret

    # --- finalisation ----------------------------------------------------

    def assemble(self) -> tuple[bytes, list[Relocation]]:
        """Resolve local jumps; return the bytes and any call relocations."""
        for fixup in self._fixups:
            if fixup.label not in self._labels:
                raise AssemblyError(f"unresolved label {fixup.label!r}")
            target = self._labels[fixup.label]
            rel = target - (fixup.position + 4)
            struct.pack_into("<i", self.code, fixup.position, rel)
        return bytes(self.code), list(self._relocations)


def _as_int64(value: int) -> int:
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= (1 << 63) else value

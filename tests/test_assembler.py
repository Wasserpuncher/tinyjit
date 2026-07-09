"""Instruction encodings must match the x86-64 architecture exactly.

Execution already proves the encodings work, but pinning the exact bytes here
turns any accidental change into an immediate, readable failure instead of a
mysterious crash later.
"""

import struct
import unittest

from tinyjit import assembler
from tinyjit.assembler import RAX, RCX, RDI, Assembler


def encode(build) -> bytes:
    asm = Assembler()
    build(asm)
    code, relocations = asm.assemble()
    assert not relocations
    return code


class EncodingTest(unittest.TestCase):
    def test_mov_reg(self):
        # mov rax, rdi
        self.assertEqual(encode(lambda a: a.mov_reg(RAX, RDI)), bytes([0x48, 0x89, 0xF8]))
        # mov rcx, rax
        self.assertEqual(encode(lambda a: a.mov_reg(RCX, RAX)), bytes([0x48, 0x89, 0xC1]))

    def test_mov_reg_extended_register(self):
        # mov r8, rax needs REX.B; mov rax, r9 needs REX.R
        self.assertEqual(encode(lambda a: a.mov_reg(assembler.R8, RAX)), bytes([0x49, 0x89, 0xC0]))
        self.assertEqual(encode(lambda a: a.mov_reg(RAX, assembler.R9)), bytes([0x4C, 0x89, 0xC8]))

    def test_mov_imm_is_movabs(self):
        self.assertEqual(
            encode(lambda a: a.mov_imm(RAX, 42)),
            bytes([0x48, 0xB8]) + struct.pack("<q", 42),
        )

    def test_mov_imm_handles_negative_and_large(self):
        self.assertEqual(encode(lambda a: a.mov_imm(RAX, -1))[2:], struct.pack("<q", -1))
        big = 2**63 - 1
        self.assertEqual(encode(lambda a: a.mov_imm(RCX, big))[2:], struct.pack("<q", big))

    def test_arithmetic(self):
        self.assertEqual(encode(lambda a: a.add_rax_rcx()), bytes([0x48, 0x01, 0xC8]))
        self.assertEqual(encode(lambda a: a.sub_rax_rcx()), bytes([0x48, 0x29, 0xC8]))
        self.assertEqual(encode(lambda a: a.imul_rax_rcx()), bytes([0x48, 0x0F, 0xAF, 0xC1]))
        self.assertEqual(encode(lambda a: a.neg_rax()), bytes([0x48, 0xF7, 0xD8]))

    def test_division_helpers(self):
        self.assertEqual(encode(lambda a: a.cqo()), bytes([0x48, 0x99]))
        self.assertEqual(encode(lambda a: a.idiv_rcx()), bytes([0x48, 0xF7, 0xF9]))

    def test_frame_memory_access(self):
        # mov [rbp - 8], rax  and  mov rax, [rbp - 8]
        self.assertEqual(
            encode(lambda a: a.store_rbp(-8, RAX)),
            bytes([0x48, 0x89, 0x85]) + struct.pack("<i", -8),
        )
        self.assertEqual(
            encode(lambda a: a.load_rbp(RAX, -8)),
            bytes([0x48, 0x8B, 0x85]) + struct.pack("<i", -8),
        )

    def test_compare_and_set(self):
        self.assertEqual(encode(lambda a: a.cmp_rax_rcx()), bytes([0x48, 0x39, 0xC8]))
        self.assertEqual(encode(lambda a: a.test_rax_rax()), bytes([0x48, 0x85, 0xC0]))
        self.assertEqual(encode(lambda a: a.setcc_al("l")), bytes([0x0F, 0x9C, 0xC0]))
        self.assertEqual(encode(lambda a: a.movzx_rax_al()), bytes([0x48, 0x0F, 0xB6, 0xC0]))

    def test_cmp_rcx_imm8_sign_extends(self):
        self.assertEqual(encode(lambda a: a.cmp_rcx_imm8(-1)), bytes([0x48, 0x83, 0xF9, 0xFF]))
        self.assertEqual(encode(lambda a: a.cmp_rcx_imm8(0)), bytes([0x48, 0x83, 0xF9, 0x00]))

    def test_prologue_shape(self):
        asm = Assembler()
        position = asm.prologue()
        asm.patch_u32(position, 0x20)
        code, _ = asm.assemble()
        # push rbp ; mov rbp,rsp ; sub rsp, 0x20
        self.assertEqual(code, bytes([0x55, 0x48, 0x89, 0xE5, 0x48, 0x81, 0xEC]) + struct.pack("<I", 0x20))


class ControlFlowTest(unittest.TestCase):
    def test_forward_jump_resolves_to_relative_offset(self):
        asm = Assembler()
        asm.jmp("skip")          # E9 rel32
        asm.add_rax_rcx()        # 3 bytes that get skipped
        asm.label("skip")
        code, _ = asm.assemble()
        # rel32 must equal the distance from end-of-jump to the label (3 bytes).
        rel = struct.unpack("<i", code[1:5])[0]
        self.assertEqual(rel, 3)

    def test_backward_conditional_jump(self):
        asm = Assembler()
        asm.label("top")
        asm.test_rax_rax()       # 3 bytes
        asm.je("top")            # 0F 84 rel32
        code, _ = asm.assemble()
        rel = struct.unpack("<i", code[-4:])[0]
        # Target is 3 bytes before the je, whose rel is measured from its end.
        self.assertEqual(rel, -(3 + 6))

    def test_unresolved_label_is_an_error(self):
        asm = Assembler()
        asm.jmp("nowhere")
        with self.assertRaises(assembler.AssemblyError):
            asm.assemble()

    def test_call_symbol_is_reported_as_a_relocation(self):
        asm = Assembler()
        asm.call_symbol("other")
        code, relocations = asm.assemble()
        self.assertEqual(code[0], 0xE8)
        self.assertEqual(len(relocations), 1)
        self.assertEqual(relocations[0].symbol, "other")
        self.assertEqual(relocations[0].position, 1)


if __name__ == "__main__":
    unittest.main()

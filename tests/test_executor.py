"""The executor: loading, entry points, and argument handling."""

import struct
import unittest

from tinyjit.executor import MAX_ARGS, Program, UnsupportedPlatform, check_supported

try:
    check_supported()
    SUPPORTED = True
except UnsupportedPlatform:
    SUPPORTED = False


@unittest.skipUnless(SUPPORTED, "requires an x86-64 POSIX host")
class ProgramTest(unittest.TestCase):
    # mov rax, rdi ; add rax, rsi ; ret   -> add(a, b)
    ADD = bytes([0x48, 0x89, 0xF8, 0x48, 0x01, 0xF0, 0xC3])
    # mov rax, 42 ; ret
    ANSWER = bytes([0x48, 0xB8]) + struct.pack("<q", 42) + bytes([0xC3])

    def test_executes_loaded_code(self):
        program = Program(self.ADD)
        add = program.entry(0, 2)
        self.assertEqual(add(19, 23), 42)
        self.assertEqual(add(-5, 100), 95)

    def test_multiple_entry_points_in_one_blob(self):
        blob = self.ADD + self.ANSWER
        program = Program(blob)
        add = program.entry(0, 2)
        answer = program.entry(len(self.ADD), 0)
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(answer(), 42)

    def test_arguments_are_reduced_to_int64(self):
        program = Program(self.ADD)
        add = program.entry(0, 2)
        # 2**64 wraps to 0, so this is just 5 + 0.
        self.assertEqual(add(2**64 + 5, 0), 5)

    def test_wrong_argument_count(self):
        add = Program(self.ADD).entry(0, 2)
        with self.assertRaises(TypeError):
            add(1)

    def test_empty_program_is_rejected(self):
        with self.assertRaises(ValueError):
            Program(b"")

    def test_offset_out_of_range(self):
        with self.assertRaises(ValueError):
            Program(self.ANSWER).entry(10_000, 0)

    def test_arity_out_of_range(self):
        with self.assertRaises(ValueError):
            Program(self.ANSWER).entry(0, MAX_ARGS + 1)


if __name__ == "__main__":
    unittest.main()

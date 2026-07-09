"""Lexing and parsing: precedence, associativity, and error reporting."""

import unittest

from tinyjit import syntax
from tinyjit.syntax import Binary, Call, Const, Function, Return, Var


class LexerTest(unittest.TestCase):
    def test_line_comments_are_ignored(self):
        tokens = syntax.tokenize("fn f() { // comment\n return 1; }")
        kinds = [t.value for t in tokens if t.kind != "eof"]
        self.assertNotIn("comment", kinds)

    def test_multi_char_operators(self):
        tokens = syntax.tokenize("a <= b != c && d")
        ops = [t.value for t in tokens if t.kind == "op"]
        self.assertEqual(ops, ["<=", "!=", "&&"])

    def test_number_touching_identifier_is_rejected(self):
        with self.assertRaises(syntax.SyntaxError_):
            syntax.tokenize("123abc")

    def test_unexpected_character(self):
        with self.assertRaises(syntax.SyntaxError_):
            syntax.tokenize("fn f() { return 1 @ 2; }")


class ParserTest(unittest.TestCase):
    def parse_expr(self, text: str):
        program = syntax.parse(f"fn f() {{ return {text}; }}")
        return program[0].body[0].value

    def test_precedence_multiplication_binds_tighter(self):
        # 1 + 2 * 3  ==  1 + (2 * 3)
        node = self.parse_expr("1 + 2 * 3")
        self.assertIsInstance(node, Binary)
        self.assertEqual(node.op, "+")
        self.assertIsInstance(node.right, Binary)
        self.assertEqual(node.right.op, "*")

    def test_left_associativity(self):
        # 10 - 3 - 2  ==  (10 - 3) - 2
        node = self.parse_expr("10 - 3 - 2")
        self.assertEqual(node.op, "-")
        self.assertIsInstance(node.left, Binary)
        self.assertEqual(node.left.left.value, 10)

    def test_comparison_below_arithmetic(self):
        # a + b < c  ==  (a + b) < c
        node = self.parse_expr("a + b < c")
        self.assertEqual(node.op, "<")
        self.assertEqual(node.left.op, "+")

    def test_parentheses_override_precedence(self):
        node = self.parse_expr("(1 + 2) * 3")
        self.assertEqual(node.op, "*")
        self.assertEqual(node.left.op, "+")

    def test_unary_chains(self):
        node = self.parse_expr("- -5")
        self.assertEqual(node.op, "-")
        self.assertEqual(node.operand.op, "-")

    def test_call_parsing(self):
        node = self.parse_expr("gcd(a, b + 1)")
        self.assertIsInstance(node, Call)
        self.assertEqual(node.name, "gcd")
        self.assertEqual(len(node.args), 2)

    def test_full_function(self):
        program = syntax.parse("fn sq(x) { return x * x; }")
        self.assertEqual(len(program), 1)
        function = program[0]
        self.assertEqual(function.name, "sq")
        self.assertEqual(function.params, ("x",))
        self.assertIsInstance(function.body[0], Return)

    def test_if_else_and_while(self):
        program = syntax.parse(
            "fn f(n) { if (n < 0) { return 0; } else { while (n > 0) { n = n - 1; } } return n; }"
        )
        self.assertEqual(program[0].name, "f")

    def test_missing_semicolon_is_reported(self):
        with self.assertRaises(syntax.SyntaxError_):
            syntax.parse("fn f() { return 1 }")

    def test_too_many_parameters(self):
        with self.assertRaises(syntax.SyntaxError_):
            syntax.parse("fn f(a, b, c, d, e, f, g) { return a; }")

    def test_empty_program_is_rejected(self):
        with self.assertRaises(syntax.SyntaxError_):
            syntax.parse("// only a comment\n")


if __name__ == "__main__":
    unittest.main()

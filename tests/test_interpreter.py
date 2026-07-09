"""The reference interpreter's semantics, especially the arithmetic corners."""

import unittest

from tinyjit import interpreter
from tinyjit.interpreter import run, wrap64

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class Wrap64Test(unittest.TestCase):
    def test_identity_in_range(self):
        self.assertEqual(wrap64(0), 0)
        self.assertEqual(wrap64(INT64_MAX), INT64_MAX)
        self.assertEqual(wrap64(INT64_MIN), INT64_MIN)

    def test_wraps_past_the_maximum(self):
        self.assertEqual(wrap64(INT64_MAX + 1), INT64_MIN)
        self.assertEqual(wrap64(2**64), 0)
        self.assertEqual(wrap64(2**63), INT64_MIN)


class ArithmeticTest(unittest.TestCase):
    def evaluate(self, expr: str, *args: int) -> int:
        params = ", ".join(f"a{i}" for i in range(len(args)))
        return run(f"fn f({params}) {{ return {expr}; }}", "f", list(args))

    def test_overflow_wraps(self):
        self.assertEqual(self.evaluate("a0 + a1", INT64_MAX, 1), INT64_MIN)
        self.assertEqual(self.evaluate("a0 * a1", 2**32, 2**32), 0)

    def test_division_truncates_toward_zero(self):
        self.assertEqual(self.evaluate("a0 / a1", 7, 2), 3)
        self.assertEqual(self.evaluate("a0 / a1", -7, 2), -3)   # not floor(-3.5) = -4
        self.assertEqual(self.evaluate("a0 / a1", 7, -2), -3)
        self.assertEqual(self.evaluate("a0 / a1", -7, -2), 3)

    def test_remainder_takes_sign_of_dividend(self):
        self.assertEqual(self.evaluate("a0 % a1", 7, 3), 1)
        self.assertEqual(self.evaluate("a0 % a1", -7, 3), -1)
        self.assertEqual(self.evaluate("a0 % a1", 7, -3), 1)

    def test_division_by_zero_is_defined_as_zero(self):
        self.assertEqual(self.evaluate("a0 / a1", 5, 0), 0)
        self.assertEqual(self.evaluate("a0 % a1", 5, 0), 0)

    def test_int64_min_over_minus_one_does_not_overflow(self):
        self.assertEqual(self.evaluate("a0 / a1", INT64_MIN, -1), INT64_MIN)
        self.assertEqual(self.evaluate("a0 % a1", INT64_MIN, -1), 0)

    def test_boolean_operators_short_circuit_and_yield_0_or_1(self):
        # Right side would divide by zero if it were evaluated.
        self.assertEqual(self.evaluate("a0 != 0 && a1 / a0 > 0", 0, 1), 0)
        self.assertEqual(self.evaluate("a0 == 0 || a1 / a0 > 0", 0, 1), 1)
        self.assertEqual(self.evaluate("!a0", 0), 1)
        self.assertEqual(self.evaluate("!a0", 5), 0)


class ControlFlowTest(unittest.TestCase):
    def test_recursion(self):
        src = "fn fib(n) { if (n < 2) { return n; } return fib(n-1) + fib(n-2); }"
        self.assertEqual(run(src, "fib", [10]), 55)

    def test_fall_off_end_returns_zero(self):
        self.assertEqual(run("fn f() { let x = 5; }", "f", []), 0)

    def test_mutual_recursion(self):
        src = (
            "fn even(n) { if (n == 0) { return 1; } return odd(n - 1); }"
            "fn odd(n) { if (n == 0) { return 0; } return even(n - 1); }"
        )
        self.assertEqual(run(src, "even", [10]), 1)
        self.assertEqual(run(src, "odd", [7]), 1)

    def test_undefined_variable(self):
        with self.assertRaises(interpreter.InterpreterError):
            run("fn f() { return x; }", "f", [])


if __name__ == "__main__":
    unittest.main()

"""The central guarantee: native code computes exactly what the interpreter does.

For every program below we compile it to x86-64, run it on the CPU, and compare
the result against the reference interpreter over a spread of inputs that
includes the values most likely to break a compiler: zero, negatives, and the
signed 64-bit extremes. If a single instruction were mis-encoded, a SHA would
not save us here — but the answers would diverge, and these tests would catch it.
"""

import random
import unittest

import tinyjit
from tinyjit import syntax
from tinyjit.executor import UnsupportedPlatform
from tinyjit.interpreter import Interpreter

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

try:
    tinyjit.compile("fn f() { return 0; }")
    JIT_AVAILABLE = True
except UnsupportedPlatform:
    JIT_AVAILABLE = False


# Interesting inputs: small magnitudes, sign changes, and the 64-bit boundaries.
EDGE_VALUES = [0, 1, -1, 2, -2, 3, 7, -7, 10, 100, -100, 255, 256,
               2**31, -(2**31), 2**32, INT64_MAX, INT64_MIN, INT64_MIN + 1]


@unittest.skipUnless(JIT_AVAILABLE, "requires an x86-64 host")
class DifferentialTest(unittest.TestCase):
    def check(self, source: str, entry: str, inputs: list[tuple]) -> None:
        module = tinyjit.compile(source)
        interp = Interpreter(syntax.parse(source))
        for args in inputs:
            native = module[entry](*args)
            reference = interp.call(entry, list(args))
            self.assertEqual(
                native, reference,
                f"{entry}{args}: native {native} != interpreter {reference}",
            )

    def _pairs(self, count: int = 400) -> list[tuple]:
        rng = random.Random(20260709)
        pairs = [(a, b) for a in EDGE_VALUES for b in EDGE_VALUES]
        pairs += [(rng.randint(INT64_MIN, INT64_MAX), rng.randint(INT64_MIN, INT64_MAX))
                  for _ in range(count)]
        return pairs

    def test_every_binary_operator(self):
        for op in ["+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "&&", "||"]:
            with self.subTest(op=op):
                self.check(f"fn f(a, b) {{ return a {op} b; }}", "f", self._pairs())

    def test_unary_operators(self):
        singles = [(v,) for v in EDGE_VALUES]
        self.check("fn f(a) { return -a; }", "f", singles)
        self.check("fn f(a) { return !a; }", "f", singles)
        self.check("fn f(a) { return - -a; }", "f", singles)

    def test_nested_expression(self):
        src = "fn f(a, b, c) { return (a + b) * c - a / (b - c + 1); }"
        rng = random.Random(1)
        inputs = [(rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6), rng.randint(-10**6, 10**6))
                  for _ in range(500)]
        self.check(src, "f", inputs)

    def test_division_edges_do_not_crash_the_host(self):
        # Every one of these would fault a naive `idiv`; the guards must hold.
        src = "fn f(a, b) { return a / b + a % b * 2; }"
        self.check(src, "f", self._pairs())

    def test_fibonacci_recursion(self):
        src = "fn fib(n) { if (n < 2) { return n; } return fib(n-1) + fib(n-2); }"
        self.check(src, "fib", [(n,) for n in range(28)])

    def test_loops_and_locals(self):
        src = "fn fact(n) { let r = 1; let i = 2; while (i <= n) { r = r * i; i = i + 1; } return r; }"
        self.check(src, "fact", [(n,) for n in range(0, 40)])

    def test_gcd_and_lcm(self):
        src = (
            "fn gcd(a, b) { while (b != 0) { let t = b; b = a % b; a = t; } return a; }"
            "fn lcm(a, b) { return a / gcd(a, b) * b; }"
        )
        rng = random.Random(2)
        pairs = [(rng.randint(1, 10**6), rng.randint(1, 10**6)) for _ in range(300)]
        self.check(src, "gcd", pairs)
        self.check(src, "lcm", pairs)

    def test_mutual_recursion(self):
        src = (
            "fn even(n) { if (n == 0) { return 1; } return odd(n - 1); }"
            "fn odd(n) { if (n == 0) { return 0; } return even(n - 1); }"
        )
        self.check(src, "even", [(n,) for n in range(30)])
        self.check(src, "odd", [(n,) for n in range(30)])

    def test_short_circuit_avoids_the_faulting_branch(self):
        # If && evaluated its right operand when a==0, this would divide by zero;
        # both engines must agree it does not.
        src = "fn f(a, b) { if (a != 0 && b / a > 3) { return 1; } return 0; }"
        self.check(src, "f", self._pairs())

    def test_six_arguments(self):
        src = "fn f(a, b, c, d, e, g) { return a + b*2 + c*3 + d*4 + e*5 + g*6; }"
        rng = random.Random(3)
        inputs = [tuple(rng.randint(-1000, 1000) for _ in range(6)) for _ in range(200)]
        self.check(src, "f", inputs)

    def test_collatz(self):
        src = (
            "fn collatz(n) { let s = 0; while (n != 1 && n > 0) {"
            "  if (n % 2 == 0) { n = n / 2; } else { n = 3*n + 1; } s = s + 1; } return s; }"
        )
        self.check(src, "collatz", [(n,) for n in range(1, 200)])


@unittest.skipUnless(JIT_AVAILABLE, "requires an x86-64 host")
class ApiTest(unittest.TestCase):
    def test_attribute_and_item_access_agree(self):
        module = tinyjit.compile("fn add(a, b) { return a + b; }")
        self.assertEqual(module.add(2, 3), module["add"](2, 3))

    def test_wrong_arity_is_rejected(self):
        module = tinyjit.compile("fn add(a, b) { return a + b; }")
        with self.assertRaises(TypeError):
            module.add(1)

    def test_python_bignum_arguments_are_reduced_mod_2_64(self):
        module = tinyjit.compile("fn id(a) { return a; }")
        self.assertEqual(module.id(2**63), INT64_MIN)  # wraps into range
        self.assertEqual(module.id(2**64 + 5), 5)

    def test_unknown_function(self):
        module = tinyjit.compile("fn f() { return 1; }")
        with self.assertRaises(KeyError):
            module["nope"]

    def test_calling_undefined_function_fails_to_compile(self):
        from tinyjit.compiler import CompileError

        with self.assertRaises(CompileError):
            tinyjit.compile("fn f() { return g(1); }")


if __name__ == "__main__":
    unittest.main()

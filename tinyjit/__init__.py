"""tinyjit: a JIT compiler that turns a small language into x86-64 machine code.

Pure Python, no dependencies, no external toolchain. It parses a C-like source
language, compiles each function to real x86-64 instructions, writes them into
executable memory, and calls them directly on the CPU.

    >>> import tinyjit
    >>> module = tinyjit.compile('''
    ...     fn gcd(a, b) {
    ...         while (b != 0) { let t = b; b = a % b; a = t; }
    ...         return a;
    ...     }
    ... ''')
    >>> module.gcd(1071, 462)
    21
"""

from .jit import Module, compile

__version__ = "1.0.0"
__all__ = ["compile", "Module", "__version__"]

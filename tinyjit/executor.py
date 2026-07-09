"""Turn a blob of machine code into callable native functions.

This is the one place where Python steps outside its own runtime: we ask the
operating system for memory, copy machine code into it, mark it executable, and
hand its address to ``ctypes`` as a function pointer. Calling that pointer
transfers control to the CPU, which runs our bytes directly.

The mechanism is deliberately boring and portable across POSIX systems:

* ``mmap`` gives us page-aligned, anonymous memory we fully control.
* ``mprotect`` flips it from writable to executable. We never leave a page
  simultaneously writable *and* executable (W^X), which is good hygiene and is
  required on hardened kernels.
* ``ctypes.CFUNCTYPE`` builds a C-callable trampoline for each entry point.

Only x86-64 System V (Linux, Intel macOS) is supported; the compiler emits that
ABI. See :func:`check_supported` for the guard.
"""

from __future__ import annotations

import ctypes
import mmap
import platform

MAX_ARGS = 6  # System V passes six integer arguments in registers


class UnsupportedPlatform(RuntimeError):
    """The current machine is not x86-64, or lacks the POSIX calls we need."""


def check_supported() -> None:
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise UnsupportedPlatform(
            f"tinyjit emits x86-64 machine code, but this is {machine!r}. "
            "It runs only on 64-bit Intel/AMD CPUs."
        )
    if not hasattr(ctypes.CDLL(None, use_errno=True), "mprotect"):
        raise UnsupportedPlatform("libc does not expose mprotect(); a POSIX system is required")


class Program:
    """A loaded blob of machine code, exposing one or more entry points.

    The backing memory stays mapped for the lifetime of this object; drop the
    last reference and it is freed. Callables obtained from :meth:`entry` must
    not outlive the Program they came from.
    """

    def __init__(self, code: bytes):
        check_supported()
        if not code:
            raise ValueError("refusing to load an empty program")

        self._size = _round_up(len(code), mmap.PAGESIZE)
        # Anonymous, writable mapping. PROT_EXEC is added only after writing.
        self._buffer = mmap.mmap(-1, self._size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._buffer.write(code)

        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
        _make_executable(self._base, self._size)
        self._entries: dict[tuple[int, int], object] = {}

    @property
    def base_address(self) -> int:
        return self._base

    def entry(self, offset: int, arity: int):
        """Return a Python callable for the function at ``offset`` in the blob."""
        if not 0 <= offset < self._size:
            raise ValueError(f"offset {offset} is outside the loaded program")
        if not 0 <= arity <= MAX_ARGS:
            raise ValueError(f"arity {arity} out of range 0..{MAX_ARGS}")

        key = (offset, arity)
        if key not in self._entries:
            signature = ctypes.CFUNCTYPE(ctypes.c_int64, *([ctypes.c_int64] * arity))
            self._entries[key] = signature(self._base + offset)
        native = self._entries[key]

        def call(*args: int) -> int:
            if len(args) != arity:
                raise TypeError(f"function takes {arity} argument(s), got {len(args)}")
            return int(native(*[_to_int64(a) for a in args]))

        return call


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, (value + multiple - 1) // multiple * multiple)


def _to_int64(value: int) -> int:
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= (1 << 63) else value


def _make_executable(address: int, size: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int

    page = address & ~(mmap.PAGESIZE - 1)
    span = size + (address - page)
    if libc.mprotect(page, span, mmap.PROT_READ | mmap.PROT_EXEC) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mprotect failed: {ctypes.errno.errorcode.get(errno, errno)}")

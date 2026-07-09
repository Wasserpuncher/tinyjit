"""The command-line interface: run, dump, --interp, and error handling."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from tinyjit import cli
from tinyjit.executor import UnsupportedPlatform

try:
    import tinyjit

    tinyjit.compile("fn f() { return 0; }")
    JIT_AVAILABLE = True
except UnsupportedPlatform:
    JIT_AVAILABLE = False

PROGRAM = "fn add(a, b) { return a + b; }\nfn main() { return add(40, 2); }\n"


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "prog.tj"
        self.path.write_text(PROGRAM)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    @unittest.skipUnless(JIT_AVAILABLE, "requires an x86-64 host")
    def test_run_default_entry(self):
        code, out, _ = self.run_cli("run", str(self.path))
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "42")

    @unittest.skipUnless(JIT_AVAILABLE, "requires an x86-64 host")
    def test_run_named_entry_with_args(self):
        code, out, _ = self.run_cli("run", str(self.path), "-e", "add", "19", "23")
        self.assertEqual((code, out.strip()), (0, "42"))

    def test_run_via_interpreter_needs_no_jit(self):
        code, out, _ = self.run_cli("run", str(self.path), "-e", "add", "19", "23", "--interp")
        self.assertEqual((code, out.strip()), (0, "42"))

    def test_dump_lists_functions_and_bytes(self):
        code, out, _ = self.run_cli("dump", str(self.path))
        self.assertEqual(code, 0)
        self.assertIn("fn add", out)
        self.assertIn("fn main", out)
        self.assertIn("total", out)

    def test_missing_file(self):
        code, _, err = self.run_cli("run", "/no/such/file.tj")
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)

    def test_syntax_error_is_reported(self):
        bad = self.path.with_name("bad.tj")
        bad.write_text("fn f( { return 1; }")
        code, _, err = self.run_cli("run", str(bad), "--interp")
        self.assertEqual(code, 1)
        self.assertIn("tinyjit:", err)


if __name__ == "__main__":
    unittest.main()

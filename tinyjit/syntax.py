"""The tinyjit language: tokens, AST, lexer, and a recursive-descent parser.

The language is deliberately small — 64-bit signed integers are the only type —
but large enough to write real programs: functions, recursion, ``if``/``else``,
``while``, local variables, and the usual arithmetic, comparison and
short-circuit boolean operators.

Grammar (EBNF)::

    program    = function+
    function   = "fn" IDENT "(" [ params ] ")" block
    params     = IDENT { "," IDENT }
    block      = "{" statement* "}"
    statement  = "let" IDENT "=" expr ";"
               | "return" expr ";"
               | "if" "(" expr ")" block [ "else" block ]
               | "while" "(" expr ")" block
               | IDENT "=" expr ";"
               | expr ";"
    expr       = or
    or         = and   { "||" and }
    and        = equ   { "&&" equ }
    equ        = cmp   { ("==" | "!=") cmp }
    cmp        = term  { ("<" | "<=" | ">" | ">=") term }
    term       = factor{ ("+" | "-") factor }
    factor     = unary { ("*" | "/" | "%") unary }
    unary      = ("-" | "!") unary | primary
    primary    = INT | IDENT | IDENT "(" [ args ] ")" | "(" expr ")"
"""

from __future__ import annotations

from dataclasses import dataclass

# --- AST ------------------------------------------------------------------


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Unary:
    op: str  # "-" or "!"
    operand: object


@dataclass(frozen=True)
class Binary:
    op: str  # + - * / % < <= > >= == != && ||
    left: object
    right: object


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple


@dataclass(frozen=True)
class Let:
    name: str
    value: object


@dataclass(frozen=True)
class Assign:
    name: str
    value: object


@dataclass(frozen=True)
class Return:
    value: object


@dataclass(frozen=True)
class If:
    condition: object
    then_body: tuple
    else_body: tuple  # possibly empty


@dataclass(frozen=True)
class While:
    condition: object
    body: tuple


@dataclass(frozen=True)
class ExprStatement:
    expr: object


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple
    body: tuple


class SyntaxError_(SyntaxError):
    """A malformed tinyjit program."""


# --- lexer ----------------------------------------------------------------

_KEYWORDS = {"fn", "let", "return", "if", "else", "while"}

# Multi-character operators must be tried before their single-character prefixes.
_OPERATORS = [
    "==", "!=", "<=", ">=", "&&", "||",
    "+", "-", "*", "/", "%", "<", ">", "=", "(", ")", "{", "}", ",", ";", "!",
]


@dataclass(frozen=True)
class Token:
    kind: str  # "int", "ident", "keyword", "op", "eof"
    value: object
    line: int
    column: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, line_start = 0, 1, 0
    n = len(source)

    while i < n:
        ch = source[i]

        if ch == "\n":
            line += 1
            line_start = i + 1
            i += 1
            continue
        if ch in " \t\r":
            i += 1
            continue
        if source.startswith("//", i):  # line comment
            while i < n and source[i] != "\n":
                i += 1
            continue

        column = i - line_start + 1

        if ch.isdigit():
            j = i
            while j < n and source[j].isdigit():
                j += 1
            if j < n and (source[j].isalpha() or source[j] == "_"):
                raise SyntaxError_(f"line {line}: invalid number/identifier at {source[i:j+1]!r}")
            tokens.append(Token("int", int(source[i:j]), line, column))
            i = j
            continue

        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            tokens.append(Token("keyword" if word in _KEYWORDS else "ident", word, line, column))
            i = j
            continue

        for op in _OPERATORS:
            if source.startswith(op, i):
                tokens.append(Token("op", op, line, column))
                i += len(op)
                break
        else:
            raise SyntaxError_(f"line {line}: unexpected character {ch!r}")

    tokens.append(Token("eof", None, line, 0))
    return tokens


# --- parser ---------------------------------------------------------------


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, kind: str, value=None) -> Token:
        token = self.current
        if token.kind != kind or (value is not None and token.value != value):
            wanted = value if value is not None else kind
            got = token.value if token.value is not None else token.kind
            raise SyntaxError_(f"line {token.line}: expected {wanted!r}, got {got!r}")
        return self._advance()

    def _accept(self, kind: str, value) -> bool:
        if self.current.kind == kind and self.current.value == value:
            self._advance()
            return True
        return False

    # program = function+
    def parse_program(self) -> list[Function]:
        functions = []
        while self.current.kind != "eof":
            functions.append(self._function())
        if not functions:
            raise SyntaxError_("program is empty; at least one function is required")
        return functions

    def _function(self) -> Function:
        self._expect("keyword", "fn")
        name = self._expect("ident").value
        self._expect("op", "(")
        params: list[str] = []
        if not (self.current.kind == "op" and self.current.value == ")"):
            params.append(self._expect("ident").value)
            while self._accept("op", ","):
                params.append(self._expect("ident").value)
        self._expect("op", ")")
        if len(params) > 6:
            raise SyntaxError_(f"function {name!r} has {len(params)} params; the limit is 6")
        return Function(name, tuple(params), tuple(self._block()))

    def _block(self) -> list:
        self._expect("op", "{")
        statements = []
        while not (self.current.kind == "op" and self.current.value == "}"):
            statements.append(self._statement())
        self._expect("op", "}")
        return statements

    def _statement(self):
        token = self.current
        if token.kind == "keyword":
            if token.value == "let":
                self._advance()
                name = self._expect("ident").value
                self._expect("op", "=")
                value = self._expr()
                self._expect("op", ";")
                return Let(name, value)
            if token.value == "return":
                self._advance()
                value = self._expr()
                self._expect("op", ";")
                return Return(value)
            if token.value == "if":
                self._advance()
                self._expect("op", "(")
                condition = self._expr()
                self._expect("op", ")")
                then_body = tuple(self._block())
                else_body: tuple = ()
                if self._accept("keyword", "else"):
                    else_body = tuple(self._block())
                return If(condition, then_body, else_body)
            if token.value == "while":
                self._advance()
                self._expect("op", "(")
                condition = self._expr()
                self._expect("op", ")")
                return While(condition, tuple(self._block()))

        # assignment `ident = ...;` versus an expression statement.
        if token.kind == "ident" and self.tokens[self.pos + 1].value == "=" and self.tokens[self.pos + 1].kind == "op":
            name = self._advance().value
            self._advance()  # '='
            value = self._expr()
            self._expect("op", ";")
            return Assign(name, value)

        expr = self._expr()
        self._expect("op", ";")
        return ExprStatement(expr)

    # Expression grammar, lowest precedence first.
    def _expr(self):
        return self._binary_left({"||"}, self._and)

    def _and(self):
        return self._binary_left({"&&"}, self._equality)

    def _equality(self):
        return self._binary_left({"==", "!="}, self._comparison)

    def _comparison(self):
        return self._binary_left({"<", "<=", ">", ">="}, self._term)

    def _term(self):
        return self._binary_left({"+", "-"}, self._factor)

    def _factor(self):
        return self._binary_left({"*", "/", "%"}, self._unary)

    def _binary_left(self, operators: set[str], sub):
        node = sub()
        while self.current.kind == "op" and self.current.value in operators:
            op = self._advance().value
            node = Binary(op, node, sub())
        return node

    def _unary(self):
        if self.current.kind == "op" and self.current.value in ("-", "!"):
            op = self._advance().value
            return Unary(op, self._unary())
        return self._primary()

    def _primary(self):
        token = self.current
        if token.kind == "int":
            self._advance()
            return Const(token.value)
        if token.kind == "ident":
            self._advance()
            if self._accept("op", "("):
                args = []
                if not (self.current.kind == "op" and self.current.value == ")"):
                    args.append(self._expr())
                    while self._accept("op", ","):
                        args.append(self._expr())
                self._expect("op", ")")
                return Call(token.value, tuple(args))
            return Var(token.value)
        if self._accept("op", "("):
            inner = self._expr()
            self._expect("op", ")")
            return inner
        raise SyntaxError_(f"line {token.line}: unexpected token {token.value!r}")


def parse(source: str) -> list[Function]:
    """Parse a complete program into a list of functions."""
    return Parser(tokenize(source)).parse_program()

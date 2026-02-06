"""
AST-Based Code Validator

Static analysis for dynamically generated or untrusted Python code.
Uses Python's ast module to detect dangerous imports, function calls,
and attribute access patterns BEFORE the code is executed.

Usage:
    from utils.code_validator import validate_code

    result = validate_code(user_submitted_code)
    if result.is_safe:
        exec(user_submitted_code)  # or stage for review
    else:
        print(result.summary())

Contributed by: Kendra
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


# ── Default Block/Allow Lists ────────────────────────────────────────────────
# Customize these for your project's security requirements.

BLOCKED_IMPORTS: Set[str] = {
    "os",
    "subprocess",
    "shutil",
    "ctypes",
    "importlib",
    "sys",
    "builtins",
    "code",
    "codeop",
    "compileall",
    "pickle",
    "shelve",
    "marshal",
    "socket",
    "http.server",
    "xmlrpc",
    "ftplib",
    "smtplib",
    "webbrowser",
    "multiprocessing",
    "signal",
    "resource",
}

BLOCKED_CALLS: Set[str] = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
}

BLOCKED_ATTRIBUTES: Set[str] = {
    "os.system",
    "os.popen",
    "os.exec",
    "os.execv",
    "os.execve",
    "os.spawn",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.check_output",
    "shutil.rmtree",
    "shutil.move",
    "shutil.copy",
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class CodeViolation:
    """A single code safety violation."""
    line: int
    category: str  # "blocked_import", "blocked_call", "blocked_attribute", "raw_open", "syntax_error"
    description: str


@dataclass
class CodeValidationResult:
    """Result of code validation."""
    is_safe: bool
    violations: List[CodeViolation] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_safe:
            return "Code validation passed."
        lines = [f"Code validation FAILED ({len(self.violations)} violation(s)):"]
        for v in self.violations:
            lines.append(f"  Line {v.line}: [{v.category}] {v.description}")
        return "\n".join(lines)


# ── AST Visitor ──────────────────────────────────────────────────────────────

class CodeValidator(ast.NodeVisitor):
    """
    AST-based code validator.

    Walks the syntax tree looking for:
    - Blocked imports (os, subprocess, etc.)
    - Dangerous function calls (exec, eval, compile, __import__)
    - Blocked attribute access (os.system, subprocess.run, etc.)
    - Raw open() calls (optional — enable with block_open=True)
    """

    def __init__(
        self,
        blocked_imports: Optional[Set[str]] = None,
        blocked_calls: Optional[Set[str]] = None,
        blocked_attributes: Optional[Set[str]] = None,
        block_open: bool = True,
    ):
        self.blocked_imports = blocked_imports or BLOCKED_IMPORTS
        self.blocked_calls = blocked_calls or BLOCKED_CALLS
        self.blocked_attributes = blocked_attributes or BLOCKED_ATTRIBUTES
        self.block_open = block_open
        self.violations: List[CodeViolation] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in self.blocked_imports:
                self.violations.append(CodeViolation(
                    line=node.lineno,
                    category="blocked_import",
                    description=f"Import of '{alias.name}' is not allowed",
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_root = node.module.split(".")[0]
            if module_root in self.blocked_imports:
                self.violations.append(CodeViolation(
                    line=node.lineno,
                    category="blocked_import",
                    description=f"Import from '{node.module}' is not allowed",
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_call_name(node)

        if func_name in self.blocked_calls:
            self.violations.append(CodeViolation(
                line=node.lineno,
                category="blocked_call",
                description=f"Call to '{func_name}()' is not allowed",
            ))

        if self.block_open and func_name == "open":
            self.violations.append(CodeViolation(
                line=node.lineno,
                category="raw_open",
                description="Direct open() calls are not allowed",
            ))

        if func_name in self.blocked_attributes:
            self.violations.append(CodeViolation(
                line=node.lineno,
                category="blocked_attribute",
                description=f"Call to '{func_name}()' is not allowed",
            ))

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        full_name = self._get_attribute_chain(node)
        if full_name in self.blocked_attributes:
            self.violations.append(CodeViolation(
                line=node.lineno,
                category="blocked_attribute",
                description=f"Access to '{full_name}' is not allowed",
            ))
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_chain(node.func)
        return ""

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Build dotted attribute chain: os.path.join -> 'os.path.join'."""
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))


# ── Public API ───────────────────────────────────────────────────────────────

def validate_code(
    code: str,
    blocked_imports: Optional[Set[str]] = None,
    blocked_calls: Optional[Set[str]] = None,
    blocked_attributes: Optional[Set[str]] = None,
    block_open: bool = True,
) -> CodeValidationResult:
    """
    Validate Python code for safety.

    Args:
        code: Python source code as a string.
        blocked_imports: Override default blocked imports.
        blocked_calls: Override default blocked function calls.
        blocked_attributes: Override default blocked attribute access.
        block_open: If True, block raw open() calls.

    Returns:
        CodeValidationResult with is_safe flag and any violations found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CodeValidationResult(
            is_safe=False,
            violations=[CodeViolation(
                line=e.lineno or 0,
                category="syntax_error",
                description=f"Code has syntax errors: {e.msg}",
            )],
        )

    validator = CodeValidator(
        blocked_imports=blocked_imports,
        blocked_calls=blocked_calls,
        blocked_attributes=blocked_attributes,
        block_open=block_open,
    )
    validator.visit(tree)

    result = CodeValidationResult(
        is_safe=len(validator.violations) == 0,
        violations=validator.violations,
    )

    if not result.is_safe:
        logger.warning(f"Code validation failed: {result.summary()}")
    else:
        logger.debug("Code validation passed")

    return result

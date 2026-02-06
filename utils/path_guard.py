"""
Filesystem Path Guard

Enforces write boundaries for untrusted or sandboxed code execution.
Prevents writes to protected directories while allowing writes to
explicitly whitelisted safe directories.

Usage:
    guard = PathGuard(
        protected_roots=[Path("/app/src")],
        safe_dirs=[Path("/tmp/sandbox"), Path("~/output").expanduser()],
        exceptions=[Path("/app/src/staging")],  # writable within protected root
    )

    if guard.is_write_allowed("/tmp/sandbox/output.txt"):
        write_file(...)  # allowed
    if guard.is_write_allowed("/app/src/main.py"):
        write_file(...)  # blocked

Contributed by: Kendra
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class PathGuard:
    """
    Filesystem boundary enforcer.

    Rules (evaluated in order):
    1. If path is inside a protected root AND inside an exception dir → ALLOW
    2. If path is inside a protected root → BLOCK
    3. If path is inside a safe dir → ALLOW
    4. Otherwise → BLOCK (default-deny)

    All paths are resolved (symlinks followed) before checking.
    """

    def __init__(
        self,
        protected_roots: Optional[List[Path]] = None,
        safe_dirs: Optional[List[Path]] = None,
        exceptions: Optional[List[Path]] = None,
    ):
        """
        Args:
            protected_roots: Directories that are read-only (e.g., project source).
            safe_dirs: Directories where writes are explicitly allowed.
            exceptions: Subdirectories within protected roots that ARE writable
                        (e.g., a staging folder inside the project).
        """
        self._protected = [p.expanduser().resolve() for p in (protected_roots or [])]
        self._safe = [p.expanduser().resolve() for p in (safe_dirs or [])]
        self._exceptions = [p.expanduser().resolve() for p in (exceptions or [])]

    def is_write_allowed(self, target_path: str) -> bool:
        """
        Check if a write to target_path is permitted.

        Args:
            target_path: File or directory path to check (string or Path-like).

        Returns:
            True if the write is allowed, False if blocked.
        """
        resolved = Path(target_path).expanduser().resolve()

        # Check protected roots
        for root in self._protected:
            if self._is_under(resolved, root):
                # Check exceptions (writable subdirs within protected roots)
                for exc in self._exceptions:
                    if self._is_under(resolved, exc):
                        return True
                logger.warning(f"BLOCKED: Write to protected directory: {resolved}")
                return False

        # Check safe directories
        for safe_dir in self._safe:
            if self._is_under(resolved, safe_dir):
                return True

        logger.warning(f"BLOCKED: Write to unallowed path: {resolved}")
        return False

    def add_safe_dir(self, path: Path) -> None:
        """Add a safe directory at runtime."""
        self._safe.append(path.expanduser().resolve())

    def add_protected_root(self, path: Path, exceptions: Optional[List[Path]] = None) -> None:
        """Add a protected root at runtime."""
        self._protected.append(path.expanduser().resolve())
        if exceptions:
            self._exceptions.extend(p.expanduser().resolve() for p in exceptions)

    @staticmethod
    def _is_under(child: Path, parent: Path) -> bool:
        """Check if child is under parent (handles ValueError from relative_to)."""
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

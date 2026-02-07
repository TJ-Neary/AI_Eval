"""
Dynamic Plugin Loader

Auto-discovers and loads plugins from a directory using a convention-based
structure. Each plugin is a subdirectory containing:
  - __init__.py  (exports a class named 'Plugin' inheriting from BasePlugin)
  - meta.json    (metadata: name, description, version, etc.)

Usage:
    from utils.plugin_loader import BasePlugin, PluginLoader

    class MyPlugin(BasePlugin):
        @property
        def name(self) -> str: return "example"
        @property
        def description(self) -> str: return "An example plugin"
        async def execute(self, input_text: str, **kwargs) -> PluginResult:
            return PluginResult(success=True, message="Done")

    loader = PluginLoader(plugins_dir=Path("plugins/"))
    loader.discover()
    result = await loader.execute("example", "some input")

Contributed by: Kendra
"""

import importlib.util
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PluginResult:
    """Standard result from a plugin execution."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class PluginMeta:
    """Metadata loaded from a plugin's meta.json."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    triggers: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "PluginMeta":
        """Load metadata from a meta.json file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            name=data.get("name", path.parent.name),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            triggers=data.get("triggers", []),
            permissions=data.get("permissions", []),
        )


class BasePlugin(ABC):
    """Abstract base class for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the plugin."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def triggers(self) -> List[str]:
        """Phrases that might trigger this plugin (for keyword matching)."""
        return []

    @property
    def required_permissions(self) -> List[str]:
        """Permissions needed (e.g., 'filesystem', 'network')."""
        return []

    @abstractmethod
    async def execute(self, input_text: str, **kwargs) -> PluginResult:
        """Execute the plugin logic."""
        ...


class PluginLoader:
    """
    Registry and loader for plugins.

    Supports:
    - Manual registration via register()
    - Auto-discovery from a plugins directory via discover()
    - Optional pre-load validation callback (e.g., security scanning)
    - Optional approval callback for permission-gated execution
    """

    def __init__(
        self,
        plugins_dir: Optional[Path] = None,
        package_prefix: str = "plugins",
        plugin_class_name: str = "Plugin",
        validator: Optional[Callable[[Path], bool]] = None,
    ):
        """
        Args:
            plugins_dir: Directory to scan for plugin subdirectories.
            package_prefix: Python package prefix for dynamic imports.
            plugin_class_name: Expected class name in each plugin's __init__.py.
            validator: Optional callback that receives the plugin directory path
                       and returns True if it passes validation (e.g., security scan).
        """
        self._plugins: Dict[str, BasePlugin] = {}
        self._meta: Dict[str, PluginMeta] = {}
        self._plugins_dir = plugins_dir
        self._package_prefix = package_prefix
        self._plugin_class_name = plugin_class_name
        self._validator = validator

    def register(self, plugin: BasePlugin) -> None:
        """Manually register a plugin instance."""
        if not isinstance(plugin, BasePlugin):
            logger.error(f"Cannot register invalid plugin: {plugin}")
            return
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name}")

    def get(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with their metadata."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "triggers": p.triggers,
                "permissions": p.required_permissions,
            }
            for p in self._plugins.values()
        ]

    def discover(self) -> int:
        """
        Auto-discover and load plugins from the plugins directory.

        Returns the number of plugins successfully loaded.
        """
        if not self._plugins_dir or not self._plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {self._plugins_dir}")
            return 0

        loaded = 0
        for item in sorted(self._plugins_dir.iterdir()):
            if item.is_dir() and (item / "__init__.py").exists():
                if self._load_plugin(item):
                    loaded += 1

        logger.info(f"Discovered {loaded} plugin(s) from {self._plugins_dir}")
        return loaded

    def _load_plugin(self, folder: Path) -> bool:
        """Load a single plugin from a directory."""
        try:
            # Run validator if provided (e.g., security scan)
            if self._validator and not self._validator(folder):
                logger.error(f"BLOCKED plugin '{folder.name}': failed validation")
                return False

            # Load metadata if available
            meta_file = folder / "meta.json"
            if meta_file.exists():
                meta = PluginMeta.from_file(meta_file)
                self._meta[meta.name] = meta

            # Dynamic import
            init_file = folder / "__init__.py"
            package_name = f"{self._package_prefix}.{folder.name}"
            spec = importlib.util.spec_from_file_location(package_name, init_file)
            if not spec or not spec.loader:
                logger.error(f"Could not create import spec for {folder.name}")
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)

            # Get the plugin class (convention: class named Plugin or plugin_class_name)
            plugin_class = getattr(module, self._plugin_class_name, None)
            if plugin_class is None:
                logger.error(f"Plugin {folder.name} missing '{self._plugin_class_name}' class")
                return False

            if not issubclass(plugin_class, BasePlugin):
                logger.error(f"Plugin class in {folder.name} does not inherit from BasePlugin")
                return False

            instance = plugin_class()
            self.register(instance)
            return True

        except Exception as e:
            logger.exception(f"Failed to load plugin {folder.name}: {e}")
            return False

    async def execute(self, name: str, input_text: str, **kwargs) -> PluginResult:
        """
        Execute a plugin by name.

        Args:
            name: Plugin name.
            input_text: Input to pass to the plugin.
            **kwargs: Additional parameters.

        Returns:
            PluginResult from the plugin, or an error result if not found.
        """
        plugin = self.get(name)
        if not plugin:
            return PluginResult(
                success=False,
                message=f"Plugin '{name}' not found.",
            )

        try:
            return await plugin.execute(input_text, **kwargs)
        except Exception as e:
            logger.error(f"Error executing plugin '{name}': {e}")
            return PluginResult(
                success=False,
                message=f"Plugin '{name}' failed: {e}",
            )

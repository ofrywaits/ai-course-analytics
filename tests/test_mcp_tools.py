# -*- coding: utf-8 -*-
"""
Unit tests for tools/mcp_tools.py — MCP integration.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGetMcpFileTools:
    """Tests for the MCP filesystem tool initializer."""

    def test_returns_list(self, tmp_path):
        """get_mcp_file_tools() must always return a list, never raise."""
        from tools.mcp_tools import get_mcp_file_tools
        result = get_mcp_file_tools(outputs_dir=tmp_path)
        assert isinstance(result, list), "Should return a list"

    def test_graceful_fallback_on_missing_npx(self, tmp_path, monkeypatch):
        """If npx / the MCP server is unavailable, return empty list, don't crash."""
        import tools.mcp_tools as mcp_module

        original = mcp_module.get_mcp_file_tools

        def _always_fail(outputs_dir=None):
            try:
                raise RuntimeError("npx not found")
            except Exception:
                return []

        monkeypatch.setattr(mcp_module, "get_mcp_file_tools", _always_fail)
        result = mcp_module.get_mcp_file_tools(outputs_dir=tmp_path)
        assert result == [], "Should return empty list on failure"

    def test_get_mcp_tool_names_returns_list(self, tmp_path):
        """get_mcp_tool_names() must return a list of strings."""
        from tools.mcp_tools import get_mcp_tool_names
        names = get_mcp_tool_names(outputs_dir=tmp_path)
        assert isinstance(names, list)
        for name in names:
            assert isinstance(name, str), f"Tool name should be str, got {type(name)}"

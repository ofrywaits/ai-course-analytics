# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) integration for the AI Course Analytics Platform.

Connects agents to an MCP filesystem server so they can read output files
directly — giving them grounded access to the actual generated reports
without hardcoding file content into every task description.

Usage
-----
    from tools.mcp_tools import get_mcp_file_tools

    tools = get_mcp_file_tools()        # returns list of CrewAI tools
    agent = Agent(..., tools=tools)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Output directory the MCP server is allowed to read
_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def get_mcp_file_tools(outputs_dir: Optional[Path] = None) -> list:
    """
    Return a list of CrewAI-compatible tools backed by an MCP filesystem server.

    The server is scoped to the outputs/ directory so agents can read
    generated reports (insights.md, evaluation_report.md, model_card.md)
    without any file-path guesswork.

    Parameters
    ----------
    outputs_dir : Path, optional
        Directory to expose via MCP. Defaults to outputs/.

    Returns
    -------
    list
        List of BaseTool instances ready to attach to a CrewAI Agent.
        Returns an empty list if npx / @modelcontextprotocol/server-filesystem
        is not installed, so the crew still runs without MCP.
    """
    directory = str(outputs_dir or _OUTPUTS_DIR)
    logger.info(f"Initialising MCP filesystem server for: {directory}")

    try:
        from crewai_tools import MCPServerAdapter

        mcp_params = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", directory],
        }

        adapter = MCPServerAdapter(mcp_params)
        tools   = adapter.tools

        logger.info(
            f"MCP server started — {len(tools)} tool(s) available: "
            f"{[t.name for t in tools]}"
        )
        return tools

    except ImportError:
        logger.warning("crewai_tools not installed — MCP integration skipped")
        return []
    except Exception as exc:
        logger.warning(f"MCP server failed to start ({exc}) — continuing without MCP")
        return []


def get_mcp_tool_names(outputs_dir: Optional[Path] = None) -> list[str]:
    """Return just the names of available MCP tools (useful for logging / tests)."""
    return [t.name for t in get_mcp_file_tools(outputs_dir)]

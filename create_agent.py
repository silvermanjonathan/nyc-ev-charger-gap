"""create_agent.py: save a new version of ev-gap-agent in Foundry.

Unlike agent_smoke_test.py, this keeps the agent. A saved version can't be
edited. A run with a changed definition saves a new version; a run with an
unchanged one got the existing version back (seen Nov 7, not in the docs).
The agent's endpoint serves the newest version unless you pin another. Run this again whenever
INSTRUCTIONS_FILE or the tool list in agent_tools.py changes. Changing what a
tool's function returns needs no new version: that runs on your Mac.

Run from the repo folder:  python create_agent.py
"""

from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import DefaultAzureCredential

from agent_tools import TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parent
PROJECT_ENDPOINT = "https://nyc-ev-gap-js.services.ai.azure.com/api/projects/nyc-ev-gap"
MODEL_DEPLOYMENT = "gpt-5.1"  # the agent's deployment; classify.py stays on gpt-5.4-nano
AGENT_NAME = "ev-gap-agent"
INSTRUCTIONS_FILE = ROOT / "prompts" / "agent_v5.md"  # v1-v4 kept unchanged for comparison


def main() -> None:
    instructions = INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    tool_names = [tool.name for tool in TOOL_DEFINITIONS]
    with AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential()) as project:
        agent = project.agents.create_version(
            agent_name=AGENT_NAME,
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT,
                instructions=instructions,
                tools=TOOL_DEFINITIONS,
            ),
            # Shown beside the version in the portal, so you can tell versions apart.
            description=f"instructions {INSTRUCTIONS_FILE.name}; tools {', '.join(tool_names)}",
        )
    print(f"Saved {agent.name}, version {agent.version}: model {MODEL_DEPLOYMENT}, tools {tool_names}")


if __name__ == "__main__":
    main()

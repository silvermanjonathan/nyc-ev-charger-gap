"""agent_smoke_test.py: can a Foundry prompt agent on gpt-5.1 call a Python function?

The agent uses gpt-5.1, not the classifier's gpt-5.4-nano: Microsoft's
Agent Service table and nano's model card both leave out function tools
for gpt-5.4-nano. This script checks that the project endpoint, your role,
the SDK and the function-call loop all work before any real tools are built.
It creates a throwaway agent with one stub function, asks one question,
runs the function the agent asks for, sends the result back, then deletes
the agent.
"""

import json
import sys

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

PROJECT_ENDPOINT = "https://nyc-ev-gap-js.services.ai.azure.com/api/projects/nyc-ev-gap"
MODEL_DEPLOYMENT = "gpt-5.1"  # the agent's deployment; classify.py stays on gpt-5.4-nano
AGENT_NAME = "smoke-test-agent"
MARKER = "BLUE-HERON-7"  # made up, so the model can only learn it from the function


def lookup_test_marker(district: str) -> dict:
    """Stub tool. Returns a fixed marker the model can't guess."""
    return {"district": district, "marker": MARKER}


# What the agent is told about the function: its name, what it does, and its arguments.
marker_tool = FunctionTool(
    name="lookup_test_marker",
    description="Look up the test marker for one community district code, such as 207.",
    parameters={
        "type": "object",
        "properties": {
            "district": {"type": "string", "description": "Community district code, such as 207"},
        },
        "required": ["district"],
        "additionalProperties": False,
    },
    strict=True,
)


def usage_line(response) -> str:
    u = response.usage
    if u is None:
        return "usage not reported"
    return f"{u.input_tokens} input + {u.output_tokens} output tokens"


def main() -> int:
    checks = {}
    with AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential()) as project:
        agent = project.agents.create_version(
            agent_name=AGENT_NAME,
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT,
                instructions="Answer by calling your tools. Never guess a marker. "
                "If you can't call a tool, say so.",
                tools=[marker_tool],
            ),
        )
        print(f"1. Agent created: {agent.name}, version {agent.version}")
        try:
            with project.get_openai_client(agent_name=agent.name) as client:
                conversation = client.conversations.create()
                try:
                    # Turn 1: the question. If the agent wants the function, it replies
                    # with a function_call item instead of an answer.
                    response = client.responses.create(
                        conversation=conversation.id,
                        input="What is the test marker for district 207?",
                    )
                    print(f"2. First reply item types: {[item.type for item in response.output]} "
                          f"({usage_line(response)})")
                    calls = [item for item in response.output if item.type == "function_call"]
                    checks["agent asked for the function"] = len(calls) > 0

                    # Foundry doesn't run the function. This script does, on your Mac.
                    outputs = []
                    for call in calls:
                        args = json.loads(call.arguments)
                        print(f"3. Agent asked for {call.name}({args})")
                        result = lookup_test_marker(**args)
                        outputs.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result),
                        })

                    if not outputs:
                        print(f"3. No function call. The agent said: {response.output_text!r}")
                    else:
                        # Turn 2: send the result back; the agent writes its answer.
                        final = client.responses.create(conversation=conversation.id, input=outputs)
                        print(f"4. Final answer: {final.output_text!r} ({usage_line(final)})")
                        checks["answer contains the marker"] = MARKER in final.output_text
                finally:
                    client.conversations.delete(conversation_id=conversation.id)
        finally:
            project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
            print("5. Cleaned up: conversation and agent version deleted.")

    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

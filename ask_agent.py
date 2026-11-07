"""ask_agent.py: ask ev-gap-agent one question and print its answer.

Run from the repo folder:  python ask_agent.py "What data do you have?"

The agent's definition (model, instructions, tool list) lives in Foundry, but
its tools run here. When the agent replies with a function_call, this script
runs that Python function and sends the result back, and it keeps going until
the agent answers in words or MAX_MODEL_CALLS is reached.
"""

import json
import sys

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from agent_tools import TOOL_FUNCTIONS
from create_agent import AGENT_NAME, PROJECT_ENDPOINT

MAX_MODEL_CALLS = 6  # one for the question, up to five after tool results; stops a runaway loop
PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT = 1.25, 10.00  # gpt-5.1 Global, dollars per 1M tokens (Nov 6)


def function_calls(response) -> list:
    """The items in the agent's reply that ask for a function."""
    return [item for item in response.output if item.type == "function_call"]


def run_tools(calls) -> list:
    """Run each function the agent asked for, and package each result for the agent."""
    outputs = []
    for call in calls:
        args = json.loads(call.arguments)
        print(f"  tool call: {call.name}({args})")
        result = TOOL_FUNCTIONS[call.name](**args)
        outputs.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result),
        })
    return outputs


def main(question: str) -> int:
    responses = []
    with AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential()) as project:
        with project.get_openai_client(agent_name=AGENT_NAME) as client:
            conversation = client.conversations.create()
            next_input = question
            for _ in range(MAX_MODEL_CALLS):
                response = client.responses.create(conversation=conversation.id, input=next_input)
                responses.append(response)
                calls = function_calls(response)
                if not calls:
                    break
                next_input = run_tools(calls)

    tokens_in = sum(r.usage.input_tokens for r in responses if r.usage)
    tokens_out = sum(r.usage.output_tokens for r in responses if r.usage)
    dollars = tokens_in * PRICE_PER_M_INPUT / 1e6 + tokens_out * PRICE_PER_M_OUTPUT / 1e6
    summary = f"{len(responses)} model calls, {tokens_in} input + {tokens_out} output tokens, about ${dollars:.4f}"

    if calls:
        print(f"Stopped: the agent still wanted tools after {MAX_MODEL_CALLS} model calls. ({summary})")
        return 1
    print()
    print(response.output_text)
    print()
    print(f"({summary}; conversation {conversation.id})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage: python ask_agent.py "your question"')
    sys.exit(main(sys.argv[1]))

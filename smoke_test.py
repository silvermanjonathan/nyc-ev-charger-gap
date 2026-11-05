"""Smoke test: one request to the gpt-5.4-nano deployment in Foundry.

Checks three things before we build the classifier:
1. Keyless sign-in works (your `az login` session plus the Cognitive Services User role).
2. The deployment answers.
3. Whether this model accepts reasoning effort "none" (cheapest) or needs "low".

Run from the repo folder with the venv active:  python smoke_test.py
Cost: a fraction of a cent.
"""
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import BadRequestError, OpenAI

# Not secrets: the endpoint and deployment name are just addresses.
BASE_URL = os.environ.get("AZURE_OPENAI_BASE_URL", "https://nyc-ev-gap-js.openai.azure.com/openai/v1/")
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")

# Dollars per 1M tokens, Global Standard, East US 2 (Azure retail price list, checked Nov 1, 2026).
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.20, 0.02, 1.25


def make_client():
    # DefaultAzureCredential finds your `az login` session, so no key appears anywhere.
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
    return OpenAI(base_url=BASE_URL, api_key=token_provider)


def ask(client, effort):
    return client.responses.create(
        model=DEPLOYMENT,
        instructions="Reply with exactly one word.",
        input="Say ok.",
        reasoning={"effort": effort},
        max_output_tokens=200,
        store=False,  # don't keep this exchange on Microsoft's side
    )


def cost(usage):
    cached = usage.input_tokens_details.cached_tokens or 0
    fresh = usage.input_tokens - cached
    return (fresh * PRICE_IN + cached * PRICE_CACHED + usage.output_tokens * PRICE_OUT) / 1_000_000


def main():
    client = make_client()
    for effort in ("none", "low"):
        try:
            response = ask(client, effort)
        except BadRequestError as err:
            print(f'effort "{effort}" rejected: {err.message}')
            continue
        usage = response.usage
        print(f'effort "{effort}" accepted')
        print(f"reply: {response.output_text!r}")
        print(f"tokens: {usage.input_tokens} in, {usage.output_tokens} out "
              f"({usage.output_tokens_details.reasoning_tokens} of them reasoning)")
        print(f"cost: ${cost(usage):.6f}")
        return
    print("Neither effort level was accepted. Paste this output to Claude.")


if __name__ == "__main__":
    main()
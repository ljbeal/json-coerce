# JSON-Coerce

Simple testing repo for attempting to "coerce" LLMs to output structured responses.

## Motivation

In the pursuit of structured responses (for RAG) using locally hosted LLMs, we encountered issues with `ollama` and tool calling ([issue here](https://github.com/ollama/ollama/issues/8517)).

This puts a blocker on _some_ models, which causes issues when we want to be as model-agnostic as possible.

### Solution

To solve this problem, we need to ensure that an LLM outputs a correct response. The behaviour of [BAML](https://docs.boundaryml.com/home) is such that 
the prompts are modified with a json-like "form" for the LLM to fill.

We can use this idea with pydantic models.

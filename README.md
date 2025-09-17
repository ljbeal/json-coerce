# JSON-Coerce

Simple testing repo for attempting to "coerce" LLMs to output structured responses.

## Motivation

In the pursuit of structured responses (for RAG) using locally hosted LLMs, we encountered issues with `ollama` and tool calling ([issue here](https://github.com/ollama/ollama/issues/8517)).

This puts a blocker on _some_ models, which causes issues when we want to be as model-agnostic as possible.

### Solution

To solve this problem, we need to ensure that an LLM outputs a correct response. The behaviour of [BAML](https://docs.boundaryml.com/home) is such that 
the prompts are modified with a json-like "form" for the LLM to fill.

We can use this idea with pydantic models.

### Premise

Similar to how BAML functions, we take an output specification (pydantic, in our case) and "decompose" it to a format such that it can be given to the LLM as a task.

As an example, the prompt

```
Extract from the following: "Jason is a 34 year old Software Developer"
```

If we want the properties `name`, `age`, and `position`, we can modify the prompt as follows:

```
Extract from the following: "Jason is a 34 year old Software Developer"

Give your response in the JSON format:
{
    // The person's name
    "name": str,
    // The person's age
    "age": int,
    // The person's job position
    "position": str
}
```

Ideally, the LLM will respond with _only_
```
{
    "name": "Jason",
    "age": 34,
    "position": "Software Developer"
}
```

Which we can then parse into a dictionary and validate with the original pydantic model

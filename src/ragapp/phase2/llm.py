"""Structured calls with per-invocation usage, no global conversation memory."""

from pydantic import BaseModel

from ragapp.shared.llm import get_generator_llm, get_judge_llm


class StructuredClient:
    def __init__(self, role="generator"):
        self.role = role
        self.calls = []

    def invoke(self, schema: type[BaseModel], messages, purpose: str):
        model = get_judge_llm() if self.role == "judge" else get_generator_llm()
        response = model.with_structured_output(schema, include_raw=True).invoke(messages)
        raw = response.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        metadata = getattr(raw, "response_metadata", None) or {}
        self.calls.append({"purpose": purpose, "input_tokens": usage.get("input_tokens"),
                           "output_tokens": usage.get("output_tokens"), "total_tokens": usage.get("total_tokens"),
                           "model": metadata.get("model_name"), "system_fingerprint": metadata.get("system_fingerprint")})
        if response.get("parsing_error") or response.get("parsed") is None:
            raise ValueError(f"Malformed structured response at {purpose}")
        return schema.model_validate(response["parsed"])

    def summary(self):
        result = {"llm_calls": len(self.calls), "calls": self.calls,
                  "scope": "logical calls and usage returned by provider; hidden SDK retries not individually metered"}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            result[name] = sum(c[name] for c in self.calls) if self.calls and all(c[name] is not None for c in self.calls) else None
        return result

"""LLM and embedding factories. Every condition uses the *same* generator model
and temperature; the judge model is used only for evaluation."""

from functools import lru_cache
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragapp.config import get_settings


def chat_options(model: str, temperature: float, effort: str | None = None) -> dict:
    s = get_settings()
    options = {"model": model, "timeout": s.llm_timeout, "max_retries": s.llm_max_retries}
    # GPT-5.4 sampling parameters require reasoning_effort=none.
    if re.match(r"^gpt-5\.(?:2|4)(?:-|$)", model):
        options["reasoning_effort"] = effort or "none"
        if options["reasoning_effort"] == "none":
            options["temperature"] = temperature
    elif model.startswith(("gpt-5", "o1", "o3", "o4")):
        if effort:
            options["reasoning_effort"] = effort
    else:
        options["temperature"] = temperature
    return options


@lru_cache
def get_generator_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(**chat_options(s.llm_model, s.llm_temperature, s.llm_reasoning_effort))


@lru_cache
def get_judge_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(**chat_options(s.judge_model, 0.0, s.judge_reasoning_effort))


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=get_settings().embedding_model)

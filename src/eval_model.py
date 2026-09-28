import os
from deepeval.models import GPTModel, AnthropicModel, GeminiModel

# Maps provider name -> (env var to check, deepeval model class, default model name)
PROVIDER_CONFIG = {
    "openai": ("OPENAI_API_KEY", GPTModel, "gpt-4.1"),
    "anthropic": ("ANTHROPIC_API_KEY", AnthropicModel, "claude-sonnet-4-6"),
    "gemini": ("GOOGLE_API_KEY", GeminiModel, "gemini-3.1-flash-lite"),
}


def get_eval_model(provider: str | None = None, model_name: str | None = None) -> GPTModel | AnthropicModel | GeminiModel:
    """
    Resolve which deepeval model to use.

    - If `provider` is passed explicitly, use that provider's env key (error if missing).
    - Otherwise, auto-detect by checking env vars in priority order.
    - `model_name` overrides the default model for whichever provider is picked.
    """
    # 1. Explicit provider choice takes precedence
    if provider is not None:
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_CONFIG)}"
            )
        env_key, model_cls, default_model = PROVIDER_CONFIG[provider]
        if not os.getenv(env_key):
            raise EnvironmentError(
                f"You chose provider '{provider}' but {env_key} is not set. "
                f"Please add {env_key} to your environment (.env file)."
            )
        return model_cls(model=model_name or default_model, temperature=0)

    # 2. Auto-detect: first provider with a key present in env, in this priority order
    for name in ("openai", "anthropic", "gemini"):
        env_key, model_cls, default_model = PROVIDER_CONFIG[name]
        if os.getenv(env_key):
            return model_cls(model=model_name or default_model, temperature=0)

    # 3. Nothing found — ask the user to add one
    all_keys = ", ".join(cfg[0] for cfg in PROVIDER_CONFIG.values())
    raise EnvironmentError(
        "No LLM provider API key found in your environment. "
        f"Please set one of the following in your .env file: {all_keys}"
    )
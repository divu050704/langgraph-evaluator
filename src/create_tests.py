from deepeval.models import GPTModel, AnthropicModel, GeminiModel
from pydantic import BaseModel

class ModelSchema(BaseModel):
    inputs: list[str]

def exec(model: GPTModel | AnthropicModel | GeminiModel, trace):
    prompt = f'''
    You are being provided the trace of a langgraph function create a list of inputs to test the agent:
    {trace}
    '''
    model_output, _cost = model.generate(
        prompt=prompt,
        schema=ModelSchema, # pyright: ignore[reportArgumentType]
    )

    if not isinstance(model_output, ModelSchema):
        raise TypeError(f"Expected ModelSchema, got {type(model_output)}: {model_output!r}")

    return model_output.inputs
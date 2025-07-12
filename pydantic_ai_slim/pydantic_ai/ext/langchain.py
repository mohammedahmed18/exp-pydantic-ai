from typing import Any, Protocol

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool as _Tool, Tool


class LangChainTool(Protocol):
    # args are like
    # {'dir_path': {'default': '.', 'description': 'Subdirectory to search in.', 'title': 'Dir Path', 'type': 'string'},
    #  'pattern': {'description': 'Unix shell regex, where * matches everything.', 'title': 'Pattern', 'type': 'string'}}
    @property
    def args(self) -> dict[str, JsonSchemaValue]: ...

    def get_input_jsonschema(self) -> JsonSchemaValue: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, *args: Any, **kwargs: Any) -> str: ...


__all__ = ("tool_from_langchain",)


def tool_from_langchain(langchain_tool: LangChainTool) -> Tool:
    """Creates a Pydantic AI tool proxy from a LangChain tool.

    Args:
        langchain_tool: The LangChain tool to wrap.

    Returns:
        A Pydantic AI tool that corresponds to the LangChain tool.
    """
    # Use direct fetches and only copy if necessary
    function_name = langchain_tool.name
    function_description = langchain_tool.description
    inputs = langchain_tool.args
    # Compute defaults and required only once
    defaults = {
        name: detail["default"]
        for name, detail in inputs.items()
        if "default" in detail
    }
    required = sorted(
        name for name, detail in inputs.items() if "default" not in detail
    )
    schema: JsonSchemaValue = langchain_tool.get_input_jsonschema()
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    if required:
        schema["required"] = required

    # Structuring proxy exactly as needed
    def proxy(*args: Any, **kwargs: Any) -> str:
        if args:
            raise AssertionError("This should always be called with kwargs")
        # Fast dictionary merge for Python 3.9+
        merged_kwargs = defaults.copy()
        merged_kwargs.update(kwargs)
        return langchain_tool.run(merged_kwargs)

    return _Tool.from_schema(
        function=proxy,
        name=function_name,
        description=function_description,
        json_schema=schema,
    )

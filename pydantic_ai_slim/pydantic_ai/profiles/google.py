from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """
    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove 'additionalProperties'; raise warning if present
        addl_props = schema.pop('additionalProperties', None)
        if addl_props is not None:
            original_schema = schema.copy()
            original_schema['additionalProperties'] = addl_props
            warnings.warn(
                '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                f' Full schema: {self.schema}\n\n'
                f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                "If Google's APIs are updated to support this properly, please create an issue on the PydanticAI GitHub"
                ' and we will fix this behavior.',
                UserWarning,
            )

        # Remove multiple known-unused keys efficiently.
        for field in (
            'title', 'default', '$schema', 'discriminator',
            'examples', 'exclusiveMaximum', 'exclusiveMinimum'
        ):
            schema.pop(field, None)

        # Convert const to enum of one value
        const_val = schema.pop('const', None)
        if const_val is not None:
            schema['enum'] = [const_val]

        # Convert enums: Gemini only supports string enums
        enum_vals = schema.get('enum')
        if enum_vals is not None:
            schema['type'] = 'string'
            # Only convert to str if not already all str to skip redundant str() for perf
            if not (all(isinstance(v, str) for v in enum_vals) if enum_vals else True):
                schema['enum'] = [str(val) for val in enum_vals]

        if 'oneOf' in schema and 'type' not in schema:
            schema['anyOf'] = schema.pop('oneOf')

        # Annotate format in description if present
        if schema.get('type') == 'string':
            format_val = schema.pop('format', None)
            if format_val is not None:
                desc = schema.get('description')
                if desc:
                    schema['description'] = f'{desc} (format: {format_val})'
                else:
                    schema['description'] = f'Format: {format_val}'

        # Error on $ref
        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        # Convert prefixItems to items in arrays for Gemini compatibility
        prefix_items = schema.pop('prefixItems', None)
        if prefix_items is not None:
            items = schema.get('items')
            existing_items = [items] if items is not None else []
            unique_items = existing_items + [item for item in prefix_items if item not in existing_items]
            ulen = len(unique_items)
            if ulen > 1:
                schema['items'] = {'anyOf': unique_items}
            elif ulen == 1:
                schema['items'] = unique_items[0]
            # Direct set, as Gemini expects minItems exactly for prefixItems
            schema['minItems'] = len(prefix_items)
            if items is None:
                schema['maxItems'] = len(prefix_items)

        return schema

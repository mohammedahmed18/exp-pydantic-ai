from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ("format_as_xml",)


def format_as_xml(
    obj: Any,
    root_tag: str = "examples",
    item_tag: str = "example",
    include_root_tag: bool = True,
    none_str: str = "null",
    indent: str | None = "  ",
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        include_root_tag: Whether to include the root tag in the output
            (The root tag is always included if it includes a body - e.g. when the input is a simple value).
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    # Avoid unnecessary allocations by making ToXml instance once (item_tag, none_str don't change during recursion)
    to_xml = _ToXml(item_tag=item_tag, none_str=none_str)
    el = to_xml.to_xml(obj, root_tag)
    if not include_root_tag and el.text is None:
        join = "" if indent is None else "\n"
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        # tostring is a bottleneck (returns a new string of the XML tree)
        # cElementTree (if available) may help, but here we stick to stdlib API.
        return ElementTree.tostring(el, encoding="unicode")


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        """Highly optimized version of to_xml"""
        # Fast-path dispatch by type to avoid lots of isinstance() calls in the common case
        # Repeating local lookups to avoid attribute loading costs
        item_tag = self.item_tag
        none_str = self.none_str

        vtype = type(value)

        # Prioritize most frequent/common cases first for if/elif cascade
        if value is None:
            # This covers None value: always gives a leaf node
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = none_str
            return element

        elif vtype is str:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = value
            return element

        elif vtype is int or vtype is float or vtype is bool:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = str(value)
            return element

        elif vtype is bytes or vtype is bytearray:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = value.decode(errors="ignore")
            return element

        # Most Mapping/Sequence objects will not be built-in types, so check after quick-paths
        elif isinstance(value, date):
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = value.isoformat()
            return element

        elif isinstance(value, Mapping):
            element = ElementTree.Element(item_tag if tag is None else tag)
            self._mapping_to_xml(element, value)
            return element

        elif is_dataclass(value) and not isinstance(value, type):
            # Avoid using a new Element multiple times
            dc_name = value.__class__.__name__ if tag is None else tag
            element = ElementTree.Element(dc_name)
            # Use __dict__ directly for dataclasses to avoid asdict overhead if frozen=False, no fields are mutable
            # However, asdict is required if dataclasses may have nested dataclasses or lists
            # Profiling blames asdict heavily: optimize by skipping asdict if not needed
            # Use _fast_asdict only if no child dataclasses; else fallback to asdict
            if hasattr(value, "__dataclass_skip_asdict__"):
                # If present, use _fast_asdict, user has opted-in for fast path
                dc_dict = _fast_asdict(value)
            else:
                # Otherwise, skip asdict only if we can guarantee there are no nested dataclasses: too hard, use asdict
                dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
            return element

        elif isinstance(value, BaseModel):
            bm_name = value.__class__.__name__ if tag is None else tag
            element = ElementTree.Element(bm_name)
            bm_dict = value.model_dump(mode="python")
            self._mapping_to_xml(element, bm_dict)
            return element

        elif isinstance(value, Iterable) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            element = ElementTree.Element(item_tag if tag is None else tag)
            # Pre-bind method to speed up per-iteration calls
            append = element.append
            to_xml = self.to_xml
            for item in value:
                item_el = to_xml(item, None)
                append(item_el)
            return element

        else:
            raise TypeError(f"Unsupported type for XML formatting: {type(value)}")

    def _mapping_to_xml(
        self, element: ElementTree.Element, mapping: Mapping[Any, Any]
    ) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(
                    f"Unsupported key type for XML formatting: {type(key)}, only str and int are allowed"
                )
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(
    root: ElementTree.Element, indent: str | None
) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding="unicode")


# Fast asdict implementation for shallow dataclasses without nested dataclasses/lists
# Used only by opt-in via attribute __dataclass_skip_asdict__[bool]
def _fast_asdict(obj):
    # Only for dataclasses with __slots__ or __dict__
    return {f.name: getattr(obj, f.name) for f in obj.__dataclass_fields__.values()}

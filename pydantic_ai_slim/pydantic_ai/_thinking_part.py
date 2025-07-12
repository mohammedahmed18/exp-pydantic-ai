from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = "<think>"
END_THINK_TAG = "</think>"


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    # Fast, memory-efficient rewrite.
    parts: list[ThinkingPart | TextPart] = []
    s = content
    st, et = START_THINK_TAG, END_THINK_TAG
    st_len, et_len = len(st), len(et)
    i = 0
    n = len(s)
    while i < n:
        si = s.find(st, i)
        if si == -1:
            # Remaining is text
            if i < n:
                parts.append(TextPart(content=s[i:]))
            break
        if si > i:
            parts.append(TextPart(content=s[i:si]))
        ti = s.find(et, si + st_len)
        if ti == -1:
            # no closing tag, treat rest as text
            parts.append(TextPart(content=s[si + st_len :]))
            break
        parts.append(ThinkingPart(content=s[si + st_len : ti]))
        i = ti + et_len
    return parts

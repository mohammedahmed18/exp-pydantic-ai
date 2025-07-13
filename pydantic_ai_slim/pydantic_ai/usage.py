from __future__ import annotations as _annotations

from copy import copy
from dataclasses import dataclass

from . import _utils
from .exceptions import UsageLimitExceeded

__all__ = 'Usage', 'UsageLimits'


@dataclass(repr=False)
class Usage:
    """LLM usage associated with a request or run.

    Responsibility for calculating usage is on the model; PydanticAI simply sums the usage information across requests.

    You'll need to look up the documentation of the model you're using to convert usage to monetary costs.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""
    request_tokens: int | None = None
    """Tokens used in processing requests."""
    response_tokens: int | None = None
    """Tokens used in generating responses."""
    total_tokens: int | None = None
    """Total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`."""
    details: dict[str, int] | None = None
    """Any extra details returned by the model."""

    def incr(self, incr_usage: Usage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        # Unroll for fixed known attribute names for performance
        self_requests     = self.requests     if self.requests     is not None else 0
        incr_requests     = incr_usage.requests     if incr_usage.requests     is not None else 0
        self.requests     = self_requests + incr_requests

        self_request_tokens = self.request_tokens if self.request_tokens is not None else 0
        incr_request_tokens = incr_usage.request_tokens if incr_usage.request_tokens is not None else 0
        self.request_tokens = self_request_tokens + incr_request_tokens if (self.request_tokens is not None or incr_usage.request_tokens is not None) else None

        self_response_tokens = self.response_tokens if self.response_tokens is not None else 0
        incr_response_tokens = incr_usage.response_tokens if incr_usage.response_tokens is not None else 0
        self.response_tokens = self_response_tokens + incr_response_tokens if (self.response_tokens is not None or incr_usage.response_tokens is not None) else None

        self_total_tokens = self.total_tokens if self.total_tokens is not None else 0
        incr_total_tokens = incr_usage.total_tokens if incr_usage.total_tokens is not None else 0
        self.total_tokens = self_total_tokens + incr_total_tokens if (self.total_tokens is not None or incr_usage.total_tokens is not None) else None

        incr_details = incr_usage.details
        if incr_details:
            self_details = self.details
            if self_details is None:
                # Direct copy if self.details is None to avoid expensive aggregation
                self.details = incr_details.copy()
            else:
                for k, v in incr_details.items():
                    self_details[k] = self_details.get(k, 0) + v

    def __add__(self, other: Usage) -> Usage:
        """Add two Usages together.

        This is provided so it's trivial to sum usage information from multiple requests and runs.
        """
        # Fast shallow copy via dataclass constructor
        new_usage = self.__class__(
            requests=self.requests,
            request_tokens=self.request_tokens,
            response_tokens=self.response_tokens,
            total_tokens=self.total_tokens,
            details=self.details.copy() if self.details is not None else None
        )
        new_usage.incr(other)
        return new_usage

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token limits as OpenTelemetry attributes."""
        result = {
            'gen_ai.usage.input_tokens': self.request_tokens,
            'gen_ai.usage.output_tokens': self.response_tokens,
        }
        for key, value in (self.details or {}).items():
            result[f'gen_ai.usage.details.{key}'] = value  # pragma: no cover
        return {k: v for k, v in result.items() if v}

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return bool(self.requests or self.request_tokens or self.response_tokens or self.details)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    request_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests to the model."""
    response_tokens_limit: int | None = None
    """The maximum number of tokens allowed in responses from the model."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None
            for limit in (self.request_tokens_limit, self.response_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: Usage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

    def check_tokens(self, usage: Usage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        request_tokens = usage.request_tokens or 0
        if self.request_tokens_limit is not None and request_tokens > self.request_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the request_tokens_limit of {self.request_tokens_limit} ({request_tokens=})'
            )

        response_tokens = usage.response_tokens or 0
        if self.response_tokens_limit is not None and response_tokens > self.response_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the response_tokens_limit of {self.response_tokens_limit} ({response_tokens=})'
            )

        total_tokens = usage.total_tokens or 0
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    __repr__ = _utils.dataclasses_no_defaults_repr

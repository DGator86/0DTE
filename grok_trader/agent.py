from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .audit import Usage
from .config import GrokConfig

SYSTEM_PROMPT = """
You are the autonomous discretionary paper trader for a quantitative 0DTE
terminal. The local framework supplies raw market data and decision-blinded
analytical outputs from Legacy, V2, and V3. You must synthesize the evidence
independently. Their final trade/no-trade decisions are intentionally hidden.

Rules:
- Paper trading only. Never request or imply a live order.
- Use only the local terminal tools. Do not use web or X search.
- You may inspect any raw data or analytical output through tools.
- You may create at most one new defined-risk trade in a cycle.
- A no-trade decision is preferred when evidence, data quality, liquidity, or
  payoff geometry is inadequate.
- Explicitly weigh supporting evidence, contradictory evidence, and why the
  proposed trade is superior to standing down.
- The deterministic firewall is authoritative. Never attempt to bypass it.
- Use submit_paper_trade for a new position, close_grok_position to reduce all
  existing Grok risk, or stand_down when no action is justified.
""".strip()


TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "get_terminal_summary",
        "description": "Return the current decision-blinded terminal summary.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "get_raw_snapshot",
        "description": "Inspect paginated raw framework data for the current tick.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": ["market", "bars", "chain", "option_rows", "weekly_option_rows"]},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 250},
            },
            "required": ["section"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_chain_slice",
        "description": "Inspect option quotes around a strike center.",
        "parameters": {
            "type": "object",
            "properties": {
                "center": {"type": "number"},
                "width": {"type": "number", "minimum": 0.25, "maximum": 50},
                "max_rows": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["center", "width"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_engine_analysis",
        "description": "Return decision-blinded analysis for one framework engine.",
        "parameters": {
            "type": "object",
            "properties": {"engine": {"type": "string", "enum": ["legacy", "v2", "v3", "shared"]}},
            "required": ["engine"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_account_state",
        "description": "Return the Grok paper account and open positions.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "submit_paper_trade",
        "description": "Propose one same-day, defined-risk paper options trade. The local firewall reprices and validates it.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "expiration": {"type": "string", "description": "Use 0DTE, same_day, or today's ISO date."},
                "family": {"type": "string", "enum": ["put_credit", "call_credit", "iron_condor", "long_call_spread", "long_put_spread"]},
                "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral", "volatility_expansion", "volatility_contraction"]},
                "legs": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["C", "P", "call", "put"]},
                            "strike": {"type": "number"},
                            "side": {"type": "string", "enum": ["buy", "sell"]},
                        },
                        "required": ["kind", "strike", "side"],
                        "additionalProperties": False,
                    },
                },
                "limit_price": {"type": "number", "exclusiveMinimum": 0},
                "risk_fraction": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.05},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "thesis": {"type": "string", "maxLength": 2000},
                "supporting_evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "contradictory_evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "invalidation_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "mandatory_exit_time": {"type": "string"},
            },
            "required": ["symbol", "expiration", "family", "direction", "legs", "limit_price", "risk_fraction", "confidence", "thesis", "supporting_evidence", "contradictory_evidence", "invalidation_conditions", "mandatory_exit_time"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "close_grok_position",
        "description": "Close all open Grok paper positions using current quotes. This can only reduce risk.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "maxLength": 1000}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "stand_down",
        "description": "Take no trade and record the current thesis/watch conditions.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "maxLength": 1500},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "watch_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            },
            "required": ["reason", "confidence", "watch_conditions"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class AgentResult:
    response_id: str | None = None
    usage: Usage = field(default_factory=Usage)
    text: str = ""
    action_name: str | None = None
    action_payload: dict | None = None
    action_result: dict | None = None
    latency_ms: int = 0
    error: str | None = None


def _usage(response: Any) -> Usage:
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    input_details = getattr(u, "input_tokens_details", None)
    output_details = getattr(u, "output_tokens_details", None)
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        cached_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
    )


def _merge_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        cached_tokens=a.cached_tokens + b.cached_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
    )


def _message_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", ()) or ():
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", ()) or ():
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


class GrokAgent:
    def __init__(self, cfg: GrokConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.cfg.api_key,
                base_url=self.cfg.base_url,
                timeout=self.cfg.timeout_seconds,
                max_retries=1,
            )
        return self._client

    def run(self, *, trigger: str, terminal_summary: dict,
            dispatch: Callable[[str, dict], dict]) -> AgentResult:
        started = time.monotonic()
        total_usage = Usage()
        action_name: str | None = None
        action_payload: dict | None = None
        action_result: dict | None = None
        try:
            client = self._get_client()
            response = client.responses.create(
                model=self.cfg.model,
                reasoning={"effort": self.cfg.reasoning_effort},
                max_output_tokens=self.cfg.max_output_tokens,
                tools=TOOLS,
                parallel_tool_calls=False,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Decision trigger: {trigger}. Review the terminal state, "
                            "investigate as needed, then use exactly one action tool.\n\n"
                            + json.dumps(terminal_summary, separators=(",", ":"), sort_keys=True)
                        ),
                    },
                ],
            )
            total_usage = _merge_usage(total_usage, _usage(response))
            rounds = 0
            while rounds < self.cfg.max_tool_rounds:
                calls = [item for item in (getattr(response, "output", ()) or ())
                         if getattr(item, "type", None) == "function_call"]
                if not calls:
                    break
                outputs: list[dict] = []
                for call in calls:
                    name = str(getattr(call, "name", ""))
                    try:
                        args = json.loads(getattr(call, "arguments", "{}") or "{}")
                    except json.JSONDecodeError as exc:
                        args = {}
                        result = {"ok": False, "error": f"invalid_json_arguments:{exc}"}
                    else:
                        result = dispatch(name, args)
                    outputs.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, default=str, separators=(",", ":")),
                    })
                    if name in {"submit_paper_trade", "close_grok_position", "stand_down"}:
                        action_name = name
                        action_payload = args
                        action_result = result
                rounds += 1
                response = client.responses.create(
                    model=self.cfg.model,
                    reasoning={"effort": self.cfg.reasoning_effort},
                    max_output_tokens=self.cfg.max_output_tokens,
                    tools=TOOLS,
                    parallel_tool_calls=False,
                    previous_response_id=response.id,
                    input=outputs,
                )
                total_usage = _merge_usage(total_usage, _usage(response))
                if action_name is not None:
                    break
            return AgentResult(
                response_id=getattr(response, "id", None),
                usage=total_usage,
                text=_message_text(response),
                action_name=action_name,
                action_payload=action_payload,
                action_result=action_result,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            return AgentResult(
                usage=total_usage,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

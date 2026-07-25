"""HTTP decision client for SPY-DER POST /v1/decision.

No imports of spy_der internals. Fail closed to deterministic ABSTAIN when
the service is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from integrations.spy_der.contracts import (
    DECISION_REQUEST_SCHEMA,
    DECISION_RESPONSE_SCHEMA,
    PARALLEL_TRACK_ID,
    PARALLEL_TRACK_LABEL,
    build_decision_request,
    dashboard_to_parallel_payload,
    validate_dashboard_packet,
)

log = logging.getLogger("integrations.spy_der.decision_client")

DEFAULT_DECISION_URL = "http://127.0.0.1:8787/v1/decision"


class DecisionTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...


@dataclass(frozen=True)
class UrllibDecisionTransport:
    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"decision service unreachable: {exc}") from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("decision service returned non-object JSON")
        return data


@dataclass(frozen=True)
class DecisionResult:
    action: str
    candidate_id: Optional[str]
    size_scalar: float
    structure: Optional[str]
    direction: Optional[str]
    confidence: float
    uncertainty: float
    rationale: str
    reason_codes: tuple[str, ...]
    provider: str
    model_id: str
    available: bool
    raw: dict[str, Any]

    def as_parallel_payload(self) -> dict[str, Any]:
        if self.raw.get("schema_version"):
            return dashboard_to_parallel_payload(self.raw)
        return {
            "track": PARALLEL_TRACK_ID,
            "label": PARALLEL_TRACK_LABEL,
            "source": self.provider,
            "mode": "shadow",
            "action": self.action,
            "structure": self.structure,
            "direction": self.direction,
            "candidate_id": self.candidate_id,
            "size_cap": self.size_scalar,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "rationale": self.rationale,
            "reason_codes": list(self.reason_codes),
            "model_id": self.model_id,
            "available": self.available,
        }


def unavailable_result(reason: str, *, code: str = "spy_der_unavailable") -> DecisionResult:
    return DecisionResult(
        action="UNAVAILABLE",
        candidate_id=None,
        size_scalar=0.0,
        structure=None,
        direction=None,
        confidence=0.0,
        uncertainty=1.0,
        rationale=reason,
        reason_codes=(code,),
        provider="none",
        model_id="",
        available=False,
        raw={
            "action": "UNAVAILABLE",
            "provider": "none",
            "available": False,
            "rationale": reason,
            "reason_codes": [code],
            "confidence": 0.0,
            "uncertainty": 1.0,
            "size_scalar": 0.0,
        },
    )


def abstain_result(reason: str, *, code: str = "spy_der_error") -> DecisionResult:
    return DecisionResult(
        action="ABSTAIN",
        candidate_id=None,
        size_scalar=0.0,
        structure=None,
        direction=None,
        confidence=0.0,
        uncertainty=1.0,
        rationale=reason,
        reason_codes=(code,),
        provider="spy_der",
        model_id="",
        available=True,
        raw={
            "action": "ABSTAIN",
            "provider": "spy_der",
            "available": True,
            "rationale": reason,
            "reason_codes": [code],
            "confidence": 0.0,
            "uncertainty": 1.0,
            "size_scalar": 0.0,
        },
    )


def _result_from_dashboard(decision: dict[str, Any]) -> DecisionResult:
    validate_dashboard_packet(decision)
    return DecisionResult(
        action=str(decision.get("action") or "ABSTAIN"),
        candidate_id=(
            str(decision["candidate_id"])
            if decision.get("candidate_id") is not None
            else None
        ),
        size_scalar=float(decision.get("size_scalar") or 0.0),
        structure=decision.get("structure"),
        direction=decision.get("direction"),
        confidence=float(decision.get("confidence") or 0.0),
        uncertainty=float(decision.get("uncertainty") or 1.0),
        rationale=str(decision.get("rationale") or ""),
        reason_codes=tuple(str(c) for c in (decision.get("reason_codes") or [])),
        provider=str(decision.get("provider") or "spy_der"),
        model_id=str(decision.get("trader_model") or decision.get("model_id") or ""),
        available=bool(decision.get("available", True)),
        raw=decision,
    )


class DecisionClient:
    """POST MarketPacket → DashboardPacket with retry + deterministic fallback."""

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        timeout: float = 8.0,
        retries: int = 2,
        retry_backoff_s: float = 0.25,
        transport: Optional[DecisionTransport] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        env_url = os.environ.get("SPY_DER_DECISION_URL", "").strip()
        self.url = url or env_url or DEFAULT_DECISION_URL
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.retry_backoff_s = retry_backoff_s
        self.transport = transport or UrllibDecisionTransport()
        self._sleep = sleeper

    def decide(
        self,
        market: dict[str, Any],
        *,
        request_id: str = "",
    ) -> DecisionResult:
        req_id = request_id or uuid.uuid4().hex
        payload = build_decision_request(market, request_id=req_id)
        if payload.get("schema_version") != DECISION_REQUEST_SCHEMA:
            return abstain_result("bad_request_schema", code="spy_der_schema_error")

        last_exc: Optional[BaseException] = None
        attempts = self.retries + 1
        for attempt in range(attempts):
            try:
                raw = self.transport.post_json(self.url, payload, self.timeout)
                if raw.get("schema_version") != DECISION_RESPONSE_SCHEMA:
                    raise RuntimeError(
                        f"unexpected response schema {raw.get('schema_version')!r}"
                    )
                decision = raw.get("decision")
                if not isinstance(decision, dict):
                    raise RuntimeError("decision response missing decision object")
                return _result_from_dashboard(decision)
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "spy_der decision attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt + 1 < attempts:
                    self._sleep(self.retry_backoff_s * (attempt + 1))

        reason = f"decision_unavailable:{type(last_exc).__name__}:{last_exc}"
        log.error("spy_der decision fallback: %s", reason)
        return unavailable_result(reason, code="spy_der_http_unavailable")

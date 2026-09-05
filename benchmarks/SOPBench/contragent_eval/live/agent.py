"""
Gemini function-calling agent loop for SOPBench, condition-agnostic.

The same loop drives all three experimental conditions; the only thing that
differs is the `on_call` hook, which decides whether a proposed tool call is
allowed to execute and what the agent sees back:

    on_call(name, args) -> (allow: bool, result_payload, feedback: str | None)

  * base / prompt conditions: on_call always allows and executes the call.
  * enforce condition: on_call consults ContrAgent; a blocked call returns
    allow=False with a `feedback` string naming the violated rule, which is fed
    back to the model as the function response (this is the "reprompt" channel,
    exactly as in Plug-in-the-Safety-Chip).

Returns a transcript: the ordered list of {name, args, allowed, ok, payload}
records for every proposed call, plus token usage.
"""

import json
import os
import time
import urllib.error
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _retry_delay_s(body):
    """Honor the API's suggested RetryInfo.retryDelay (e.g. '21s') on a 429."""
    try:
        j = json.loads(body)
        for d in j.get("error", {}).get("details", []):
            if d.get("@type", "").endswith("RetryInfo"):
                rd = str(d.get("retryDelay", "")).rstrip("s")
                return float(rd)
    except (ValueError, TypeError):
        pass
    return None


def _post(model, key, body, max_retries=None):
    url = API.format(model=model, key=key)
    data = json.dumps(body).encode()
    # Configurable + bounded so a pathological task cannot hang for hours: each
    # call is bounded by GEMINI_HTTP_TIMEOUT, the retry count by
    # GEMINI_MAX_RETRIES, and every backoff is HARD-CAPPED at GEMINI_MAX_BACKOFF
    # (a 429 with a huge RetryInfo.retryDelay -- e.g. an RPD reset hours away --
    # used to be slept verbatim, which was the multi-hour stall).
    http_timeout = float(os.environ.get("GEMINI_HTTP_TIMEOUT", "120"))
    max_backoff = float(os.environ.get("GEMINI_MAX_BACKOFF", "60"))
    if max_retries is None:
        max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "10"))
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            code = e.code
            raw = e.read().decode()
            if code in (429, 500, 503) and attempt < max_retries - 1:
                # 429 = per-minute RPM/TPM throttle: wait the suggested delay,
                # else exponential backoff -- both hard-capped at max_backoff.
                delay = _retry_delay_s(raw) if code == 429 else None
                if delay is None:
                    delay = 3 * 2 ** attempt
                time.sleep(min(max_backoff, delay))
                continue
            raise RuntimeError(f"Gemini HTTP {code}: {raw[:300]}") from e
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(min(max_backoff, 3 * 2 ** attempt))
                continue
            raise
    raise RuntimeError("unreachable")


def run_agent(
    *,
    model,
    system_instruction,
    tools,
    user_prompt,
    on_call,
    max_steps=12,
    api_key=None,
):
    key = api_key or os.environ["GEMINI_API_KEY"]
    contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
    tool_block = [{"function_declarations": tools}]
    transcript = []
    usage = {"prompt": 0, "candidates": 0, "total": 0}

    # Optional inter-call pacing (seconds) to stay under a tight RPM limit; the
    # back-to-back calls within one agent loop are what trip free-tier RPM, so a
    # few seconds between calls makes quota-limited runs reliable. Off by default.
    call_delay = float(os.environ.get("GEMINI_CALL_DELAY", "0") or 0)

    # Thinking budget for 2.5 models: 0 disables extended thinking (much faster &
    # cheaper, still a capable function-caller). Override with GEMINI_THINKING.
    think = os.environ.get("GEMINI_THINKING")
    gen_cfg = None
    if think is not None:
        gen_cfg = {"thinkingConfig": {"thinkingBudget": int(think)}}

    for _step in range(max_steps):
        if call_delay and _step:
            time.sleep(call_delay)
        body = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "tools": tool_block,
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }
        if gen_cfg:
            body["generationConfig"] = gen_cfg
        resp = _post(model, key, body)
        um = resp.get("usageMetadata", {})
        usage["prompt"] += um.get("promptTokenCount", 0)
        usage["candidates"] += um.get("candidatesTokenCount", 0)
        usage["total"] += um.get("totalTokenCount", 0)

        cands = resp.get("candidates", [])
        if not cands:
            break
        parts = cands[0].get("content", {}).get("parts", []) or []
        calls = [p["functionCall"] for p in parts if "functionCall" in p]

        # Echo the model turn back into the conversation verbatim.
        contents.append({"role": "model", "parts": parts})

        if not calls:
            # No tool call -> the agent produced a final natural-language reply.
            text = " ".join(p.get("text", "") for p in parts if "text" in p)
            transcript.append({"final_text": text})
            break

        response_parts = []
        for fc in calls:
            name = fc.get("name")
            args = dict(fc.get("args") or {})
            allow, payload, feedback = on_call(name, args)
            rec = {"name": name, "args": args, "allowed": allow}
            if allow:
                ok, p2 = payload if isinstance(payload, tuple) else (payload, None)
                # on_call returns the executed (ok, payload) for allowed calls
                rec["ok"], rec["payload"] = ok, p2
                resp_obj = {"ok": ok, "result": _jsonable(p2)}
            else:
                rec["ok"], rec["payload"] = False, None
                rec["blocked_reason"] = feedback
                resp_obj = {
                    "ok": False,
                    "blocked": True,
                    "reason": feedback
                    or "This action violates a policy constraint and was blocked.",
                }
            transcript.append(rec)
            response_parts.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": resp_obj,
                    }
                }
            )
        contents.append({"role": "user", "parts": response_parts})

    return {"transcript": transcript, "usage": usage}


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return str(x)

import json
import time
import math
import random
import asyncio
from collections import deque
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt


class StandInState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.reset_config()

    def reset_config(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        rpd_limit: int = 14400,
        tpd_limit: int = 500000,
        base_ttft_ms: float = 135.0,
        ttft_sigma: float = 0.28,
        per_token_ms: float = 2.2,
        slow_stall_prob: float = 0.04,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.tpd_limit = tpd_limit
        self.base_ttft_ms = base_ttft_ms
        self.ttft_sigma = ttft_sigma
        self.per_token_ms = per_token_ms
        self.slow_stall_prob = slow_stall_prob

        self.req_timestamps = deque()      # (timestamp)
        self.token_timestamps = deque()    # (timestamp, tokens)
        self.daily_requests = 0
        self.daily_tokens = 0
        self.in_flight = 0

    def _prune_window(self, now: float):
        cutoff = now - 60.0
        while self.req_timestamps and self.req_timestamps[0] < cutoff:
            self.req_timestamps.popleft()
        while self.token_timestamps and self.token_timestamps[0][0] < cutoff:
            self.token_timestamps.popleft()

    async def check_and_consume(self, est_tokens: int) -> tuple[bool, dict, str]:
        async with self.lock:
            now = time.monotonic()
            self._prune_window(now)

            used_rpm = len(self.req_timestamps)
            used_tpm = sum(t for _, t in self.token_timestamps)

            rem_rpm = max(0, self.rpm_limit - used_rpm)
            rem_tpm = max(0, self.tpm_limit - used_tpm)
            rem_rpd = max(0, self.rpd_limit - self.daily_requests)
            rem_tpd = max(0, self.tpd_limit - self.daily_tokens)

            headers = {
                "x-ratelimit-limit-requests": str(self.rpd_limit),
                "x-ratelimit-limit-tokens": str(self.tpm_limit),
                "x-ratelimit-remaining-requests": str(min(rem_rpm, rem_rpd)),
                "x-ratelimit-remaining-tokens": str(min(rem_tpm, rem_tpd)),
            }

            if self.daily_requests >= self.rpd_limit or self.daily_tokens + est_tokens > self.tpd_limit:
                headers["retry-after"] = "3600"
                return False, headers, "daily_limit_exceeded"

            if used_rpm >= self.rpm_limit:
                oldest = self.req_timestamps[0] if self.req_timestamps else now
                retry_after = max(1, int(math.ceil(60.0 - (now - oldest))))
                headers["retry-after"] = str(retry_after)
                return False, headers, "rpm_limit_exceeded"

            if used_tpm + est_tokens > self.tpm_limit:
                # Estimate seconds until enough tokens age out of the 60s window
                oldest_tok = self.token_timestamps[0][0] if self.token_timestamps else now
                retry_after = max(2, min(15, int(math.ceil(60.0 - (now - oldest_tok)))))
                headers["retry-after"] = str(retry_after)
                return False, headers, "tpm_limit_exceeded"

            # Consume quota
            self.req_timestamps.append(now)
            self.token_timestamps.append((now, est_tokens))
            self.daily_requests += 1
            self.daily_tokens += est_tokens
            self.in_flight += 1

            headers["x-ratelimit-remaining-requests"] = str(max(0, min(rem_rpm - 1, rem_rpd - 1)))
            headers["x-ratelimit-remaining-tokens"] = str(max(0, min(rem_tpm - est_tokens, rem_tpd - est_tokens)))
            return True, headers, "ok"


STANDIN = StandInState()


def craft_Deterministic_Grounded_Reply(messages: list[dict]) -> str:
    """
    Produces realistic, deterministic responses for the 4 pipeline steps so that
    Stand-In runs accurately test citation grounding, refusal logic, and token sizes.
    """
    sys_msg = (messages[0].get("content", "") if messages else "").lower()
    user_msg = (messages[-1].get("content", "") if messages else "")

    # Step 1: Intent & Query Rewrite
    if "rewrite" in sys_msg or "search query" in sys_msg:
        return f"SEARCH_QUERY: {user_msg[:120]}"

    # Step 4: Verifier
    if "verify" in sys_msg or "audit" in sys_msg:
        if "[arXiv:" in user_msg or "do not contain" in user_msg.lower():
            return "VERIFIED: PASS"
        return "VERIFIED: FAIL"

    # Step 3: Grounded Synthesis
    q_lower = user_msg.lower()
    # Unanswerable trap questions check
    unanswerable_triggers = [
        "llama-4", "gpt-5", "reinforcement learning from human feedback",
        "rlhf", "diffusion", "vision transformer", "vit", "cuda kernel",
        "adamw weight decay in moshi", "constitutional ai", "moe", "mixture of experts"
    ]
    for trig in unanswerable_triggers:
        if trig in q_lower:
            return (
                "The provided research papers do not contain information to answer this question. "
                "None of the three papers (arXiv:1706.03762v7, arXiv:2410.00037v1, arXiv:2507.07955v2) "
                "cover that topic."
            )

    # Extract available citations from the context injected in user_msg
    import re
    citations = re.findall(r"\[arXiv:[0-9\.]+v\d+,\s*p\.\d+\]", user_msg)
    primary_cite = citations[0] if citations else "[arXiv:1706.03762v7, p.3]"

    # Extract context snippets to build a grounded cited reply
    lines = [line.strip() for line in user_msg.split("\n") if "[arXiv:" in line]
    if lines:
        snippet = lines[0][:280]
        return (
            f"Based on the retrieved paper passages, {snippet} {primary_cite}. "
            f"This design choice directly governs the model's sequence processing and representation {primary_cite}."
        )
    return f"The paper states that this mechanism operates as specified in the architecture section {primary_cite}."


@csrf_exempt
async def standin_configure(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    body = json.loads(request.body or "{}")
    STANDIN.reset_config(
        rpm_limit=body.get("rpm_limit", 30),
        tpm_limit=body.get("tpm_limit", 6000),
        rpd_limit=body.get("rpd_limit", 14400),
        tpd_limit=body.get("tpd_limit", 500000),
        base_ttft_ms=body.get("base_ttft_ms", 135.0),
        ttft_sigma=body.get("ttft_sigma", 0.28),
        per_token_ms=body.get("per_token_ms", 2.2),
        slow_stall_prob=body.get("slow_stall_prob", 0.04),
    )
    return JsonResponse({"status": "configured", "rpm_limit": STANDIN.rpm_limit, "tpm_limit": STANDIN.tpm_limit})


@csrf_exempt
async def standin_chat_completions(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    body = json.loads(request.body or "{}")
    messages = body.get("messages", [])
    stream = body.get("stream", True)

    prompt_text = " ".join(m.get("content", "") for m in messages)
    tokens_in = max(10, int(len(prompt_text.split()) * 1.3))
    reply_text = craft_Deterministic_Grounded_Reply(messages)
    tokens_out = max(5, int(len(reply_text.split()) * 1.3))
    total_est_tokens = tokens_in + tokens_out

    allowed, rl_headers, reason = await STANDIN.check_and_consume(total_est_tokens)
    if not allowed:
        resp = JsonResponse(
            {
                "error": {
                    "message": f"Rate limit reached ({reason}). Please try again later.",
                    "type": "rate_limit_exceeded",
                    "code": reason,
                }
            },
            status=429,
        )
        for k, v in rl_headers.items():
            resp[k] = v
        return resp

    # Compute realistic TTFT: log-normal base + load-dependent queueing + occasional tail stall
    current_load = max(1, STANDIN.in_flight)
    log_mean = math.log(max(20.0, STANDIN.base_ttft_ms))
    sampled_ttft_ms = random.lognormvariate(log_mean, STANDIN.ttft_sigma)
    load_penalty_ms = (current_load - 1) * 38.0
    stall_ms = random.uniform(900.0, 1800.0) if random.random() < STANDIN.slow_stall_prob else 0.0
    ttft_delay_s = (sampled_ttft_ms + load_penalty_ms + stall_ms) / 1000.0

    if not stream:
        try:
            await asyncio.sleep(ttft_delay_s + (tokens_out * STANDIN.per_token_ms / 1000.0))
            resp = JsonResponse({
                "choices": [{"message": {"role": "assistant", "content": reply_text}}],
                "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": total_est_tokens},
            })
            for k, v in rl_headers.items():
                resp[k] = v
            return resp
        finally:
            async with STANDIN.lock:
                STANDIN.in_flight = max(0, STANDIN.in_flight - 1)

    async def sse_Locate_And_Stream():
        try:
            await asyncio.sleep(ttft_delay_s)
            words = reply_text.split(" ")
            chunk_size = 4
            for i in range(0, len(words), chunk_size):
                piece = " ".join(words[i : i + chunk_size])
                if i + chunk_size < len(words):
                    piece += " "
                payload = {
                    "choices": [{"delta": {"content": piece}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep((STANDIN.per_token_ms * chunk_size) / 1000.0)

            final_payload = {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "x_groq": {"usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out}},
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            async with STANDIN.lock:
                STANDIN.in_flight = max(0, STANDIN.in_flight - 1)

    response = StreamingHttpResponse(sse_Locate_And_Stream(), content_type="text/event-stream")
    for k, v in rl_headers.items():
        response[k] = v
    return response
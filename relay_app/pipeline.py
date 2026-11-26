import re
import json
import time
import random
import asyncio
import httpx
from datetime import datetime, timezone
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from relay_app.retriever import PaperLibrary

REAL_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
STANDIN_URL = "http://127.0.0.1:8000/api/standin/v1/chat/completions"


class BudgetGovernor:
    """Tracks provider rate-limit headers and governs admission & degradation."""
    def __init__(self):
        self.rem_requests = 30
        self.rem_tokens = 6000
        self.daily_exhausted = False
        self.active_queue = 0
        self.peak_queue = 0

    def update_from_headers(self, headers: httpx.Headers):
        if "x-ratelimit-remaining-requests" in headers:
            try:
                self.rem_requests = int(headers["x-ratelimit-remaining-requests"])
            except ValueError:
                pass
        if "x-ratelimit-remaining-tokens" in headers:
            try:
                self.rem_tokens = int(headers["x-ratelimit-remaining-tokens"])
            except ValueError:
                pass

    def is_constrained(self) -> bool:
        return self.rem_tokens < 1800 or self.rem_requests < 8


GOVERNOR = BudgetGovernor()


class RelayPipeline:
    def __init__(self, client: httpx.AsyncClient, source: str = "stand-in", client_timeout_ms: float = 8500.0):
        self.client = client
        self.source = source
        self.endpoint = REAL_GROQ_URL if source == "real" else STANDIN_URL
        self.client_timeout_ms = client_timeout_ms
        self.library = PaperLibrary.get_instance()

    async def _call_llm(
        self,
        messages: list[dict],
        stream: bool,
        req_start_mono: float,
        max_retries: int = 1,
    ) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.source == "real":
            headers["Authorization"] = f"Bearer {settings.GROQ_API_KEY}"

        payload = {
            "model": settings.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 260,
            "stream": stream,
        }

        retries = 0
        retry_wait_ms = 0.0
        first_token_ms = None
        content_parts = []
        tokens_in = max(10, int( len(" ".join(m["content"] for m in messages).split()) * 1.3 ))
        tokens_out = 0

        while True:
            elapsed_ms = (time.perf_counter() - req_start_mono) * 1000.0
            if elapsed_ms >= self.client_timeout_ms:
                return {
                    "status": "timeout",
                    "text": "",
                    "first_token_ms": first_token_ms,
                    "retries": retries,
                    "retry_wait_ms": retry_wait_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }

            try:
                if stream:
                    async with self.client.stream(
                        "POST", self.endpoint, json=payload, headers=headers, timeout=8.0
                    ) as resp:
                        GOVERNOR.update_from_headers(resp.headers)
                        if resp.status_code == 429:
                            err_bytes = await resp.aread()
                            if b"daily_limit_exceeded" in err_bytes:
                                GOVERNOR.daily_exhausted = True
                            retry_after_s = float(resp.headers.get("retry-after", "2.0"))
                            wait_s = min(retry_after_s, 3.5) + random.uniform(0.05, 0.25)
                            now_elapsed = (time.perf_counter() - req_start_mono) * 1000.0
                            # Deadline-aware retry check
                            if retries < max_retries and (now_elapsed + wait_s * 1000.0 + 300.0) < self.client_timeout_ms and not GOVERNOR.daily_exhausted:
                                retries += 1
                                t_w0 = time.perf_counter()
                                await asyncio.sleep(wait_s)
                                retry_wait_ms += (time.perf_counter() - t_w0) * 1000.0
                                continue
                            return {
                                "status": "rate_limited",
                                "text": "",
                                "first_token_ms": first_token_ms,
                                "retries": retries,
                                "retry_wait_ms": retry_wait_ms,
                                "tokens_in": tokens_in,
                                "tokens_out": 0,
                            }

                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    if first_token_ms is None:
                                        first_token_ms = (time.perf_counter() - req_start_mono) * 1000.0
                                    content_parts.append(delta)
                                usage = chunk.get("x_groq", {}).get("usage") or chunk.get("usage")
                                if usage:
                                    tokens_in = usage.get("prompt_tokens", tokens_in)
                                    tokens_out = usage.get("completion_tokens", tokens_out)
                            except json.JSONDecodeError:
                                continue

                        full_text = "".join(content_parts).strip()
                        if tokens_out == 0:
                            tokens_out = max(1, int(len(full_text.split()) * 1.3))
                        return {
                            "status": "ok",
                            "text": full_text,
                            "first_token_ms": first_token_ms or ((time.perf_counter() - req_start_mono) * 1000.0),
                            "retries": retries,
                            "retry_wait_ms": retry_wait_ms,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        }
                else:
                    resp = await self.client.post(
                        self.endpoint, json=payload, headers=headers, timeout=8.0
                    )
                    GOVERNOR.update_from_headers(resp.headers)
                    if resp.status_code == 429:
                        if "daily_limit_exceeded" in resp.text:
                            GOVERNOR.daily_exhausted = True
                        retry_after_s = float(resp.headers.get("retry-after", "2.0"))
                        wait_s = min(retry_after_s, 3.5) + random.uniform(0.05, 0.25)
                        now_elapsed = (time.perf_counter() - req_start_mono) * 1000.0
                        if retries < max_retries and (now_elapsed + wait_s * 1000.0 + 300.0) < self.client_timeout_ms and not GOVERNOR.daily_exhausted:
                            retries += 1
                            t_w0 = time.perf_counter()
                            await asyncio.sleep(wait_s)
                            retry_wait_ms += (time.perf_counter() - t_w0) * 1000.0
                            continue
                        return {
                            "status": "rate_limited",
                            "text": "",
                            "first_token_ms": None,
                            "retries": retries,
                            "retry_wait_ms": retry_wait_ms,
                            "tokens_in": tokens_in,
                            "tokens_out": 0,
                        }

                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", tokens_in)
                    tokens_out = usage.get("completion_tokens", max(1, int(len(text.split()) * 1.3)))
                    return {
                        "status": "ok",
                        "text": text,
                        "first_token_ms": (time.perf_counter() - req_start_mono) * 1000.0,
                        "retries": retries,
                        "retry_wait_ms": retry_wait_ms,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                    }

            except httpx.TimeoutException:
                return {
                    "status": "timeout",
                    "text": "",
                    "first_token_ms": first_token_ms,
                    "retries": retries,
                    "retry_wait_ms": retry_wait_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }
            except Exception:
                return {
                    "status": "error",
                    "text": "",
                    "first_token_ms": first_token_ms,
                    "retries": retries,
                    "retry_wait_ms": retry_wait_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }

    async def execute(
        self,
        request_id: str,
        question: str,
        scenario: str = "steady",
        retrieval_setting: str = "500_5_yes",
        force_setting: bool = False,
    ) -> dict:
        start_wall = datetime.now(timezone.utc).isoformat()
        t_start = time.perf_counter()

        GOVERNOR.active_queue += 1
        GOVERNOR.peak_queue = max(GOVERNOR.peak_queue, GOVERNOR.active_queue)

        waiting_ms = 0.0
        retrying_ms = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        total_retries = 0

        try:
            # Admission pacing when queue is deep
            if GOVERNOR.active_queue > 8:
                pace_delay = min(0.35, (GOVERNOR.active_queue - 8) * 0.025)
                t_w = time.perf_counter()
                await asyncio.sleep(pace_delay)
                waiting_ms += (time.perf_counter() - t_w) * 1000.0

            # Adaptive retrieval step-down when approaching per-minute limits
            active_setting = retrieval_setting
            constrained = GOVERNOR.is_constrained()
            if constrained and not force_setting:
                active_setting = "250_3_no"

            # STEP 1: Intent & Query Decomposition (LLM Call 1 - bypassed in Constrained Mode)
            search_query = question
            if not constrained and not GOVERNOR.daily_exhausted:
                step1_msgs = [
                    {
                        "role": "system",
                        "content": "Rewrite the engineer's question into a concise keyword search query for arXiv papers (1706.03762v7, 2410.00037v1, 2507.07955v2). Output only the query.",
                    },
                    {"role": "user", "content": question},
                ]
                r1 = await self._call_llm(step1_msgs, stream=False, req_start_mono=t_start, max_retries=1)
                total_retries += r1["retries"]
                retrying_ms += r1["retry_wait_ms"]
                if r1["status"] == "ok":
                    total_tokens_in += r1["tokens_in"]
                    total_tokens_out += r1["tokens_out"]
                    search_query = f"{question} {r1['text']}"
                elif r1["status"] in ("rate_limited", "timeout") and GOVERNOR.daily_exhausted:
                    pass
                elif r1["status"] != "ok":
                    total_ms = (time.perf_counter() - t_start) * 1000.0
                    return {
                        "request_id": request_id,
                        "scenario": scenario,
                        "source": self.source,
                        "start_time": start_wall,
                        "first_token_ms": round(total_ms, 1),
                        "total_ms": round(total_ms, 1),
                        "status": r1["status"],
                        "retries": total_retries,
                        "tokens_in": total_tokens_in,
                        "tokens_out": total_tokens_out,
                        "retrieval_setting": active_setting,
                        "reply": "",
                        "waiting_ms": round(waiting_ms, 1),
                        "retrying_ms": round(retrying_ms, 1),
                        "working_ms": round(max(0.0, total_ms - waiting_ms - retrying_ms), 1),
                    }

            # STEP 2: Tool Call — Local Passage Retrieval
            passages, retrieval_ms = self.library.retrieve(search_query, setting_name=active_setting)

            # 3:00 PM Daily Budget Exhausted Fallback (Graceful Local Excerpt Reply)
            if GOVERNOR.daily_exhausted:
                excerpts = "\n".join(
                    f"- {p['citation']} ({p['title']}): \"{p['text'][:220]}...\""
                    for p in passages[:3]
                )
                fallback_reply = (
                    "DAILY_QUOTA_EXHAUSTED_FALLBACK: Daily AI synthesis budget reached (resets at 00:00 UTC). "
                    "To avoid guessing, here are the highest-ranking verbatim passages from the library:\n"
                    f"{excerpts}"
                )
                total_ms = (time.perf_counter() - t_start) * 1000.0
                return {
                    "request_id": request_id,
                    "scenario": scenario,
                    "source": self.source,
                    "start_time": start_wall,
                    "first_token_ms": round(total_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "status": "rate_limited",
                    "retries": total_retries,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "retrieval_setting": active_setting,
                    "reply": fallback_reply,
                    "waiting_ms": round(waiting_ms, 1),
                    "retrying_ms": round(retrying_ms, 1),
                    "working_ms": round(retrieval_ms, 1),
                }

            # STEP 3 & STEP 4 (with Conditional Verification Loop)
            max_loops = 1 if constrained else 2
            final_reply = ""
            first_token_ms = None
            final_status = "ok"

            for loop_idx in range(max_loops):
                context_block = "\n\n".join(
                    f"Passage {i+1} {p['citation']} ({p['title']}):\n{p['text']}"
                    for i, p in enumerate(passages)
                )
                synth_msgs = [
                    {
                        "role": "system",
                        "content": (
                            "You are an engineering research assistant answering strictly from the provided passages "
                            "of three papers: Attention Is All You Need (1706.03762v7), Moshi (2410.00037v1), and "
                            "Dynamic Chunking / H-Net (2507.07955v2). "
                            "Every factual claim MUST include an inline citation in the exact format [arXiv:ID, p.X]. "
                            "If the provided passages do not answer the question, you MUST explicitly state: "
                            "'The provided research papers do not contain information to answer this question.' "
                            "Never guess or extrapolate from outside memory."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Retrieved Passages:\n{context_block}\n\nQuestion: {question}",
                    },
                ]

                r3 = await self._call_llm(synth_msgs, stream=True, req_start_mono=t_start, max_retries=1)
                total_retries += r3["retries"]
                retrying_ms += r3["retry_wait_ms"]
                if first_token_ms is None and r3["first_token_ms"] is not None:
                    first_token_ms = r3["first_token_ms"]

                if r3["status"] != "ok":
                    final_status = r3["status"]
                    break

                total_tokens_in += r3["tokens_in"]
                total_tokens_out += r3["tokens_out"]
                final_reply = r3["text"]

                # STEP 4: Citation & Grounding Verification (Branch / Loop)
                has_valid_cite = bool(re.search(r"\[arXiv:(1706\.03762v7|2410\.00037v1|2507\.07955v2),\s*p\.\d+\]", final_reply))
                is_refusal = "do not contain" in final_reply.lower() or "does not say" in final_reply.lower() or "not mentioned" in final_reply.lower()

                if has_valid_cite or is_refusal:
                    break  # Verification passed!

                # If verification failed and we have a loop iteration left, broaden retrieval and re-synthesize
                if loop_idx + 1 < max_loops and not GOVERNOR.is_constrained():
                    passages, extra_ret_ms = self.library.retrieve(question, setting_name="1000_8_yes")
                    retrieval_ms += extra_ret_ms

            total_ms = (time.perf_counter() - t_start) * 1000.0
            if first_token_ms is None:
                first_token_ms = total_ms

            working_ms = max(0.0, total_ms - waiting_ms - retrying_ms)
            return {
                "request_id": request_id,
                "scenario": scenario,
                "source": self.source,
                "start_time": start_wall,
                "first_token_ms": round(first_token_ms, 1),
                "total_ms": round(total_ms, 1),
                "status": final_status,
                "retries": total_retries,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "retrieval_setting": active_setting,
                "reply": final_reply,
                "waiting_ms": round(waiting_ms, 1),
                "retrying_ms": round(retrying_ms, 1),
                "working_ms": round(working_ms, 1),
            }
        finally:
            GOVERNOR.active_queue = max(0, GOVERNOR.active_queue - 1)


@csrf_exempt
async def ask_relay_view(request):
    """Live Django ASGI endpoint for interactive testing and the live follow-up interview."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    body = json.loads(request.body or "{}")
    question = body.get("question", "What is the frame rate and frame size of Mimi in Moshi?")
    source = body.get("source", "stand-in")
    setting = body.get("retrieval_setting", "500_5_yes")

    async with httpx.AsyncClient() as client:
        pipeline = RelayPipeline(client=client, source=source)
        result = await pipeline.execute(
            request_id=f"live_{int(time.time()*1000)}",
            question=question,
            scenario="live",
            retrieval_setting=setting,
        )
    return JsonResponse(result)
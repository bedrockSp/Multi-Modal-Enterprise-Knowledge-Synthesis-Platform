import asyncio
import contextvars
import itertools
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone

from google import genai
from langchain_core.output_parsers import PydanticOutputParser
from openai import AsyncOpenAI

from core.config import settings
from core.constants import FALLBACK_GEMINI_MODEL, FALLBACK_OPENAI_MODEL, SWITCHES
from core.utils.llm_output_sanitizer import parse_llm_json, sanitize_llm_json

logger = logging.getLogger("llm.client")

# Directory for logging parse failures
_PARSE_ERRORS_DIR = "DEBUG/parse_errors"
os.makedirs(_PARSE_ERRORS_DIR, exist_ok=True)


def _log_parse_failure(
    source: str,
    attempt: int,
    raw_output: str,
    error: str,
    schema_name: str,
    prompt_snippet: str = "",
):
    """Log a parse failure to a JSONL file for later analysis."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "attempt": attempt,
        "schema": schema_name,
        "error": error,
        "raw_output": raw_output[:5000],
        "prompt_tail": prompt_snippet[-500:] if prompt_snippet else "",
    }
    try:
        log_path = os.path.join(_PARSE_ERRORS_DIR, "failures.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Don't let logging failures break the main flow

if SWITCHES["REMOTE_GPU"]:
    import core.llm.configurations.remote_llm as llm_module
elif SWITCHES.get("USE_VLLM", False):
    import core.llm.configurations.vllm_llm as llm_module
else:
    import core.llm.configurations.local_llm as llm_module

MyServerLLM = llm_module.MyServerLLM

# Always import INTERNALLLM so the class is available when the user
# toggles USE_INTERNAL on at runtime (the switch is checked at call time).
INTERNALLLM = None
try:
    from core.llm.configurations.INTERNAL_llm import INTERNALLLM
    print("INTERNALLLM imported successfully")
except ImportError as e:
    print(f"INTERNALLLM import failed: {e}. INTERNAL API will be unavailable.")
    INTERNALLLM = None
except Exception as e:
    print(f"Unexpected error importing INTERNALLLM: {e}. INTERNAL API will be unavailable.")
    INTERNALLLM = None

# ── Rate limiter for INTERNAL API (3 calls per minute) ─────────────


class _RateLimiter:
    """Async sliding-window rate limiter."""

    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < self.window]
            if len(self._timestamps) >= self.max_calls:
                wait_time = self.window - (now - self._timestamps[0])
                if wait_time > 0:
                    print(
                        f"[Rate limit] INTERNAL API: {self.max_calls} calls in last "
                        f"{self.window:.0f}s, waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    now = time.time()
                    self._timestamps = [
                        t for t in self._timestamps if now - t < self.window
                    ]
            self._timestamps.append(time.time())


_internal_rate_limiter = _RateLimiter(max_calls=3, window_seconds=60.0)

# ── Sticky fallback: once INTERNAL fails for a request, skip it for
#    all subsequent invoke_llm() calls in the same async context. ───

_skip_internal: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_skip_internal", default=False
)

# Cache LLM client instances to avoid repeated initialization overhead
_llm_cache = {}


def _get_cached_llm(model: str, port: int) -> MyServerLLM:
    """Return a cached MyServerLLM instance, creating one if needed."""
    key = (model, port)
    if key not in _llm_cache:
        _llm_cache[key] = MyServerLLM(model=model, port=port)
    return _llm_cache[key]


API_KEYS = [
    settings.API_KEY_1,
    settings.API_KEY_2,
    settings.API_KEY_3,
    settings.API_KEY_4,
    settings.API_KEY_5,
    settings.API_KEY_6,
]

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API)
MAX_RETRIES = 4  # Reduced from 8: JSON sanitizer + json_repair handles most parse errors on first attempt

# Thread-safe API key cycling
_api_key_cycle = itertools.cycle(API_KEYS)
_api_key_lock = asyncio.Lock()


async def _next_api_key():
    """Get the next API key in round-robin fashion, safely under concurrency."""
    async with _api_key_lock:
        return next(_api_key_cycle)


def _check_empty_lists(result, response_schema) -> None:
    """
    Reject outputs where ALL required list fields are empty.
    This triggers a self-correction retry instead of accepting useless data
    (Pattern 2: valid JSON structure but empty arrays).

    Only checks list fields that have NO default / default_factory (i.e. required).
    Fields like `attribution_warnings: List[str] = Field(default_factory=list)` are skipped.
    """
    from pydantic.fields import PydanticUndefined

    model_fields = getattr(response_schema, "model_fields", {})
    required_list_fields = []
    for name, info in model_fields.items():
        annotation = info.annotation
        # Check if it's a List type (typing.List or list with __origin__)
        origin = getattr(annotation, "__origin__", None)
        if origin is not list:
            continue
        # Skip fields with defaults or default_factory (optional lists)
        if info.default is not PydanticUndefined or info.default_factory is not None:
            continue
        required_list_fields.append(name)

    if not required_list_fields:
        return

    all_empty = all(
        len(getattr(result, f, None) or []) == 0 for f in required_list_fields
    )
    if all_empty:
        raise ValueError(
            f"All required list fields are empty ({', '.join(required_list_fields)}). "
            "Expected actual data items, not empty arrays."
        )


def _try_parse(raw_output: str, parser, response_schema):
    """
    Attempt to parse LLM output with sanitization and repair fallbacks.

    Strategy:
    1. Sanitize + PydanticOutputParser.parse() (existing path)
    2. Sanitize + json.loads + strip schema metadata + model_validate (fast path for schema-echo)
    3. parse_llm_json() with json_repair + model_validate (handles malformed JSON)

    Post-validation: reject outputs with all-empty list fields (Pattern 2).

    Returns parsed structured data or raises on failure.
    """
    cleaned = sanitize_llm_json(raw_output)

    # Strategy 1: Sanitized output through LangChain's parser
    try:
        result = parser.parse(cleaned)
        _check_empty_lists(result, response_schema)
        return result
    except Exception:
        pass

    # Strategy 2: Direct parse with schema metadata stripping
    # Handles Pattern 1 (schema + data mixed) without the overhead of json_repair
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            from core.utils.llm_output_sanitizer import _strip_schema_metadata
            parsed = _strip_schema_metadata(parsed, response_schema)
        result = response_schema.model_validate(parsed)
        _check_empty_lists(result, response_schema)
        return result
    except Exception:
        pass

    # Strategy 3: json_repair + Pydantic model_validate (handles malformed JSON)
    result = parse_llm_json(raw_output, response_schema)
    _check_empty_lists(result, response_schema)
    return result


def _serialize_prompt_messages(messages: list) -> str:
    """
    Convert a list of role/parts message dicts into a readable multi-section
    prompt string.  This preserves the intent of each section (system instructions,
    user question, etc.) rather than dumping a raw Python list repr.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "system").upper()
        content = msg.get("parts", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


async def invoke_llm(
    gpu_model,
    response_schema,
    contents,
    port=11434,
    remove_thinking=False,
):
    """
    Unified structured LLM invocation with retries and fallbacks:
    - GPU server
    - Gemini API
    - OpenAI API
    Each returns parsed structured data using the same logic.
    """

    # Initialize the parser for structured output
    parser = PydanticOutputParser(pydantic_object=response_schema)

    # Serialize contents properly — multi-turn role/parts dicts become readable
    # prompt sections instead of a raw Python list repr.
    if isinstance(contents, list) and contents and isinstance(contents[0], dict) and "role" in contents[0]:
        serialized = _serialize_prompt_messages(contents)
    else:
        serialized = str(contents)

    # Use different framing for answer-generating schemas vs pure extraction schemas.
    # Answer schemas need the LLM to generate rich content in the "answer" field;
    # "Extract structured data" framing causes short, terse outputs.
    is_answer_schema = hasattr(response_schema, "model_fields") and "answer" in response_schema.model_fields

    if is_answer_schema:
        prompt = f"""{serialized}

RESPONSE FORMAT — CRITICAL:
You MUST respond with a single valid JSON object matching this schema:
{parser.get_format_instructions()}

JSON RULES:
1. Output ONLY the JSON object — no markdown fences, no commentary, no text before or after.
2. Escape newlines as \\n and tabs as \\t within JSON string values.
3. If you use internal reasoning (e.g. <think> tags), produce the JSON AFTER the closing tag.
4. The "answer" field should contain your FULL, DETAILED response following the guidelines above. Do NOT truncate or shorten it.
5. For tables inside the answer field, use HTML <table> tags, NOT Markdown pipe tables.
6. Do NOT echo the schema definition. Never include "$defs", "$ref", "properties", "required", "title", "type":"object" or "description" as top-level keys. Only output the DATA that conforms to the schema.
"""
    else:
        prompt = f"""Extract structured data according to this model:
{parser.get_format_instructions()}

Input:
{serialized}

CRITICAL OUTPUT RULES:
1. Output must be valid JSON.
2. Escape newlines as \\n and tabs as \\t within JSON strings.
3. If you generate internal reasoning (e.g. inside <think> tags), you MUST produce the final JSON object AFTER the closing </think> tag.
4. Do not output any text before or after the JSON object.
5. Do NOT echo the schema definition. Never include "$defs", "$ref", "properties", "required", "title", "type":"object" or "description" as top-level keys. Only output the DATA that conforms to the schema.
6. Every list/array field must contain actual items. Do not return empty arrays unless the input data genuinely contains zero relevant items.
"""

    # ── Helper: build effective prompt with self-correction context ──
    def _build_prompt(base, failed_output, parse_error):
        if failed_output and parse_error:
            print("[Self-correction] Injecting previous output + error into prompt")
            return (
                f"{base}\n\n"
                "--- PREVIOUS ATTEMPT FAILED ---\n"
                "Your previous output could not be parsed. Fix the errors and output valid JSON only.\n\n"
                f"Previous output (rejected):\n{failed_output[:2000]}\n\n"
                f"Parse error:\n{parse_error}\n\n"
                "Fix the above errors and return ONLY valid JSON matching the schema."
            )
        return base

    # ── Phase 0: INTERNAL API (full retry cycle if enabled) ───────
    use_internal = (
        SWITCHES.get("USE_INTERNAL", False)
        and INTERNALLLM is not None
        and not _skip_internal.get()
    )

    if use_internal:
        last_failed_output = None
        last_parse_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n=== INTERNAL Attempt {attempt}/{MAX_RETRIES} ===")
            effective_prompt = _build_prompt(prompt, last_failed_output, last_parse_error)

            internal_output = None
            # Retry blank responses within the same attempt (up to 2 tries)
            for blank_retry in range(2):
                try:
                    if SWITCHES.get("RATE_LIMIT_INTERNAL", True):
                        await _internal_rate_limiter.acquire()
                    print("Trying INTERNAL API...")
                    internal_llm = INTERNALLLM(
                        model=settings.INTERNAL_MODEL_ID,
                        base_url=settings.INTERNAL_BASE_URL,
                        client_key=settings.INTERNAL_CLIENT_KEY,
                        api_token=settings.INTERNAL_API_TOKEN,
                        user_email=settings.INTERNAL_USER_EMAIL,
                    )
                    s = time.time()
                    internal_output = await asyncio.to_thread(internal_llm._call, effective_prompt)
                    e = time.time()
                    print(f"Success via INTERNAL API, LLM call took {e - s:.2f}s")

                    # Blank/empty response — retry immediately without burning the attempt
                    if not internal_output or not internal_output.strip():
                        print(f"[Blank response] INTERNAL returned empty output, retrying same attempt ({blank_retry + 1}/2)")
                        internal_output = None
                        continue

                    structured = _try_parse(internal_output, parser, response_schema)
                    return structured
                except Exception as exc:
                    error_str = str(exc)
                    tb_str = traceback.format_exc()
                    logger.error(
                        f"INTERNAL API attempt {attempt}/{MAX_RETRIES} failed "
                        f"(schema={response_schema.__name__}): {error_str}\n{tb_str}"
                    )
                    print(f"INTERNAL API failed: {error_str}")
                    if internal_output:
                        # Parse failure — retry with self-correction
                        last_failed_output = internal_output
                        last_parse_error = error_str
                        logger.error(
                            f"INTERNAL raw output ({len(internal_output)} chars): "
                            f"{internal_output[:1000]}"
                        )
                        _log_parse_failure(
                            source="internal",
                            attempt=attempt,
                            raw_output=internal_output,
                            error=error_str,
                            schema_name=response_schema.__name__,
                            prompt_snippet=effective_prompt if isinstance(effective_prompt, str) else str(effective_prompt),
                        )
                        print(f"[Self-correction] Captured failed INTERNAL output ({len(internal_output)} chars)")
                    else:
                        # Network/API error — no point retrying INTERNAL, break to GPU
                        logger.error(
                            f"INTERNAL network/API error (no output received): {error_str}\n{tb_str}"
                        )
                        print("[Sticky fallback] INTERNAL network error, breaking to GPU")
                    break  # Exit blank-retry loop on error
            else:
                # Both blank retries exhausted
                print(f"[Blank response] INTERNAL returned empty output twice, moving to next attempt")
                continue
            # If we broke out due to network error, also break the outer loop
            if not internal_output and not last_failed_output:
                break

        # All INTERNAL attempts exhausted (or network error) — sticky fallback
        print(
            f"[Sticky fallback] INTERNAL exhausted {MAX_RETRIES} attempts, "
            "switching to GPU for this request"
        )
        _skip_internal.set(True)

    # ── Phase 1: GPU SERVER (full retry cycle) ────────────────────
    last_failed_output = None
    last_parse_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n=== Attempt {attempt}/{MAX_RETRIES} ===")
        effective_prompt = _build_prompt(prompt, last_failed_output, last_parse_error)

        if gpu_model:
            llm_output = None
            # Retry blank responses within the same attempt (up to 2 tries)
            for blank_retry in range(2):
                try:
                    print("Trying GPU server...")
                    gpu_llm = _get_cached_llm(gpu_model, port)
                    s = time.time()
                    llm_output = await asyncio.to_thread(gpu_llm._call, effective_prompt)
                    e = time.time()
                    print(f"Success via GPU server, LLM call took {e - s:.2f}s")

                    # Blank/empty response — retry immediately without burning the attempt
                    if not llm_output or not llm_output.strip():
                        print(f"[Blank response] GPU returned empty output, retrying same attempt ({blank_retry + 1}/2)")
                        llm_output = None
                        continue

                    structured = _try_parse(llm_output, parser, response_schema)
                    return structured
                except Exception as e:
                    error_str = str(e)
                    print(f"GPU server failed at port {port}: {error_str}")
                    if llm_output:
                        last_failed_output = llm_output
                        last_parse_error = error_str
                        _log_parse_failure(
                            source="gpu",
                            attempt=attempt,
                            raw_output=llm_output,
                            error=error_str,
                            schema_name=response_schema.__name__,
                            prompt_snippet=effective_prompt if isinstance(effective_prompt, str) else str(effective_prompt),
                        )
                        print(f"[Self-correction] Captured failed GPU output ({len(llm_output)} chars)")
                    break  # Parse/network error — exit blank-retry loop, proceed to next attempt
            else:
                # Both blank retries exhausted — treat as a failed attempt
                print(f"[Blank response] GPU returned empty output twice, moving to next attempt")
            if last_failed_output:
                continue  # Retry with self-correction

        # === 2. GEMINI FALLBACK ===
        if SWITCHES["FALLBACK_TO_GEMINI"]:
            print("Falling back to Gemini...")

            for _ in range(len(API_KEYS)):
                api_key = await _next_api_key()
                client = genai.Client(api_key=api_key)
                s = time.time()
                try:
                    config = genai.types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=200000,
                        response_mime_type="text/plain",
                        safety_settings=[],
                    )

                    if remove_thinking:
                        config.thinking_config = genai.types.ThinkingConfig(
                            thinking_budget=0
                        )

                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.models.generate_content,
                            model=FALLBACK_GEMINI_MODEL,
                            contents=effective_prompt,
                            config=config,
                        ),
                        timeout=80,
                    )

                    # Try to extract the raw text content
                    raw_output = None
                    try:
                        raw_output = response.text or str(response)
                    except Exception:
                        raw_output = str(response)

                    structured = _try_parse(raw_output, parser, response_schema)
                    e = time.time()
                    print(f"Success via Gemini, LLM call took {e - s:.2f}s")
                    return structured

                except asyncio.TimeoutError:
                    print("Gemini timeout — switching key...")
                except Exception as e:
                    print(f"Gemini error: {e}")
                    if raw_output:
                        _log_parse_failure(
                            source="gemini",
                            attempt=attempt,
                            raw_output=raw_output,
                            error=str(e),
                            schema_name=response_schema.__name__,
                        )
                    await asyncio.sleep(0.2)

        # === 3. OPENAI FALLBACK ===
        if SWITCHES["FALLBACK_TO_OPENAI"]:
            openai_raw = None
            try:
                print("Falling back to OpenAI...")
                s = time.time()
                response = await openai_client.chat.completions.create(
                    model=FALLBACK_OPENAI_MODEL,
                    messages=[{"role": "user", "content": effective_prompt}],
                    temperature=0.2,
                )

                openai_raw = response.choices[0].message.content
                structured = _try_parse(openai_raw, parser, response_schema)
                e = time.time()
                print(f"Success via OpenAI, LLM call took {e - s:.2f}s")
                return structured

            except Exception as e:
                print(f"OpenAI fallback error: {e}")
                if openai_raw:
                    _log_parse_failure(
                        source="openai",
                        attempt=attempt,
                        raw_output=openai_raw,
                        error=str(e),
                        schema_name=response_schema.__name__,
                    )

        await asyncio.sleep(2)

    # If all attempts exhausted
    raise RuntimeError(f"All fallback attempts failed (INTERNAL + GPU + Gemini + OpenAI).")

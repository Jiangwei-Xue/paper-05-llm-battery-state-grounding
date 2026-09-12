# Qwen3.7-Plus E2b sensitivity protocol v1

This is a separately versioned, prospective sensitivity tier. It reuses the
byte-frozen E2b task substrate and the existing four carrier branches, adding
only the direct `qwen3.7-plus` model condition. Historical F1/F0 records,
denominators, prompts, parsers, and reports remain immutable and are not pooled
with this tier.

The formal plan contains 60 inherited F1 scenarios and 15 inherited F0
scenarios. Each block has C1, C2, S1, and S2 branches, giving 75 blocks and
300 planned logical provider calls. A separate two-block, eight-call smoke is
required first and is excluded from the formal denominator.

The inherited controls are retained: direct DashScope endpoint, no fallback,
no tools or web search, disabled Qwen thinking, temperature 0, JSON response
format, maximum three transport attempts (only 429/502/503/504 are retryable),
eight asynchronous workers, and exact prompt/request/response/parsed/score
evidence with SHA-256 hashes. C1/C2 and S1/S2 must have byte-identical request
payloads. Parser failures and transport failures remain visible in the formal
denominator; no output is repaired or retried for content reasons.

The model identity is frozen as `qwen3.7-plus` and every successful response
must report that exact model identifier. A mismatch is a protocol exclusion and
causes a fail-closed result; it is never silently accepted as an alias.

No provider call is permitted until the pre-call manifest passes. The formal
run must use a new empty run directory and cannot modify historical E2b/F0
artifacts.

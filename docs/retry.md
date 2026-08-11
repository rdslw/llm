(retry)=
# Retrying failed prompts

The `llm prompt` command can retry a failed request automatically. Some failures are transient - overloaded servers, rate limits, connection problems - and a second attempt a moment later often succeeds.

## The --cr/--chain-retry option

```bash
llm 'Ten names for cheesecakes' --cr 3
```
The value is the maximum total number of attempts:

- `--cr 1` (the default) - one attempt, no retry. This is the historical behavior.
- `--cr 3` - up to three attempts: the initial request plus two retries.
- `--cr 0` - retry without limit, until the request succeeds or a non-retryable condition stops it.

The option applies to `llm prompt` only. It works with `--async` and with tool calls (see the limitation below).

## Backoff

The delay before retry `n` (counting from 1) is:

    min(0.125 * 4 ** (n - 1), 600)

which produces 0.125s, 0.5s, 2s, 8s, 32s, 128s, 512s, then 600s for every following retry. `Ctrl+C` interrupts the wait immediately.

Each retry prints a notice to **stderr**, for example:

    Retrying (attempt 2/3) in 0.125s: Error code: 503 - {'error': {'message': 'Our servers are currently overloaded. ...'}}

Stdout stays clean, so pipes and shell substitutions receive only the model response.

## Which errors are retried

A retry happens only when the error looks transient. The check lives in `_is_retryable_error()` in `llm/cli.py`:

- `openai.APIConnectionError`, including timeouts (`APITimeoutError`).
- Any exception with a `status_code` of 408, 409, 429, or 500 and above.
- `llm.ModelError` whose message contains `overloaded`, `try again`, `rate limit`, or `temporarily` (case-insensitive). This catches plugins that surface server errors as plain `ModelError` with only a message string - for example `llm-openai-codex`, which converts in-stream `response.failed` events this way.

**Future change.** Raw HTTPX transport errors raised by model plugins are not covered yet. This includes `httpx.ConnectError`, `httpx.ReadTimeout`, and `httpx.RemoteProtocolError`. `httpx.HTTPStatusError` is also not covered because its status is available as `exception.response.status_code` rather than directly as `exception.status_code`. A future version should classify these the same way as their OpenAI equivalents.

Everything else - authentication failures, invalid options, schema errors, unknown models - fails immediately, exactly as before.

Two additional guards can cancel a retry even for a retryable error:

1. **Output already started.** If any response text reached stdout before the failure (a stream that died mid-reply), the command fails as before. A retry would duplicate the partial text in pipes. Reasoning output does not count - it goes to stderr and never blocks a retry.
2. **A tool chain already progressed.** See the next section.

## Interaction with logging

Nothing is written to `logs.db` until a request fully succeeds, and each attempt builds a fresh response object. Failed attempts are therefore never logged, and a successful retry produces exactly one log entry.

## Tool calls: current limitation and TBD

When a prompt uses tools, the request becomes a chain: model turn, tool execution, next model turn, and so on. Retry currently covers only the **first** request of that chain:

- Failure during the first model turn (even after long reasoning output): retried normally.
- Failure on a later turn, after at least one turn completed and its tools ran: **not retried**. The completed turn is already recorded on the conversation, and naively resending the original prompt would duplicate the user message on top of a history that ends in unresolved tool calls. The `len(conversation.responses)` guard in the retry loop detects this and fails fast instead.

**TBD - planned approach for mid-chain retry.** The codebase already contains a resume mechanism built for exactly this state: `ChainResponse.responses()` (`llm/models.py`, the `_pending_tool_calls` / `_resume_prompt` path) detects a history that ends in unresolved tool calls, re-executes those calls, and continues the chain. A future version of the retry loop can use it:

1. When the conversation grew during a failed attempt, re-call `conversation.chain(None, ...)` - no prompt text, attachments, or fragments (all already in history), same `tools` and `options`.
2. The resume path re-runs the trailing tool calls and continues from the crash point; completed turns are preserved and logged once.
3. Update the `responses_before` snapshot on every retry so later mid-chain failures can also resume.

Open questions before implementing this:

- Tool calls of the interrupted round would execute a **second time** (their first results were lost with the failed request). Acceptable for read-only tools; mutating tools would get duplicated side effects. Possibly needs an opt-in flag.
- Whether `system`/`schema` may be passed again on the resume call without duplication.
- Test coverage: a flaky model that emits tool calls in turn 1 and fails on turn 2.

Until then, mid-chain failures behave exactly as they did before this feature existed.

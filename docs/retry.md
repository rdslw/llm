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

The option applies to `llm prompt` only. It works with `--async` and with tool calls, as described below.

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

Two additional rules apply even for retryable errors:

1. **Output already started.** If any response text reached stdout before the failure (a stream that died mid-reply), the command fails as before. A retry would duplicate the partial text in pipes. Reasoning output does not count - it goes to stderr and never blocks a retry.
2. **Tool chains retry provider turns only.** A failed provider turn can be retried as described below. Failures after that turn completed, such as failures in local tool hooks, are not retried as provider requests.

## Interaction with logging

Nothing is written to `logs.db` until the prompt or tool chain succeeds. Failed provider attempts are never logged. A successful one-turn retry produces one turn in the log; a successful tool chain logs each of its completed provider turns exactly once.

## Tool calls and mid-chain retries

When a prompt uses local tools, the request becomes a chain: provider turn, local tool execution, next provider turn, and so on. A retryable failure can be retried on any provider turn as long as no response text from the chain reached stdout.

Later turns are retried without restarting the chain. The failed turn's `Prompt` contains the exact message history sent to the provider, including the results and attachments from local tools that already ran. Retry creates a new response from that same `Prompt`, preserves the chain's completed responses, and continues normally after success. It does **not** resend the original user prompt or execute those local tools again.

This applies when a provider turn emitted only reasoning before failing: reasoning is displayed on stderr, so the retry may display reasoning from more than one attempt, but stdout remains clean. If any text event reached stdout during an earlier turn or the failed turn, retry is refused to avoid duplicate output.

Only an incomplete provider response can use this path. A failure while executing a local tool or its `before_call`/`after_call` hook does not retry the provider turn that already completed.

Provider-managed tools run inside the provider request rather than in LLM's local tool chain. If a failed provider request already caused such a tool to have side effects, retry cannot guarantee exactly-once execution for that provider-managed operation.

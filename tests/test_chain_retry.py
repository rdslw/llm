"""Tests for the --cr/--chain-retry option: retry retryable failures
with exponential backoff, never retry anything else.
"""

import asyncio

import httpx
import openai
import pytest
from click.testing import CliRunner

import llm
from llm.cli import _is_retryable_error, _retry_delay, cli
from llm.parts import StreamEvent
from llm.plugins import pm

OVERLOADED = "Our servers are currently overloaded. Please try again later."


class FlakyModel(llm.Model):
    model_id = "flaky"
    can_stream = True
    supports_tools = True

    def __init__(self):
        self.calls = 0
        self.fail_times = 0
        self.exception = None
        self.partial_before_fail = False
        self.prompts = []

    def execute(self, prompt, stream, response, conversation):
        self.calls += 1
        if prompt.tools:
            self.prompts.append(prompt)
            if self.calls == 1:
                yield StreamEvent(type="reasoning", chunk="planning")
                response.add_tool_call(
                    llm.ToolCall(
                        name="retry_probe",
                        arguments={"value": "input"},
                        tool_call_id="tc_retry_probe",
                    )
                )
            elif self.calls <= self.fail_times + 1:
                raise llm.ModelError(OVERLOADED)
            else:
                yield "success"
            return
        if self.calls <= self.fail_times:
            if self.partial_before_fail:
                yield "partial "
            raise self.exception or llm.ModelError(OVERLOADED)
        yield "success"


class FlakyAsyncModel(llm.AsyncModel):
    model_id = "flaky"
    can_stream = True
    supports_tools = True

    def __init__(self):
        self.calls = 0
        self.fail_times = 0
        self.loops = []
        self.prompts = []

    async def execute(self, prompt, stream, response, conversation):
        self.calls += 1
        self.loops.append(asyncio.get_running_loop())
        if prompt.tools:
            self.prompts.append(prompt)
            if self.calls == 1:
                yield StreamEvent(type="reasoning", chunk="planning")
                response.add_tool_call(
                    llm.ToolCall(
                        name="retry_probe",
                        arguments={"value": "input"},
                        tool_call_id="tc_retry_probe",
                    )
                )
            elif self.calls <= self.fail_times + 1:
                raise llm.ModelError(OVERLOADED)
            else:
                yield "success"
            return
        if self.calls <= self.fail_times:
            raise llm.ModelError(OVERLOADED)
        yield "success"


@pytest.fixture
def flaky_model():
    model = FlakyModel()
    async_model = FlakyAsyncModel()
    model.async_model = async_model
    tool_executions = []
    model.tool_executions = async_model.tool_executions = tool_executions

    def retry_probe(value: str) -> str:
        tool_executions.append(value)
        return value.upper()

    class FlakyModelPlugin:
        __name__ = "FlakyModelPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(model, async_model=async_model)

        @llm.hookimpl
        def register_tools(self, register):
            register(retry_probe)

    pm.register(FlakyModelPlugin(), name="undo-FlakyModelPlugin")
    try:
        yield model
    finally:
        pm.unregister(name="undo-FlakyModelPlugin")


@pytest.fixture
def delays(monkeypatch):
    captured = []

    async def async_sleep(delay):
        captured.append(delay)

    monkeypatch.setattr("llm.cli.time.sleep", captured.append)
    monkeypatch.setattr("llm.cli.asyncio.sleep", async_sleep)
    return captured


def test_retry_then_succeed(flaky_model, delays):
    flaky_model.fail_times = 2
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["-m", "flaky", "hi", "--cr", "3", "--no-log"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "success" in result.stdout
    assert "partial" not in result.stdout
    assert flaky_model.calls == 3
    assert delays == [0.125, 0.5]
    assert "Retrying (attempt 2/3) in 0.125s" in result.stderr
    assert "Retrying (attempt 3/3) in 0.5s" in result.stderr


def test_non_retryable_error_not_retried(flaky_model, delays):
    flaky_model.fail_times = 1
    flaky_model.exception = llm.ModelError("schema validation broke")
    runner = CliRunner()
    with pytest.raises(llm.ModelError):
        runner.invoke(
            cli,
            ["-m", "flaky", "hi", "--cr", "5", "--no-log"],
            catch_exceptions=False,
        )
    assert flaky_model.calls == 1
    assert delays == []


def test_default_is_no_retry(flaky_model, delays):
    flaky_model.fail_times = 1
    runner = CliRunner()
    with pytest.raises(llm.ModelError):
        runner.invoke(
            cli,
            ["-m", "flaky", "hi", "--no-log"],
            catch_exceptions=False,
        )
    assert flaky_model.calls == 1
    assert delays == []


def test_partial_output_blocks_retry(flaky_model, delays):
    flaky_model.fail_times = 1
    flaky_model.partial_before_fail = True
    runner = CliRunner()
    with pytest.raises(llm.ModelError):
        runner.invoke(
            cli,
            ["-m", "flaky", "hi", "--cr", "3", "--no-log"],
            catch_exceptions=False,
        )
    assert flaky_model.calls == 1
    assert delays == []


def test_unlimited_retries(flaky_model, delays):
    flaky_model.fail_times = 4
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["-m", "flaky", "hi", "--cr", "0", "--no-log"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "success" in result.stdout
    assert flaky_model.calls == 5
    assert delays == [0.125, 0.5, 2, 8]
    assert "Retrying (attempt 2/unlimited)" in result.stderr


@pytest.mark.parametrize(
    "attempt,expected",
    ((1, 0.125), (7, 512), (8, 600), (513, 600)),
)
def test_retry_delay_is_capped_before_exponentiation(attempt, expected):
    assert _retry_delay(attempt) == expected


def test_async_retry_then_succeed(flaky_model, delays):
    async_model = flaky_model.async_model
    async_model.fail_times = 2
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["-m", "flaky", "hi", "--cr", "3", "--no-log", "--async"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "success" in result.stdout
    assert async_model.calls == 3
    assert delays == [0.125, 0.5]
    assert all(loop is async_model.loops[0] for loop in async_model.loops)


def test_failed_attempts_are_not_logged(flaky_model, delays, logs_db):
    flaky_model.fail_times = 2
    result = CliRunner().invoke(
        cli,
        ["-m", "flaky", "hi", "--cr", "3"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert flaky_model.calls == 3
    assert logs_db["turns"].count == 1


def tool_chain_args(async_, retries, *extra):
    args = ["-m", "flaky", "hi", "--cr", retries, "-T", "retry_probe", *extra]
    return args + (["--async"] if async_ else [])


@pytest.mark.parametrize("async_", (False, True))
def test_retry_failed_provider_turn_without_rerunning_tools(
    flaky_model, delays, logs_db, async_
):
    model = flaky_model.async_model if async_ else flaky_model
    model.fail_times = 2
    result = CliRunner().invoke(
        cli, tool_chain_args(async_, "3"), catch_exceptions=False
    )

    assert result.exit_code == 0
    assert result.stdout == "success\n"
    assert model.calls == 4
    assert all(prompt is model.prompts[1] for prompt in model.prompts[2:])
    assert model.tool_executions == ["input"]
    assert delays == [0.125, 0.5]
    assert logs_db["turns"].count == 2
    if async_:
        assert all(loop is model.loops[0] for loop in model.loops)


@pytest.mark.parametrize("async_", (False, True))
def test_non_streaming_does_not_retry_after_tools(flaky_model, delays, async_):
    model = flaky_model.async_model if async_ else flaky_model
    model.fail_times = 1
    result = CliRunner().invoke(
        cli, tool_chain_args(async_, "2", "--no-stream", "--no-log")
    )

    assert isinstance(result.exception, llm.ModelError)
    assert model.calls == 2
    assert model.tool_executions == ["input"]
    assert delays == []


@pytest.mark.parametrize("async_", (False, True))
def test_midchain_retry_preserves_chain_limit(flaky_model, delays, async_):
    model = flaky_model.async_model if async_ else flaky_model
    model.fail_times = 1
    result = CliRunner().invoke(
        cli, tool_chain_args(async_, "2", "--cl", "2", "--no-log")
    )

    assert result.exit_code == 1
    assert "Chain limit of 2 exceeded" in result.stderr
    assert model.calls == 3
    assert model.tool_executions == ["input"]
    assert delays == [0.125]


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "exception,expected",
    (
        (StatusError(429), True),
        (StatusError(503), True),
        (StatusError(400), False),
        (
            openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
            True,
        ),
        (llm.ModelError("Rate limit hit"), True),
        (llm.ModelError(OVERLOADED), True),
        (llm.ModelError("bad request"), False),
        (llm.NeedsKeyException("no key found"), False),
        (ValueError("overloaded"), False),
    ),
)
def test_is_retryable_error(exception, expected):
    assert _is_retryable_error(exception) is expected

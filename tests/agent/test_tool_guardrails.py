"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    TOOL_CALL_BRIDGE_NAME,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    bridge_inner_tool_names,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)




def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_gateway_platform_defaults_to_hard_stop_without_changing_interactive_defaults():
    interactive_configs = [
        ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        for platform in ("cli", "tui", "desktop", "acp")
    ]
    telegram_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="telegram")
    cron_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cron")

    assert all(cfg.hard_stop_enabled is False for cfg in interactive_configs)
    assert telegram_cfg.hard_stop_enabled is True
    assert cron_cfg.hard_stop_enabled is True


def test_non_interactive_hard_stop_can_be_disabled_explicitly():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"non_interactive_hard_stop_enabled": False},
        platform="telegram",
    )

    assert cfg.hard_stop_enabled is False
    assert cfg.non_interactive_hard_stop_enabled is False


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_skill_read_tools_are_idempotent_and_block_repeated_identical_success_output():
    cases = [
        (
            "skill_view",
            {"name": "gui-agent-ml-operations"},
            '{"success":true,"name":"gui-agent-ml-operations","content":"same"}',
        ),
        (
            "skills_list",
            {"category": "mlops"},
            '{"success":true,"skills":[{"name":"gui-agent-ml-operations"}]}',
        ),
    ]

    for tool_name, args, result in cases:
        controller = ToolCallGuardrailController(
            ToolCallGuardrailConfig(
                hard_stop_enabled=True,
                no_progress_warn_after=2,
                no_progress_block_after=2,
            )
        )

        assert controller.before_call(tool_name, args).action == "allow"
        assert controller.after_call(tool_name, args, result, failed=False).action == "allow"
        assert controller.before_call(tool_name, args).action == "allow"
        warn = controller.after_call(tool_name, args, result, failed=False)
        assert warn.action == "warn"
        assert warn.code == "idempotent_no_progress_warning"

        blocked = controller.before_call(tool_name, args)
        assert blocked.action == "block"
        assert blocked.code == "idempotent_no_progress_block"


def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"


def test_identical_call_streak_halts_any_tool_when_hard_stop_enabled():
    # #89069 / #100849 bundle: a model replaying the same SUCCESSFUL
    # terminal/skill_view call with a byte-identical result is not covered by
    # the idempotent_tools no-progress block. The consecutive-identical
    # streak (observe_call) is tool-agnostic; under hard_stop it must halt.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=5)
    )
    args = {"command": "hermes config get memory.provider"}
    for i in range(1, 5):
        controller.after_call("terminal", args, "local\n", failed=False)
        controller.observe_call("terminal", args, "local\n", failed=False)
        assert controller.halt_decision is None, f"halted early at {i}"

    controller.after_call("terminal", args, "local\n", failed=False)
    controller.observe_call("terminal", args, "local\n", failed=False)
    halt = controller.halt_decision
    assert halt is not None and halt.should_halt
    assert halt.code == "identical_call_streak_halt"
    assert halt.tool_name == "terminal" and halt.count == 5


def test_identical_call_streak_never_halts_when_hard_stop_disabled_or_for_pollers():
    soft = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False, no_progress_block_after=2)
    )
    for _ in range(6):
        soft.observe_call("terminal", {"command": "ls"}, "a\nb\n", failed=False)
    assert soft.halt_decision is None  # notice-only in interactive sessions

    hard = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2)
    )
    for _ in range(6):
        hard.observe_call("process_manage", {"action": "poll", "session_id": "p1"}, "running", failed=False)
    assert hard.halt_decision is None  # an unchanged poll is legitimate progress

    # A changed result resets the streak.
    for i in range(6):
        hard.observe_call("terminal", {"command": "date"}, f"t{i}", failed=False)
    assert hard.halt_decision is None






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == LoopCapConfig().max_web_searches
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == LoopCapConfig().max_subagents


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True












# ── Legitimate flows must survive hard stops (Teknium, Sep 2026) ────────────
# Hard stops default ON for unattended platforms. These pin the flows that
# must NEVER be cut off there: edit -> re-run loops, diagnostic sweeps of
# distinct red commands, and browser retry-after-action — while the pure
# replay (same call, nothing changed between attempts) is still stopped.

_HARD = lambda: ToolCallGuardrailController(
    ToolCallGuardrailConfig(hard_stop_enabled=True)
)
_PYTEST = {"command": "pytest tests/test_x.py -q"}
_RED = '{"output": "1 failed", "exit_code": 1}'


def _run_red(c, args=_PYTEST):
    assert c.before_call("terminal", args).allows_execution
    return c.after_call("terminal", args, _RED, failed=True)


def test_fix_retest_loop_is_never_hard_stopped():
    c = _HARD()
    for i in range(12):
        d = _run_red(c)
        assert not d.should_halt, f"halted on red run {i + 1}"
        # the model edits between runs — a landed mutation is progress
        c.after_call("patch", {"path": "x.py", "old_string": "a", "new_string": f"b{i}"},
                     '{"success": true, "diff": "..."}', failed=False)
    assert c.halt_decision is None
    assert c.before_call("terminal", _PYTEST).allows_execution


def test_pure_replay_with_no_intervening_change_is_still_blocked():
    c = _HARD()
    for _ in range(5):
        _run_red(c)
    d = c.before_call("terminal", _PYTEST)
    assert d.action == "block" and d.code == "repeated_exact_failure_block"


def test_intervening_mutation_resets_the_replay_streak_only_once():
    # 4 reds, one edit, then 4 reds with NO edit: the second run of 4 is a
    # fresh streak, and the 5th unchanged retry after it is blocked.
    c = _HARD()
    for _ in range(4):
        _run_red(c)
    c.after_call("write_file", {"path": "x.py", "content": "y"}, '{"bytes_written": 1}', failed=False)
    for _ in range(5):
        assert c.before_call("terminal", _PYTEST).allows_execution
        c.after_call("terminal", _PYTEST, _RED, failed=True)
    assert c.before_call("terminal", _PYTEST).action == "block"


def test_distinct_failing_terminal_commands_warn_but_never_halt():
    # A diagnostic sweep: grep with no matches, missing binaries, red builds.
    c = _HARD()
    for i in range(12):
        args = {"command": f"grep -q needle{i} haystack.txt"}
        d = c.after_call("terminal", args, _RED, failed=True)
        assert not d.should_halt, f"same_tool halt on distinct command #{i + 1}"
    assert c.halt_decision is None
    # ...while a non-tolerant tool failing 8 distinct ways still halts.
    c2 = _HARD()
    last = None
    for i in range(8):
        last = c2.after_call("send_message", {"to": f"u{i}"}, '{"error": "no route"}', failed=True)
    assert last.should_halt and last.code == "same_tool_failure_halt"


def test_browser_retry_after_action_is_not_a_replay():
    c = _HARD()
    nav = {"url": "https://example.test/app"}
    for _ in range(8):
        assert c.before_call("browser_navigate", nav).allows_execution
        c.after_call("browser_navigate", nav, '{"error": "timeout"}', failed=True)
        c.after_call("browser_click", {"selector": "#retry"}, '{"ok": true}', failed=False)
    assert c.halt_decision is None


def test_supervised_task_platforms_keep_warning_only_default():
    for platform in ("subagent", "api_server", "cli"):
        cfg = ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        assert cfg.hard_stop_enabled is False, platform
    for platform in ("telegram", "discord", "cron", "kanban"):
        cfg = ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        assert cfg.hard_stop_enabled is True, platform


def test_harness_refusals_are_not_tool_failures_on_either_classifier(tmp_path):
    """Every loop refusal the file tools emit (read dedup block, consecutive-read block,
    repeated-search block) carries `"error"` for the model's benefit -- exactly what the
    substring tests key on. Neither `classify_tool_failure` nor the executor's live seam
    `_detect_tool_failure` may count them, or the cheap refusal feeds the streak that
    fires `repeated_exact_failure_block` over calls that never failed."""
    from agent.display import _detect_tool_failure
    from tools.file_tools import _dedup_stub_or_block, read_file_tool, search_tool

    target = tmp_path / "responses.ts"
    target.write_text("export const x = 1;\n" * 30, encoding="utf-8")

    task = {"dedup_hits": {}}
    for _ in range(3):
        dedup_block = _dedup_stub_or_block(task, (str(target), 1, 999), str(target))
    consecutive_block = [read_file_tool(str(target), offset=1, limit=5, task_id="t-read") for _ in range(4)][-1]
    search_block = [search_tool("const x", path=str(tmp_path), task_id="t-search") for _ in range(4)][-1]

    for refusal in (dedup_block, consecutive_block, search_block):
        assert json.loads(refusal)["error"].startswith("BLOCKED"), refusal
        assert classify_tool_failure("read_file", refusal) == (False, ""), refusal
        assert _detect_tool_failure("read_file", refusal) == (False, ""), refusal


def test_a_real_tool_error_is_still_a_failure():
    """The exemption is keyed on the marker, not on the word: a body that
    genuinely failed still counts, or the streak that stops a real loop is gone."""
    from agent.display import _detect_tool_failure

    real = '{"error": "ENOENT: no such file"}'
    assert classify_tool_failure("read_file", real)[0] is True
    assert _detect_tool_failure("read_file", real)[0] is True
    # The marker is only honoured as the literal boolean, never as truthy prose.
    assert classify_tool_failure("read_file", '{"error": "x", "guardrail_refusal": "yes"}')[0] is True


def test_identical_streak_ignores_volatile_execution_metadata():
    # execute_code results carry per-call metadata (execution_count, duration_seconds) that
    # changes on every run. Hashing it made each empty replay look "new", so a model re-ran
    # the same empty probe 147 times without the identical-call halt ever firing.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=5)
    )
    args = {"code": "import subprocess\nprint(subprocess.run(['true']).returncode) if False else None"}

    def result(n):
        return json.dumps({
            "status": "success", "output": "", "exit_code": 0, "tool_calls_made": 0,
            "duration_seconds": 1.0 + n / 100,
            "kernel": {"mode": "session", "reused": True, "execution_count": 100 + n, "state_reset": False},
            "stdout_truncated": False, "stdout_bytes_captured": 0,
        })

    for i in range(4):
        controller.observe_call("execute_code", args, result(i), failed=False)
        assert controller.halt_decision is None, f"halted early at {i}"
    controller.observe_call("execute_code", args, result(4), failed=False)
    halt = controller.halt_decision
    assert halt is not None, "volatile metadata defeated the identical-call streak"
    assert halt.code == "identical_call_streak_halt"


def test_identical_streak_still_resets_when_real_output_changes():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=3)
    )
    args = {"code": "print(1)"}
    for i in range(6):
        out = json.dumps({"status": "success", "output": f"line {i}", "duration_seconds": 1.0,
                          "kernel": {"execution_count": i}})
        controller.observe_call("execute_code", args, out, failed=False)
    assert controller.halt_decision is None, "different real output must not count as a replay"


def test_other_tools_keep_duration_and_execution_count_as_real_output():
    # Only execute_code's known metadata locations are volatile. For any other tool these keys
    # can be the actual answer (e.g. a job-status tool reporting how long a job ran), so results
    # that differ only there are different results and must not form an identical streak.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=3)
    )
    args = {"job": "build-42"}
    for i in range(6):
        out = json.dumps({"status": "done", "duration_seconds": 10 + i,
                          "stats": {"execution_count": 100 + i}})
        controller.observe_call("mcp_ci_job_status", args, out, failed=False)
    assert controller.halt_decision is None, "a real change in duration_seconds/execution_count was treated as a replay"


def test_execute_code_replay_streak_notice_fires_on_warn_only_desktop_config():
    # #124072: on an interactive surface hard stops are off, so the appended notice is
    # the only signal the model gets. 186 no-op print("...") cells whose results differed
    # only in kernel.execution_count / duration_seconds produced zero notices.
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping({}, platform="desktop"))
    args = {"code": 'print("...")'}

    def result(n):
        return json.dumps({
            "status": "success", "output": "...\n", "exit_code": 0, "tool_calls_made": 0,
            "duration_seconds": 0.001 * n,
            "kernel": {"mode": "session", "reused": True, "execution_count": n, "state_reset": False},
            "stdout_truncated": False, "stdout_bytes_captured": 4, "stdout_bytes_total": 4,
            "stdout_bytes_omitted": 0,
        })

    notices = [
        controller.observe_call("execute_code", args, result(i), tool_call_id=f"c{i}").notice
        for i in range(1, 7)
    ]
    assert notices[:2] == [None, None]
    assert all(n is not None and "consecutive identical call to execute_code" in n for n in notices[2:]), notices
    assert controller.halt_decision is None, "warn-only surfaces must not halt"


# --- tool_call bridge: same-tool failures key on the inner tool; per-turn total cap ----------------


def _bridge_call(name, **arguments):
    """Arguments of one ``tool_call`` bridge call as the batch expander emits them."""
    return {"calls": [{"name": name, "arguments": arguments}]}


def _fail(controller, tool_name, args):
    return controller.after_call(tool_name, args, '{"error":"boom"}', failed=True)


def test_bridge_name_matches_the_tool_search_catalog():
    from tools.tool_search_catalog import TOOL_CALL_NAME

    assert TOOL_CALL_BRIDGE_NAME == TOOL_CALL_NAME


def test_bridge_inner_tool_names_reads_calls_entries_in_order_and_rejects_unreadable_batches():
    batch = {"calls": [{"name": "mcp_a", "arguments": {}}, {"name": "mcp_b"}, {"name": "mcp_a"}, {"name": " "}]}
    assert bridge_inner_tool_names(batch) == ("mcp_a", "mcp_b")
    assert bridge_inner_tool_names({"calls": json.dumps([{"name": "mcp_c", "arguments": {}}])}) == ("mcp_c",)
    assert bridge_inner_tool_names({"calls": "not a batch"}) == ()
    assert bridge_inner_tool_names({"calls": [{"arguments": {}}, "mcp_x", None]}) == ()
    assert bridge_inner_tool_names({"query": "no calls key"}) == ()
    assert bridge_inner_tool_names(None) == ()


def test_eight_failures_of_one_inner_tool_through_the_bridge_halt_naming_the_inner_tool():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))

    decisions = [_fail(controller, "tool_call", _bridge_call("mcp_gsc_query", page=i)) for i in range(8)]

    assert all(not d.should_halt for d in decisions[:7])
    assert decisions[2].code == "same_tool_failure_warning"
    assert decisions[2].tool_name == "mcp_gsc_query"
    assert "mcp_gsc_query has failed 3 times" in decisions[2].message
    halt = decisions[7]
    assert halt.action == "halt"
    assert halt.code == "same_tool_failure_halt"
    assert halt.tool_name == "mcp_gsc_query"
    assert halt.count == 8
    assert "Stopped mcp_gsc_query" in halt.message
    assert controller.halt_decision is halt


def test_eight_failures_of_eight_different_inner_tools_through_the_bridge_do_not_halt():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))

    decisions = [_fail(controller, "tool_call", _bridge_call(f"mcp_tool_{i}", q=i)) for i in range(8)]

    assert [d.action for d in decisions] == ["allow"] * 8
    assert controller.halt_decision is None


def test_sixteen_failures_of_sixteen_different_inner_tools_through_the_bridge_halt_on_the_total_cap():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))

    decisions = [_fail(controller, "tool_call", _bridge_call(f"mcp_tool_{i}", q=i)) for i in range(16)]

    assert all(not d.should_halt for d in decisions[:15])
    halt = decisions[15]
    assert halt.action == "halt"
    assert halt.code == "total_tool_failure_halt"
    assert halt.tool_name == "mcp_tool_15"
    assert halt.count == 16
    assert "16 tool calls failed this turn" in halt.message
    assert controller.halt_decision is halt


def test_direct_calls_keep_the_same_tool_halt_at_eight():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))

    decisions = [_fail(controller, "web_search", {"query": str(i)}) for i in range(8)]

    assert all(not d.should_halt for d in decisions[:7])
    assert decisions[7].code == "same_tool_failure_halt"
    assert decisions[7].tool_name == "web_search"
    assert decisions[7].count == 8


def test_direct_calls_of_different_tools_do_not_halt_at_eight_but_do_at_the_total_cap():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))

    decisions = [_fail(controller, f"tool_{i}", {"n": i}) for i in range(16)]

    assert [d.action for d in decisions[:8]] == ["allow"] * 8
    assert all(not d.should_halt for d in decisions[:15])
    assert decisions[15].code == "total_tool_failure_halt"
    assert decisions[15].count == 16


def test_total_cap_is_a_setting_skips_failure_tolerant_tools_and_resets_on_progress():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, total_tool_failure_halt_after=4)
    )

    # Distinct red terminal commands are diagnosis: never a halt, never counted toward the total.
    for i in range(10):
        assert not _fail(controller, "terminal", {"command": f"cmd-{i}"}).should_halt
    for i in range(3):
        assert not _fail(controller, f"tool_{i}", {"n": i}).should_halt
    assert controller.halt_decision is None

    landed = controller.after_call("patch", {"path": "a.py"}, '{"success": true}', failed=False)
    assert landed.action == "allow"

    for i in range(3):
        assert not _fail(controller, f"other_{i}", {"n": i}).should_halt
    halt = _fail(controller, "other_3", {"n": 3})
    assert halt.code == "total_tool_failure_halt"
    assert halt.count == 4


def test_total_cap_never_halts_with_hard_stops_disabled():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))

    decisions = [_fail(controller, "tool_call", _bridge_call(f"mcp_tool_{i}", q=i)) for i in range(20)]

    assert all(d.action == "allow" for d in decisions)
    assert controller.halt_decision is None


def test_bridge_success_clears_only_the_inner_tool_it_names():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for i in range(5):
        _fail(controller, "tool_call", _bridge_call("mcp_x", q=i))
        _fail(controller, "tool_call", _bridge_call("mcp_y", q=i))

    ok = controller.after_call("tool_call", _bridge_call("mcp_x", q=99), '{"ok": true}', failed=False)
    assert ok.action == "allow"

    assert _fail(controller, "tool_call", _bridge_call("mcp_x", q=100)).action == "allow"
    for i in range(5, 7):
        assert not _fail(controller, "tool_call", _bridge_call("mcp_y", q=i)).should_halt
    halt = _fail(controller, "tool_call", _bridge_call("mcp_y", q=7))
    assert halt.code == "same_tool_failure_halt"
    assert halt.tool_name == "mcp_y"
    assert halt.count == 8


def test_bridge_batch_without_readable_inner_names_counts_against_the_outer_name():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))

    decisions = [_fail(controller, "tool_call", {"calls": "garbage", "attempt": i}) for i in range(8)]

    assert decisions[7].code == "same_tool_failure_halt"
    assert decisions[7].tool_name == "tool_call"


def test_multi_entry_bridge_batch_counts_every_inner_tool_once_per_call():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    batch = {"calls": [{"name": "conn_a", "arguments": {}}, {"name": "conn_b", "arguments": {}}]}

    decisions = [_fail(controller, "tool_call", {**batch, "attempt": i}) for i in range(8)]

    halt = decisions[7]
    assert halt.code == "same_tool_failure_halt"
    assert halt.tool_name == "conn_a"
    assert halt.count == 8


def test_from_mapping_reads_total_tool_failure_nested_and_flat_with_default_sixteen():
    assert ToolCallGuardrailConfig().total_tool_failure_halt_after == 16
    nested = ToolCallGuardrailConfig.from_mapping({"hard_stop_after": {"total_tool_failure": 20}})
    assert nested.total_tool_failure_halt_after == 20
    assert nested.same_tool_failure_halt_after == 8
    flat = ToolCallGuardrailConfig.from_mapping({"total_tool_failure_halt_after": 12})
    assert flat.total_tool_failure_halt_after == 12

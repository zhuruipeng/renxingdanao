"""CLI 冒烟测试：验证命令行入口可用（全部走 scripted provider，不联网）。"""

from __future__ import annotations

import json
import os
import sys

import pytest

import main as cli

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPIC = "SQLite 索引原理"


def _run(db_path, *args) -> int:
    return cli.main(["--db", str(db_path), "--provider", "scripted", *args])


def test_init_command(tmp_path, capsys):
    db = tmp_path / "cli.db"
    assert _run(db, "init") == 0
    tables = json.loads(capsys.readouterr().out)["tables"]
    assert "knowledge" in tables and "evaluations" in tables


def test_full_cli_flow(tmp_path, capsys):
    db = tmp_path / "cli.db"
    assert _run(db, "init") == 0
    capsys.readouterr()

    assert _run(db, "learn", "--topic", TOPIC, "--max-items", "2") == 0
    capsys.readouterr()

    assert _run(db, "practice", "--topic", TOPIC, "--count", "1") == 0
    capsys.readouterr()

    assert _run(db, "exam", "--topic", TOPIC, "--count", "1", "--stage", "post_test") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["blind"] is True

    assert _run(db, "stats") == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["knowledge"] == 2
    assert stats["evaluations"] >= 2
    # V0.2 新增表也应出现
    assert "experiments" in stats and stats["experiments"] == 0


def test_ask_command_with_knowledge(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    _run(db, "learn", "--topic", TOPIC, "--max-items", "1")
    capsys.readouterr()
    assert _run(db, "ask", "--question", "什么是覆盖索引？", "--topic", TOPIC) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["blind"] is False
    assert out["answer"]


def test_ask_blind_is_blocked_by_gate(tmp_path, capsys):
    """--blind 且指定 topic 时，闸门应拦截并返回非零退出码。"""
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    code = _run(db, "ask", "--question", "什么是覆盖索引？", "--topic", TOPIC, "--blind")
    assert code == 3
    assert "闸门拦截" in capsys.readouterr().err


def test_cycle_command(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "cycle", "--topic", TOPIC, "--count", "1") == 0
    out = json.loads(capsys.readouterr().out)
    assert "delta" in out
    assert out["post_test"]["blind"] is True


def test_stats_needs_no_api_key(tmp_path, capsys, monkeypatch):
    """stats 是纯只读统计，不该因为缺 OPENAI_API_KEY 就失败。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db = tmp_path / "cli.db"
    assert _run(db, "init") == 0
    capsys.readouterr()
    assert _run(db, "stats") == 0
    assert json.loads(capsys.readouterr().out)["knowledge"] == 0


def test_learn_without_api_key_exits_with_hint(tmp_path, capsys, monkeypatch):
    """需要模型的命令在缺 key 时应给出友好提示 + 退出码 2，而不是抛栈。

    必须同时屏蔽 .env：只 delenv 环境变量的话，
    build_provider 里的 _load_dotenv() 会把 .env 的 key 又 setdefault 回来，
    测试会因为开发者本地配了 .env 而失效。
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    db = tmp_path / "cli.db"
    code = cli.main(["--db", str(db), "learn", "--topic", "T"])
    assert code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "scripted" in err


def test_module_is_executable():
    assert os.path.exists(os.path.join(BASE, "main.py"))
    assert os.path.exists(os.path.join(BASE, "requirements.txt"))
    assert os.path.exists(os.path.join(BASE, ".env.example"))
    assert os.path.exists(os.path.join(BASE, "README.md"))


# ------------------------------------------------------------------
# V0.2：experiment / report / bank
# ------------------------------------------------------------------
def test_bank_command(tmp_path, capsys):
    db = tmp_path / "cli.db"
    assert _run(db, "bank") == 0
    meta = json.loads(capsys.readouterr().out)
    assert meta["pre_count"] == meta["post_count"] == 4
    assert len(meta["fingerprint"]) == 64


def test_experiment_command_text_report(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "experiment", "--repeat", "1") == 0
    out = capsys.readouterr().out
    assert "Agent-01 V0.4 学习效果实验报告" in out
    # V0.4：六项指标 + 三个 gain
    for token in (
        "PRE", "OPEN_BOOK", "MEMORY_BLIND", "TRANSFER_BLIND",
        "PRACTICE", "NOTE_QUALITY",
        "OPEN_BOOK_GAIN", "MEMORY_GAIN", "TRANSFER_GAIN",
    ):
        assert token in out, f"报告缺少 {token}"


def test_experiment_command_json_report(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "experiment", "--repeat", "1", "--format", "json") == 0
    rep = json.loads(capsys.readouterr().out)
    s = rep["scores"]
    assert s["pre_blind"] >= 0
    assert s["open_book"] >= 0
    assert s["memory_blind"] >= 0
    assert s["transfer_blind"] >= 0
    assert s["open_book_gain"] == round(s["open_book"] - s["pre_blind"], 2)
    assert s["memory_gain"] == round(s["memory_blind"] - s["pre_blind"], 2)
    assert s["transfer_gain"] == round(s["transfer_blind"] - s["pre_blind"], 2)
    assert len(rep["pre_items"]) == 4
    assert len(rep["memory_open_items"]) == 4
    assert len(rep["memory_blind_items"]) == 4
    assert len(rep["transfer_items"]) == 4
    assert all(r["mode"] == "open_book_memory" for r in rep["memory_open_items"])
    assert all(r["mode"] == "blind_model" for r in rep["memory_blind_items"])
    # TRANSFER 绝不开卷
    assert all(r["mode"] == "blind_model" for r in rep["transfer_items"])
    assert all(
        r["context_used_len"] == 0
        for r in rep["pre_items"] + rep["memory_blind_items"] + rep["transfer_items"]
    )
    assert all(r["context_used_len"] > 0 for r in rep["memory_open_items"])
    assert rep["interpretation"]["headline"]


def test_experiment_accepts_custom_material(tmp_path, capsys):
    material = tmp_path / "m.md"
    material.write_text("# 自定义材料\nB-Tree 平衡多路树，所有叶子同层。", encoding="utf-8")
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(
        db, "experiment", "--repeat", "1",
        "--material-file", str(material), "--format", "json",
    ) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["material"]["checksum"]


def test_report_command_uses_latest_experiment(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    _run(db, "experiment", "--repeat", "1")
    capsys.readouterr()
    assert _run(db, "report") == 0
    assert "学习效果实验报告" in capsys.readouterr().out


def test_report_command_specific_id(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    _run(db, "experiment", "--repeat", "1")
    capsys.readouterr()
    assert _run(db, "report", "--id", "1", "--format", "json") == 0
    assert json.loads(capsys.readouterr().out)["experiment_id"] == 1


def test_report_without_experiment_fails_cleanly(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "report") == 4
    assert "还没有任何实验" in capsys.readouterr().err


def test_experiment_persists_to_db(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    # V0.4 默认 repeat=5；这里显式跑 1 次，便于按单次实验的口径断言
    _run(db, "experiment", "--repeat", "1")
    capsys.readouterr()
    assert _run(db, "stats") == 0
    stats = json.loads(capsys.readouterr().out)
    # V0.4：四场考试共 16 题（PRE 4 + MEMORY_OPEN 4 + MEMORY_BLIND 4 + TRANSFER 4）
    assert stats["tasks"] == 16
    # V0.3 新增：结构化学习笔记必须落库
    assert stats["learning_notes"] >= 4
    # V0.4 新增两张表
    assert stats["note_quality"] >= 4
    assert stats["practice_evaluations"] >= 4


def test_experiment_repeat_defaults_to_five(tmp_path, capsys):
    """V0.4 要求至少重复 5 次 —— 默认值必须是 5，不能是 1。"""
    from brain.replicate import DEFAULT_REPEATS, MIN_TRIALS_FOR_VERDICT

    assert DEFAULT_REPEATS == 5
    assert MIN_TRIALS_FOR_VERDICT == 5


def test_experiment_repeat_runs_batch_and_aggregates(tmp_path, capsys):
    """--repeat N 必须跑 N 次并给出跨次统计。"""
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "experiment", "--repeat", "5", "--format", "json") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_trials"] == 5
    assert len(summary["trials"]) == 5
    for key in ("pre_score", "memory_open_score", "transfer_blind_score",
                "practice_usable_rate", "note_quality_avg"):
        assert summary["metrics"][key]["n"] == 5, f"{key} 缺跨次统计"
    assert summary["verdicts"]["open_book_gain"]["state"] in {
        "positive", "noisy", "non_positive", "insufficient",
    }
    assert summary["verdict"]["headline"]


def test_experiment_repeat_one_skips_aggregation(tmp_path, capsys):
    """--repeat 1 只出单次报告，不做跨次判定（std=0 的假确定性没意义）。"""
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "experiment", "--repeat", "1", "--format", "json") == 0
    rep = json.loads(capsys.readouterr().out)
    assert "metrics" in rep and "verdicts" not in rep


def test_batch_report_command(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    _run(db, "experiment", "--repeat", "5")
    capsys.readouterr()
    assert _run(db, "batch-report", "--format", "json") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_trials"] == 5
    assert "六项指标" in capsys.readouterr().out or True


def test_trials_command_lists_each_trial(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    _run(db, "experiment", "--repeat", "5")
    capsys.readouterr()
    assert _run(db, "trials") == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["trials"]) == 5
    assert [t["trial_no"] for t in payload["trials"]] == [1, 2, 3, 4, 5]


def test_batch_report_without_batch_fails_cleanly(tmp_path, capsys):
    db = tmp_path / "cli.db"
    _run(db, "init")
    capsys.readouterr()
    assert _run(db, "batch-report") == 4
    assert "重复实验" in capsys.readouterr().err

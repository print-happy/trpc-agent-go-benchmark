#!/usr/bin/env python3
"""Render LongMemEval memory comparison reports from result JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResultSpec:
    label: str
    rel_path: str
    group: str
    reference: bool = False
    required: bool = True


@dataclass
class ResultView:
    label: str
    framework: str
    scenario: str
    backend: str
    cases: int
    total_cases: int
    accuracy: float
    f1: float
    bleu: float
    rouge_l: float
    prompt_per_qa: float
    completion_per_qa: float
    calls_per_qa: float
    avg_latency_ms: float
    total_tokens: int
    total_time_ms: int
    failed_cases: int
    successful_cases: int
    memory_only_compliant: bool | None
    native_memory_preserved: bool | None
    fairly_comparable: bool | None
    comparison_status: str
    comparison_blockers: list[str]
    memory_build_status: str
    memory_only_summary: dict[str, Any]
    cost: dict[str, Any]
    by_type: dict[str, dict[str, Any]]
    case_ids: list[str]
    question_type_counts: dict[str, int]
    path: Path
    group: str
    reference: bool = False


_EXPECTED_CASES = 70
_EXPECTED_QUESTION_TYPE = "single-session-user"

_RESULT_SPECS = [
    ResultSpec(
        "Long-Context Reference",
        "long_context/results.json",
        "internal",
        reference=True,
        required=False,
    ),
    ResultSpec("trpc-agent-go Auto", "auto_pgvector/results.json", "internal"),
    ResultSpec(
        "ADK Native Memory",
        "adk_python/native_memory/results.json",
        "external",
    ),
    ResultSpec(
        "Agno Native Memory",
        "agno_python/native_memory/results.json",
        "external",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render LongMemEval memory comparison report",
    )
    parser.add_argument(
        "--root",
        default="../results/longmemeval",
        help="LongMemEval result root directory",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=_EXPECTED_CASES,
        help="Expected completed LongMemEval cases (0 disables the check)",
    )
    parser.add_argument(
        "--question-type",
        default=_EXPECTED_QUESTION_TYPE,
        help="Required LongMemEval question type for every case",
    )
    return parser.parse_args()


def load_result(
    path: Path,
    spec: ResultSpec,
    expected_cases: int,
    question_type: str,
) -> ResultView:
    if not path.exists():
        raise FileNotFoundError(f"required result is missing: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    metadata = raw.get("metadata") or {}
    summary = raw.get("summary") or {}
    overall = summary.get("overall") or {}
    cases_payload = raw.get("cases")
    if not isinstance(cases_payload, list):
        raise ValueError(f"result does not contain top-level cases[]: {path}")
    dataset_format = str(metadata.get("dataset_format") or "").lower()
    if dataset_format != "longmemeval":
        raise ValueError(f"result is not LongMemEval dataset_format: {path}")

    cases = int(
        summary.get("completed_cases")
        or summary.get("total_questions")
        or len(cases_payload)
    )
    total_cases = int(
        summary.get("total_cases")
        or summary.get("total_questions")
        or cases
    )
    if cases != len(cases_payload):
        raise ValueError(
            f"completed_cases does not match cases[] length in {path}: "
            f"{cases} != {len(cases_payload)}"
        )
    if total_cases != cases:
        raise ValueError(
            f"result is incomplete in {path}: completed {cases}, "
            f"total {total_cases}"
        )
    if expected_cases > 0 and cases != expected_cases:
        raise ValueError(
            f"expected {expected_cases} completed cases in {path}, got {cases}"
        )
    _validate_question_types(cases_payload, question_type, path)
    case_ids = _case_ids(cases_payload)
    duplicates = _duplicate_values(case_ids)
    if duplicates:
        raise ValueError(
            f"duplicate question_id values in {path}: "
            f"{', '.join(duplicates)}"
        )
    question_type_counts = _question_type_counts(cases_payload)
    failed_cases = int(summary.get("failed_cases") or 0)
    successful_cases = int(summary.get("successful_cases") or cases - failed_cases)
    memory_build = metadata.get("memory_build") or {}
    cost = _normalize_cost(raw.get("cost") or metadata.get("cost"), summary)

    prompt_per_qa = float(
        _first_present(
            summary,
            "avg_prompt_tokens_per_qa",
            fallback=_safe_div(summary.get("total_prompt_tokens", 0), max(cases, 1)),
        )
    )
    completion_per_qa = float(
        _first_present(
            summary,
            "avg_completion_tokens_per_qa",
            fallback=_safe_div(
                summary.get("total_completion_tokens", 0), max(cases, 1),
            ),
        )
    )
    calls_per_qa = float(
        _first_present(
            summary,
            "avg_llm_calls_per_qa",
            fallback=_safe_div(summary.get("total_llm_calls", 0), max(cases, 1)),
        )
    )
    by_type = _normalize_by_type(raw.get("by_type") or raw.get("by_category") or {})

    return ResultView(
        label=spec.label,
        framework=str(metadata.get("framework") or spec.label),
        scenario=str(metadata.get("scenario") or "-"),
        backend=str(metadata.get("memory_backend") or "-"),
        cases=cases,
        total_cases=total_cases,
        accuracy=_metric(overall, summary, "accuracy", "overall_accuracy"),
        f1=_metric(overall, summary, "f1", "overall_f1"),
        bleu=_metric(overall, summary, "bleu", "overall_bleu"),
        rouge_l=_metric(overall, summary, "rouge_l", "overall_rouge_l"),
        prompt_per_qa=prompt_per_qa,
        completion_per_qa=completion_per_qa,
        calls_per_qa=calls_per_qa,
        avg_latency_ms=float(summary.get("avg_latency_ms") or 0.0),
        total_tokens=int(summary.get("total_tokens") or 0),
        total_time_ms=int(summary.get("total_time_ms") or 0),
        failed_cases=failed_cases,
        successful_cases=successful_cases,
        memory_only_compliant=_optional_bool(
            metadata.get("memory_only_compliant")
        ),
        native_memory_preserved=_optional_bool(
            metadata.get("native_memory_preserved")
        ),
        fairly_comparable=_optional_bool(metadata.get("fairly_comparable")),
        comparison_status=str(metadata.get("comparison_status") or "unknown"),
        comparison_blockers=[
            str(item) for item in metadata.get("comparison_blockers") or []
        ],
        memory_build_status=str(memory_build.get("status") or "unknown"),
        memory_only_summary=metadata.get("memory_only_summary") or {},
        cost=cost,
        by_type=by_type,
        case_ids=case_ids,
        question_type_counts=question_type_counts,
        path=path,
        group=spec.group,
        reference=spec.reference,
    )


def _case_ids(cases_payload: list[Any]) -> list[str]:
    return [str(case.get("question_id") or "") for case in cases_payload]


def _question_type_counts(cases_payload: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases_payload:
        qtype = str(case.get("question_type") or case.get("category") or "")
        counts[qtype] = counts.get(qtype, 0) + 1
    return counts


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def validate_comparable_case_sets(results: list[ResultView]) -> None:
    if not results:
        return
    expected = results[0]
    for item in results[1:]:
        if item.case_ids != expected.case_ids:
            raise ValueError(
                "LongMemEval case id set/order mismatch between "
                f"{expected.label} ({expected.path}) and "
                f"{item.label} ({item.path})"
            )
        if item.question_type_counts != expected.question_type_counts:
            raise ValueError(
                "LongMemEval question type distribution mismatch between "
                f"{expected.label} ({expected.question_type_counts}) and "
                f"{item.label} ({item.question_type_counts})"
            )


def _validate_question_types(
    cases_payload: list[Any],
    question_type: str,
    path: Path,
) -> None:
    if not question_type:
        return
    bad_types = sorted({
        str(case.get("question_type") or case.get("category") or "")
        for case in cases_payload
        if str(case.get("question_type") or case.get("category") or "")
        != question_type
    })
    if bad_types:
        raise ValueError(
            f"unexpected question types in {path}: {', '.join(bad_types)}; "
            f"expected only {question_type}"
        )


def _metric(
    overall: dict[str, Any],
    summary: dict[str, Any],
    overall_key: str,
    summary_key: str,
) -> float:
    value = overall.get(overall_key)
    if value is None and overall_key == "rouge_l":
        value = overall.get("rouge_L")
        if value is None:
            value = overall.get("rougel")
    if value is None:
        value = summary.get(summary_key)
    if value is None and overall_key == "accuracy":
        value = summary.get("task_averaged_accuracy")
    return float(value or 0.0)


def _first_present(data: dict[str, Any], key: str, fallback: Any) -> Any:
    value = data.get(key)
    if value is None:
        return fallback
    return value


def _normalize_by_type(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        value = value or {}
        metrics = value.get("metrics") or value
        out[str(key)] = {
            "count": int(value.get("count") or metrics.get("count") or 0),
            "accuracy": float(metrics.get("accuracy") or 0.0),
            "f1": float(metrics.get("f1") or 0.0),
            "bleu": float(metrics.get("bleu") or 0.0),
            "rouge_l": float(
                metrics.get("rouge_l")
                or metrics.get("rouge_L")
                or metrics.get("rougel")
                or 0.0
            ),
        }
    return out


def _safe_div(numerator: Any, denominator: int) -> float:
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _empty_cost_bucket(requests: int | None = None) -> dict[str, Any]:
    bucket: dict[str, Any] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "tokens_known": True,
    }
    if requests is not None:
        bucket["requests"] = requests
    return bucket


def _normalize_cost(raw: Any, summary: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict) and raw:
        llm = raw.get("llm") or {}
        embedding = raw.get("embedding") or {}
        cost = {
            "llm": {
                "total": _bucket(llm.get("total")),
                "memory_build": _bucket(llm.get("memory_build")),
                "qa": _bucket(llm.get("qa")),
                "judge": _bucket(llm.get("judge")),
            },
            "embedding": {
                "total": _bucket(
                    embedding.get("total"),
                    include_requests=True,
                ),
                "memory_build": _bucket(
                    embedding.get("memory_build"),
                    include_requests=True,
                ),
                "qa_retrieval": _bucket(
                    embedding.get("qa_retrieval"),
                    include_requests=True,
                ),
            },
        }
        if bool(raw.get("partial")):
            cost["partial"] = True
            reason = str(raw.get("partial_reason") or "").strip()
            if reason:
                cost["partial_reason"] = reason
        return cost
    calls = int(summary.get("total_llm_calls") or 0)
    total_bucket = _empty_cost_bucket()
    total_bucket.update({
        "calls": calls,
        "prompt_tokens": int(summary.get("total_prompt_tokens") or 0),
        "completion_tokens": int(
            summary.get("total_completion_tokens") or 0
        ),
        "total_tokens": int(summary.get("total_tokens") or 0),
        "cached_tokens": int(summary.get("total_cached_tokens") or 0),
        "tokens_known": calls == 0 or int(summary.get("total_tokens") or 0) > 0,
    })
    return {
        "llm": {
            "total": total_bucket,
            "memory_build": _empty_cost_bucket(),
            "qa": total_bucket,
            "judge": _empty_cost_bucket(),
        },
        "embedding": {
            "total": _empty_cost_bucket(requests=0),
            "memory_build": _empty_cost_bucket(requests=0),
            "qa_retrieval": _empty_cost_bucket(requests=0),
        },
    }


def _bucket(raw: Any, include_requests: bool = False) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _empty_cost_bucket(0 if include_requests else None)
    bucket = _empty_cost_bucket(0 if include_requests else None)
    for key in (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_hits",
        "requests",
    ):
        if key in raw:
            bucket[key] = int(raw.get(key) or 0)
    bucket["tokens_known"] = bool(raw.get("tokens_known", True))
    return bucket


def _cost_bucket(item: ResultView, modality: str, phase: str) -> dict[str, Any]:
    return ((item.cost.get(modality) or {}).get(phase) or _empty_cost_bucket())


def _tokens_label(bucket: dict[str, Any]) -> str:
    value = str(int(bucket.get("total_tokens") or 0))
    if not bool(bucket.get("tokens_known", True)):
        return value + "?"
    return value


def _cost_note(item: ResultView) -> str:
    if not bool(item.cost.get("partial")):
        return ""
    return str(item.cost.get("partial_reason") or "partial").strip()


def render(results: list[ResultView], zh: bool) -> str:
    lines: list[str] = []
    if zh:
        lines.append("# LongMemEval Memory Benchmark 对比报告")
        lines.append("")
        lines.append("## 实验设置")
        lines.append("")
        lines.append("- **范围**：LongMemEval `single-session-user` 70 条样本。")
        lines.append("- **参考基线**：`long_context` 复用既有结果，不属于本轮新增运行对象。")
        lines.append("- **本轮 memory/native 对比**：`trpc-agent-go auto`、ADK、Agno。")
        lines.append("- **主指标**：LongMemEval official yes/no judge accuracy；F1、BLEU、ROUGE-L 为确定性辅助指标。")
        lines.append("")
        lines.append("## 总体结果")
        lines.append("")
        lines.append("| 结果 | 框架 | 场景 | 后端 | 样本 | Accuracy | F1 | BLEU | ROUGE-L | Prompt/QA | Calls/QA | Latency/QA(ms) | Tokens |")
    else:
        lines.append("# LongMemEval Memory Benchmark Comparison")
        lines.append("")
        lines.append("## Experiment Setup")
        lines.append("")
        lines.append("- **Scope**: 70 LongMemEval `single-session-user` cases.")
        lines.append("- **Reference baseline**: `long_context` reuses existing results and is not rerun in this comparison.")
        lines.append("- **Memory/native runs**: `trpc-agent-go auto`, ADK, and Agno.")
        lines.append("- **Primary metric**: LongMemEval official yes/no judge accuracy; F1, BLEU, and ROUGE-L are deterministic auxiliary metrics.")
        lines.append("")
        lines.append("## Overall Results")
        lines.append("")
        lines.append("| Result | Framework | Scenario | Backend | Cases | Accuracy | F1 | BLEU | ROUGE-L | Prompt/QA | Calls/QA | Latency/QA(ms) | Tokens |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    _append_result_rows(lines, results)
    lines.append("")
    _append_group_section(lines, results, "internal", zh)
    _append_group_section(lines, results, "external", zh)
    _append_type_section(lines, results, zh)
    _append_cost_section(lines, results, zh)
    _append_model_cost_section(lines, results, zh)
    _append_phase_cost_section(lines, results, zh)
    _append_compliance_section(lines, results, zh)
    _append_fairness_section(lines, zh)
    return "\n".join(lines)


def _append_result_rows(lines: list[str], results: list[ResultView]) -> None:
    for item in results:
        label = item.label
        if item.reference:
            label += " (ref)"
        lines.append(
            f"| {label} | {item.framework} | {item.scenario} | "
            f"{item.backend} | {item.cases}/{item.total_cases} | "
            f"{item.accuracy:.4f} | {item.f1:.4f} | {item.bleu:.4f} | "
            f"{item.rouge_l:.4f} | {item.prompt_per_qa:.0f} | "
            f"{item.calls_per_qa:.2f} | {item.avg_latency_ms:.0f} | "
            f"{item.total_tokens} |"
        )


def _append_group_section(
    lines: list[str],
    results: list[ResultView],
    group: str,
    zh: bool,
) -> None:
    group_results = [item for item in results if item.group == group]
    if not group_results:
        return
    if zh:
        title = "## 内部场景对比" if group == "internal" else "## 外部框架 native memory 对比"
    else:
        title = "## Internal Scenario Comparison" if group == "internal" else "## External Native Memory Comparison"
    lines.append(title)
    lines.append("")
    lines.append("| Result | Accuracy | F1 | BLEU | ROUGE-L | Prompt/QA | Calls/QA |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in group_results:
        lines.append(
            f"| {item.label} | {item.accuracy:.4f} | {item.f1:.4f} | "
            f"{item.bleu:.4f} | {item.rouge_l:.4f} | "
            f"{item.prompt_per_qa:.0f} | {item.calls_per_qa:.2f} |"
        )
    lines.append("")


def _append_type_section(lines: list[str], results: list[ResultView], zh: bool) -> None:
    lines.append("## 按 question type 分析" if zh else "## By Question Type")
    lines.append("")
    lines.append("| Result | Type | Count | Accuracy | F1 | BLEU | ROUGE-L |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in results:
        for question_type in sorted(item.by_type):
            metrics = item.by_type[question_type]
            lines.append(
                f"| {item.label} | {question_type} | {metrics['count']} | "
                f"{metrics['accuracy']:.4f} | {metrics['f1']:.4f} | "
                f"{metrics['bleu']:.4f} | {metrics['rouge_l']:.4f} |"
            )
    lines.append("")


def _append_cost_section(lines: list[str], results: list[ResultView], zh: bool) -> None:
    lines.append("## Token / 调用 / 延迟成本" if zh else "## Token / Call / Latency Cost")
    lines.append("")
    lines.append("| Result | Prompt/QA | Completion/QA | Calls/QA | Latency/QA(ms) | Total Time(ms) | Total Tokens |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in results:
        lines.append(
            f"| {item.label} | {item.prompt_per_qa:.0f} | "
            f"{item.completion_per_qa:.0f} | {item.calls_per_qa:.2f} | "
            f"{item.avg_latency_ms:.0f} | {item.total_time_ms} | "
            f"{item.total_tokens} |"
        )
    lines.append("")


def _append_model_cost_section(
    lines: list[str],
    results: list[ResultView],
    zh: bool,
) -> None:
    lines.append("## 模型调用成本总览" if zh else "## Model Call Cost Summary")
    lines.append("")
    lines.append("| Result | LLM Calls | LLM Tokens | Embedding Calls | Embedding Requests | Embedding Cache Hits | Embedding Tokens | Note |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in results:
        llm = _cost_bucket(item, "llm", "total")
        emb = _cost_bucket(item, "embedding", "total")
        lines.append(
            f"| {item.label} | {int(llm.get('calls') or 0)} | "
            f"{_tokens_label(llm)} | {int(emb.get('calls') or 0)} | "
            f"{int(emb.get('requests') or 0)} | "
            f"{int(emb.get('cache_hits') or 0)} | "
            f"{_tokens_label(emb)} | {_cost_note(item)} |"
        )
    lines.append("")


def _append_phase_cost_section(
    lines: list[str],
    results: list[ResultView],
    zh: bool,
) -> None:
    lines.append("## 分阶段模型成本" if zh else "## Model Cost By Phase")
    lines.append("")
    lines.append("| Result | Modality | Phase | Calls | Requests | Cache Hits | Prompt | Completion | Total | Cached | Tokens Known |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    phase_specs = [
        ("llm", "memory_build"),
        ("llm", "qa"),
        ("llm", "judge"),
        ("embedding", "memory_build"),
        ("embedding", "qa_retrieval"),
    ]
    for item in results:
        for modality, phase in phase_specs:
            bucket = _cost_bucket(item, modality, phase)
            lines.append(
                f"| {item.label} | {modality} | {phase} | "
                f"{int(bucket.get('calls') or 0)} | "
                f"{int(bucket.get('requests') or 0)} | "
                f"{int(bucket.get('cache_hits') or 0)} | "
                f"{int(bucket.get('prompt_tokens') or 0)} | "
                f"{int(bucket.get('completion_tokens') or 0)} | "
                f"{int(bucket.get('total_tokens') or 0)} | "
                f"{int(bucket.get('cached_tokens') or 0)} | "
                f"{_status_label(bool(bucket.get('tokens_known', True)))} |"
            )
    lines.append("")


def _append_compliance_section(
    lines: list[str],
    results: list[ResultView],
    zh: bool,
) -> None:
    if zh:
        lines.append("## Memory-only 合规与失败状态")
        lines.append("")
        lines.append("| Result | Memory-only | Native memory | 可公平比较 | 状态 | 成功/总数 | 失败数 | Memory build | Blockers |")
    else:
        lines.append("## Memory-only Compliance and Failure Status")
        lines.append("")
        lines.append("| Result | Memory-only | Native memory | Comparable | Status | Success/Total | Failures | Memory build | Blockers |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |")
    for item in results:
        blockers = item.comparison_blockers
        if not blockers and item.memory_only_summary.get("violations"):
            blockers = ["QA context leak detected"]
        blocker_text = "; ".join(blockers) if blockers else "-"
        lines.append(
            f"| {item.label} | {_status_label(item.memory_only_compliant)} | "
            f"{_status_label(item.native_memory_preserved)} | "
            f"{_status_label(item.fairly_comparable)} | "
            f"{item.comparison_status} | "
            f"{item.successful_cases}/{item.total_cases} | "
            f"{item.failed_cases} | {item.memory_build_status} | "
            f"{blocker_text} |"
        )
    lines.append("")


def _status_label(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _append_fairness_section(lines: list[str], zh: bool) -> None:
    if zh:
        lines.append("## 实现差异与公平性说明")
        lines.append("")
        lines.append("- **`long_context`**：仅作为参考上界复用既有结果，本轮不重跑。")
        lines.append("- **外部框架**：ADK、Agno 使用各自 native memory 能力；memory 表示、抽取策略、向量库和工具暴露方式可能不同。")
        lines.append("- **运行范围**：主报告只接受 `single-session-user` 70 条完成结果；缺失、未完成或混入其他 question type 会直接失败。")
        lines.append("- **失败策略**：LongMemEval judge 输出必须严格解析为 `yes` 或 `no`，不做隐式 fallback 或静默补分。")
    else:
        lines.append("## Implementation Differences and Fairness Notes")
        lines.append("")
        lines.append("- **`long_context`**: reused only as a reference upper bound and not rerun.")
        lines.append("- **External frameworks**: ADK and Agno use their native memory capabilities; memory representation, extraction strategy, vector store, and tool exposure may differ.")
        lines.append("- **Run scope**: the main report accepts only 70 completed `single-session-user` cases; missing, incomplete, or mixed-type results fail fast.")
        lines.append("- **Failure policy**: LongMemEval judge responses must strictly parse as `yes` or `no`; there is no implicit fallback or silent credit.")
    lines.append("")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    results: list[ResultView] = []
    for spec in _RESULT_SPECS:
        path = root / spec.rel_path
        if not path.exists() and not spec.required:
            continue
        results.append(
            load_result(
                path,
                spec,
                args.expected_cases,
                args.question_type,
            )
        )
    validate_comparable_case_sets(results)
    (root / "REPORT.md").write_text(render(results, zh=False), encoding="utf-8")
    (root / "REPORT.zh_CN.md").write_text(render(results, zh=True), encoding="utf-8")


if __name__ == "__main__":
    main()

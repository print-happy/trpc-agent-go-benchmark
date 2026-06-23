"""Shared LongMemEval helpers for Python memory benchmark runners."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from importlib import util as importlib_util
from pathlib import Path
from typing import Any


_LME_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
}

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "and", "or", "but",
    "if", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once",
})

_PUNCT_RE = re.compile(r"[^\w\s]")


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Session:
    session_id: str
    session_date: str
    turns: list[Turn]
    observation: str = ""
    summary: str = ""


@dataclass
class QAItem:
    question_id: str
    question: str
    answer: str
    category: str
    evidence: list[str] = field(default_factory=list)
    question_date: str = ""
    question_type: str = ""
    is_abstention: bool = False


@dataclass
class LoCoMoSample:
    sample_id: str
    speakers: list[str]
    conversation: list[Session]
    qa: list[QAItem]

    def build_full_conversation(self) -> str:
        parts: list[str] = []
        for sess in self.conversation:
            header = f"[Session {sess.session_id}"
            if sess.session_date:
                header += f" - {sess.session_date}"
            parts.append(header + "]")
            for turn in sess.turns:
                parts.append(f"{turn.speaker}: {turn.text}")
            parts.append("")
        return "\n".join(parts)


def dataset_path(data_dir: str, filename: str) -> Path:
    path = Path(data_dir)
    if path.suffix in {".json", ".jsonl"}:
        return path
    return path / filename


def require_dataset_file(data_dir: str, filename: str) -> Path:
    path = dataset_path(data_dir, filename)
    if not path.exists():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"dataset path is not a file: {path}")
    return path


def ensure_output_directory(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    probe = output_dir / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"output directory is not writable: {output_dir}") from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
    return output_dir


def require_model_name(model_name: str, label: str) -> None:
    if not str(model_name or "").strip():
        raise ValueError(f"{label} model name is empty")


def require_openai_credentials() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if os.environ.get("AZURE_API_KEY"):
        return
    raise EnvironmentError(
        "OPENAI_API_KEY or AZURE_API_KEY must be set before evaluation"
    )


def require_python_modules(modules: list[str]) -> None:
    missing = [name for name in modules if importlib_util.find_spec(name) is None]
    if missing:
        raise ImportError(
            "missing required Python modules: " + ", ".join(sorted(missing))
        )


def preflight_common(
    data_dir: str,
    filename: str,
    output_dir: str,
    model_name: str,
    eval_model_name: str,
    required_modules: list[str] | None = None,
) -> dict[str, str]:
    dataset = require_dataset_file(data_dir, filename)
    output = ensure_output_directory(output_dir)
    require_model_name(model_name, "answer")
    require_model_name(eval_model_name, "judge")
    require_openai_credentials()
    if required_modules:
        require_python_modules(required_modules)
    return {
        "dataset_path": str(dataset),
        "output_dir": str(output),
        "model": model_name,
        "eval_model": eval_model_name,
    }


def parse_question_types(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def sample_case_ids(samples: list[LoCoMoSample]) -> list[str]:
    return [sample.sample_id for sample in samples]


def sample_question_type_counts(samples: list[LoCoMoSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        if not sample.qa:
            continue
        qtype = sample.qa[0].question_type or sample.qa[0].category
        counts[qtype] = counts.get(qtype, 0) + 1
    return counts


def sample_set_metadata(samples: list[LoCoMoSample]) -> dict[str, Any]:
    return {
        "case_count": len(samples),
        "case_ids": sample_case_ids(samples),
        "question_type_counts": sample_question_type_counts(samples),
    }


def result_case_ids(cases: list[dict[str, Any]]) -> list[str]:
    return [str(case.get("question_id") or "") for case in cases]


def result_question_type_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        qtype = str(
            case.get("question_type") or case.get("category") or ""
        )
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


def validate_sample_set(
    samples: list[LoCoMoSample],
    expected_case_ids: list[str] | None = None,
    expected_question_types: list[str] | None = None,
    expected_question_type_counts: dict[str, int] | None = None,
) -> None:
    if not samples:
        raise ValueError("no LongMemEval samples to evaluate")
    actual = sample_case_ids(samples)
    duplicates = _duplicate_values(actual)
    if duplicates:
        raise ValueError(
            "duplicate LongMemEval case ids: "
            f"{', '.join(duplicates)}"
        )
    missing_qa = [sample.sample_id for sample in samples if not sample.qa]
    if missing_qa:
        raise ValueError(
            "LongMemEval samples without QA items: "
            f"{', '.join(missing_qa)}"
        )
    if expected_case_ids is not None:
        if actual != expected_case_ids:
            raise ValueError(
                "LongMemEval case id set/order mismatch: "
                f"expected {expected_case_ids}, got {actual}"
            )
    allowed = {item.lower() for item in expected_question_types or []}
    if allowed:
        bad_types = sorted({
            (sample.qa[0].question_type or sample.qa[0].category)
            for sample in samples
            if sample.qa
            and (sample.qa[0].question_type or sample.qa[0].category).lower()
            not in allowed
        })
        if bad_types:
            raise ValueError(
                "unexpected LongMemEval question types: "
                f"{', '.join(bad_types)}"
            )
    if expected_question_type_counts is not None:
        expected_counts = {
            str(key): int(value)
            for key, value in expected_question_type_counts.items()
        }
        actual_counts = sample_question_type_counts(samples)
        if actual_counts != expected_counts:
            raise ValueError(
                "LongMemEval question type distribution mismatch: "
                f"expected {expected_counts}, got {actual_counts}"
            )


def detect_direct_context_leaks(
    sample: LoCoMoSample,
    payloads: dict[str, str],
    min_chars: int = 200,
) -> list[str]:
    violations: list[str] = []
    transcript = sample.build_full_conversation().strip()
    if len(transcript) >= min_chars:
        for name, payload in payloads.items():
            if transcript and transcript in (payload or ""):
                violations.append(f"{name} contains the full transcript")
    for sess in sample.conversation:
        session_text = "\n".join(
            f"{turn.speaker}: {turn.text}" for turn in sess.turns
        ).strip()
        if len(session_text) < min_chars:
            continue
        for name, payload in payloads.items():
            if session_text and session_text in (payload or ""):
                violations.append(
                    f"{name} contains full session {sess.session_id}"
                )
    for qa in sample.qa:
        for idx, evidence in enumerate(getattr(qa, "evidence", []) or []):
            evidence_text = str(evidence).strip()
            if len(evidence_text) < min_chars:
                continue
            for name, payload in payloads.items():
                if evidence_text and evidence_text in (payload or ""):
                    violations.append(
                        f"{name} contains gold evidence {idx + 1}"
                    )
    return sorted(set(violations))


def memory_only_compliance_status(
    cases: list[dict[str, Any]],
    native_memory_preserved: bool,
) -> dict[str, Any]:
    summary = memory_only_summary(cases)
    failed_cases = [
        str(case.get("question_id") or "")
        for case in cases
        if str(case.get("status") or "completed") != "completed"
    ]
    comparable = bool(
        native_memory_preserved and summary["compliant"] and not failed_cases
    )
    reasons: list[str] = []
    if not native_memory_preserved:
        reasons.append("native memory mechanism is not preserved")
    if not summary["compliant"]:
        reasons.append("QA context leak detected")
    if failed_cases:
        reasons.append("one or more cases failed during evaluation")
    return {
        "memory_only_compliant": bool(summary["compliant"]),
        "native_memory_preserved": bool(native_memory_preserved),
        "fairly_comparable": comparable,
        "status": "comparable" if comparable else "not_comparable",
        "reasons": reasons,
        "failed_cases": failed_cases,
        "summary": summary,
    }


def memory_only_failure_reason(violations: list[str]) -> str:
    if not violations:
        return ""
    return "Memory-only context leak detected: " + "; ".join(violations)


def build_memory_only_question(qa: Any) -> str:
    question_date = str(getattr(qa, "question_date", "") or "").strip()
    if question_date:
        return f"Current Date: {question_date}\nQuestion: {qa.question}"
    return str(qa.question)


def memory_only_policy_metadata(
    framework: str,
    memory_source: str,
    qa_runtime: str,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "framework": framework,
        "qa_runtime": qa_runtime,
        "allowed_inputs": [
            "current_question",
            "question_date",
            memory_source,
        ],
        "forbidden_inputs": [
            "full_conversation_transcript",
            "full_session_transcript",
            "longmemeval_haystack",
            "gold_evidence",
            "gold_answer_except_judge_prompt",
        ],
        "leak_detection_scopes": [
            "qa_user_message",
            "qa_agent_instruction",
            "qa_session_state",
        ],
    }


def memory_only_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [
        str(case.get("question_id") or "")
        for case in cases
        if not bool(case.get("memory_only_compliant", False))
    ]
    violations: dict[str, list[str]] = {}
    for case in cases:
        case_violations = case.get("context_leak_violations") or []
        if case_violations:
            violations[str(case.get("question_id") or "")] = [
                str(item) for item in case_violations
            ]
    return {
        "compliant": not failed,
        "checked_cases": len(cases),
        "failed_cases": failed,
        "violations": violations,
    }


def cost_bucket(
    calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    tokens_known: bool | None = None,
    cache_hits: int = 0,
    requests: int | None = None,
) -> dict[str, Any]:
    if tokens_known is None:
        tokens_known = (
            calls == 0
            or total_tokens > 0
            or prompt_tokens > 0
            or completion_tokens > 0
            or cached_tokens > 0
        )
    bucket: dict[str, Any] = {
        "calls": int(calls),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "cached_tokens": int(cached_tokens),
        "tokens_known": bool(tokens_known),
    }
    if cache_hits:
        bucket["cache_hits"] = int(cache_hits)
    if requests is not None:
        bucket["requests"] = int(requests)
    return bucket


def token_usage_cost_bucket(usage: Any | None) -> dict[str, Any]:
    if usage is None:
        return cost_bucket()
    calls = int(getattr(usage, "llm_calls", 0) or 0)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)
    cached = int(getattr(usage, "cached_tokens", 0) or 0)
    known = getattr(usage, "tokens_known", None)
    return cost_bucket(
        calls=calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        tokens_known=known,
    )


def add_cost_buckets(*buckets: dict[str, Any] | None) -> dict[str, Any]:
    total = cost_bucket(tokens_known=True)
    seen = False
    for bucket in buckets:
        if not bucket:
            continue
        seen = True
        total["calls"] += int(bucket.get("calls") or 0)
        total["prompt_tokens"] += int(bucket.get("prompt_tokens") or 0)
        total["completion_tokens"] += int(
            bucket.get("completion_tokens") or 0
        )
        total["total_tokens"] += int(bucket.get("total_tokens") or 0)
        total["cached_tokens"] += int(bucket.get("cached_tokens") or 0)
        total["cache_hits"] = int(total.get("cache_hits") or 0) + int(
            bucket.get("cache_hits") or 0
        )
        if "requests" in bucket or "requests" in total:
            total["requests"] = int(total.get("requests") or 0) + int(
                bucket.get("requests") or 0
            )
        total["tokens_known"] = bool(total["tokens_known"]) and bool(
            bucket.get("tokens_known", True)
        )
    if not seen:
        return cost_bucket()
    if not total.get("cache_hits"):
        total.pop("cache_hits", None)
    if not total.get("requests"):
        total.pop("requests", None)
    return total


def build_cost_report(
    llm_memory_build: dict[str, Any] | None = None,
    llm_qa: dict[str, Any] | None = None,
    llm_judge: dict[str, Any] | None = None,
    embedding_memory_build: dict[str, Any] | None = None,
    embedding_qa_retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm_memory_build = llm_memory_build or cost_bucket()
    llm_qa = llm_qa or cost_bucket()
    llm_judge = llm_judge or cost_bucket()
    embedding_memory_build = embedding_memory_build or cost_bucket(
        requests=0,
    )
    embedding_qa_retrieval = embedding_qa_retrieval or cost_bucket(
        requests=0,
    )
    return {
        "llm": {
            "total": add_cost_buckets(
                llm_memory_build, llm_qa, llm_judge,
            ),
            "memory_build": llm_memory_build,
            "qa": llm_qa,
            "judge": llm_judge,
        },
        "embedding": {
            "total": add_cost_buckets(
                embedding_memory_build, embedding_qa_retrieval,
            ),
            "memory_build": embedding_memory_build,
            "qa_retrieval": embedding_qa_retrieval,
        },
    }


def load_longmemeval_samples(
    data_dir: str,
    filename: str,
    question_types: list[str] | None = None,
    manifest_path: str | None = None,
) -> list[LoCoMoSample]:
    path = dataset_path(data_dir, filename)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"LongMemEval file must contain a list: {path}")
    allowed = {q.strip().lower() for q in question_types or [] if q.strip()}
    samples: list[LoCoMoSample] = []
    for idx, item in enumerate(raw):
        sample = _parse_instance(item, idx)
        qa = sample.qa[0]
        if allowed and qa.category.lower() not in allowed:
            continue
        samples.append(sample)
    if not samples:
        raise ValueError("no LongMemEval cases remain after filtering")
    if manifest_path:
        samples = filter_samples_by_manifest(samples, manifest_path)
    return samples


def filter_samples_by_manifest(
    samples: list[LoCoMoSample],
    manifest_path: str,
) -> list[LoCoMoSample]:
    path = Path(manifest_path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    case_ids = raw.get("case_ids") if isinstance(raw, dict) else None
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError(f"LongMemEval manifest {path} has no case_ids")
    by_id = {sample.sample_id: sample for sample in samples}
    filtered: list[LoCoMoSample] = []
    for raw_id in case_ids:
        case_id = str(raw_id).strip()
        sample = by_id.get(case_id)
        if sample is None:
            raise ValueError(
                f"LongMemEval manifest case_id {case_id} not found in dataset"
            )
        filtered.append(sample)
    return filtered


def _parse_instance(item: dict[str, Any], idx: int) -> LoCoMoSample:
    qid = str(item.get("question_id") or f"longmemeval_{idx + 1}").strip()
    qtype = str(item.get("question_type") or "").strip()
    question = str(item.get("question") or "").strip()
    answer = _decode_answer(item.get("answer"))
    dates = item.get("haystack_dates") or []
    session_ids = item.get("haystack_session_ids") or []
    sessions_raw = item.get("haystack_sessions") or []
    if not qid or not qtype or not question:
        raise ValueError(f"invalid LongMemEval instance at index {idx}")
    if len(dates) != len(sessions_raw) or len(session_ids) != len(sessions_raw):
        raise ValueError(f"haystack metadata count mismatch for {qid}")
    sessions: list[Session] = []
    for sess_idx, turns_raw in enumerate(sessions_raw):
        turns: list[Turn] = []
        for turn in turns_raw:
            role = str(turn.get("role") or "").strip()
            content = str(turn.get("content") or "")
            if role not in {"user", "assistant"}:
                raise ValueError(f"invalid role {role!r} for {qid}")
            if not content.strip():
                continue
            turns.append(Turn(speaker=role, text=content))
        sessions.append(Session(
            session_id=str(session_ids[sess_idx]),
            session_date=str(dates[sess_idx]),
            turns=turns,
        ))
    qa = QAItem(
        question_id=qid,
        question=question,
        answer=answer,
        category=qtype,
        question_type=qtype,
        question_date=str(item.get("question_date") or ""),
        is_abstention="_abs" in qid,
    )
    return LoCoMoSample(
        sample_id=qid,
        speakers=["user", "assistant"],
        conversation=sessions,
        qa=[qa],
    )


def _decode_answer(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float):
        return format(raw, "g")
    raise ValueError(f"unsupported LongMemEval answer: {raw!r}")


def is_longmemeval_qa(qa: Any) -> bool:
    qtype = str(getattr(qa, "question_type", "") or getattr(qa, "category", ""))
    return qtype in _LME_TYPES


def is_longmemeval_sample(sample: Any) -> bool:
    return bool(getattr(sample, "qa", None)) and is_longmemeval_qa(sample.qa[0])


def normalize_tokens(text: str) -> list[str]:
    text = text.replace("<｜end▁of▁sentence｜>", " ").lower()
    text = _PUNCT_RE.sub(" ", text)
    return [tok for tok in text.split() if tok and tok not in _STOP_WORDS]


def compute_rouge_l(prediction: str, reference: str) -> float:
    pred = normalize_tokens(prediction)
    ref = normalize_tokens(reference)
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    lcs = _lcs_len(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return (2 * precision * recall) / (precision + recall)


def compute_rouge_n(prediction: str, reference: str, n: int) -> float:
    pred = normalize_tokens(prediction)
    ref = normalize_tokens(reference)
    if len(pred) < n or len(ref) < n:
        return 1.0 if not pred and not ref else 0.0
    pred_ngrams = _ngrams(pred, n)
    ref_ngrams = _ngrams(ref, n)
    matches = sum((pred_ngrams & ref_ngrams).values())
    total_pred = sum(pred_ngrams.values())
    total_ref = sum(ref_ngrams.values())
    if total_pred == 0 or total_ref == 0 or matches == 0:
        return 0.0
    precision = matches / total_pred
    recall = matches / total_ref
    return (2 * precision * recall) / (precision + recall)


def _ngrams(tokens: list[str], n: int) -> Counter[str]:
    return Counter(
        " ".join(tokens[i:i + n])
        for i in range(0, len(tokens) - n + 1)
    )


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def build_longmemeval_judge_prompt(qa: Any, prediction: str) -> str:
    question = qa.question
    answer = qa.answer
    qtype = getattr(qa, "question_type", "") or qa.category
    if getattr(qa, "is_abstention", False):
        return (
            "I will give you an unanswerable question, an explanation, "
            "and a response from a model. Please answer yes if the model "
            "correctly identifies the question as unanswerable. The model "
            "could say that the information is incomplete, or some other "
            "information is given but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    if qtype in {"single-session-user", "single-session-assistant", "multi-session"}:
        lead = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If "
            "the response only contains a subset of the information required "
            "by the answer, answer no. "
        )
    elif qtype == "temporal-reasoning":
        lead = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If "
            "the response only contains a subset of the information required "
            "by the answer, answer no. In addition, do not penalize off-by-one "
            "errors for the number of days. If the question asks for the "
            "number of days/weeks/months, etc., and the model makes off-by-one "
            "errors (e.g., predicting 19 days when the answer is 18), the "
            "model's response is still correct. "
        )
    elif qtype == "knowledge-update":
        lead = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response contains "
            "some previous information along with an updated answer, the "
            "response should be considered as correct as long as the updated "
            "answer is the required answer."
        )
    elif qtype == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        raise ValueError(f"unsupported LongMemEval question type {qtype!r}")
    return (
        f"{lead}\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\n"
        f"Model Response: {prediction}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


def parse_longmemeval_judge_label(response: str) -> bool:
    normalized = response.strip().lower().strip(".! \n\t\r")
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise ValueError(f"judge response is not an exact yes/no: {response!r}")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ndcg_placeholder() -> float:
    return math.nan


def total_turns(sample: Any) -> int:
    return sum(
        len(sess.turns)
        for sess in getattr(sample, "conversation", []) or []
    )


def longmemeval_case_from_qa(sample: Any, qa: Any, latency_ms: int) -> dict[str, Any]:
    metrics_obj = {
        "f1": float(getattr(qa, "f1", 0.0)),
        "bleu": float(getattr(qa, "bleu", 0.0)),
        "rouge_1": float(getattr(qa, "rouge_1", 0.0)),
        "rouge_2": float(getattr(qa, "rouge_2", 0.0)),
        "rouge_l": float(getattr(qa, "rouge_l", 0.0)),
        "accuracy": float(getattr(qa, "accuracy", 0.0)),
    }
    turn_count = int(getattr(sample, "total_turns", 0) or total_turns(sample))
    session_count = int(
        getattr(sample, "total_sessions", 0)
        or len(getattr(sample, "conversation", []) or [])
    )
    return {
        "question_id": qa.question_id,
        "question_type": getattr(qa, "question_type", "") or qa.category,
        "question": qa.question,
        "question_date": getattr(qa, "question_date", ""),
        "expected": qa.reference,
        "predicted": qa.prediction,
        "is_abstention": bool(getattr(qa, "is_abstention", False)),
        "correct": bool(getattr(qa, "correct", False)),
        "metrics": metrics_obj,
        "latency_ms": int(latency_ms),
        "token_usage": {
            "prompt_tokens": int(getattr(qa, "prompt_tokens", 0)),
            "completion_tokens": int(getattr(qa, "completion_tokens", 0)),
            "total_tokens": int(getattr(qa, "total_tokens", 0)),
            "llm_calls": int(getattr(qa, "llm_calls", 0)),
        },
        "retry_count": 0,
        "total_turns": turn_count,
        "total_sessions": session_count,
    }


def aggregate_longmemeval_cases(cases: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    abstention_total = 0
    abstention_correct = 0
    non_abstention_total = 0
    metric_keys = ("f1", "bleu", "rouge_1", "rouge_2", "rouge_l", "accuracy")
    overall_values = {key: [] for key in metric_keys}
    type_acc: dict[str, list[float]] = {}
    total_latency = 0
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_calls = 0

    for case in cases:
        metrics_obj = case.get("metrics") or {}
        qtype = str(case.get("question_type") or "")
        bucket = by_type.setdefault(
            qtype,
            {"count": 0, "metrics": {key: 0.0 for key in metric_keys}},
        )
        bucket["count"] += 1
        for key in metric_keys:
            value = float(metrics_obj.get(key) or 0.0)
            bucket["metrics"][key] += value
            overall_values[key].append(value)
        type_acc.setdefault(qtype, []).append(float(metrics_obj.get("accuracy") or 0.0))
        if case.get("is_abstention"):
            abstention_total += 1
            if case.get("correct"):
                abstention_correct += 1
        else:
            non_abstention_total += 1
        total_latency += int(case.get("latency_ms") or 0)
        usage = case.get("token_usage") or {}
        total_prompt += int(usage.get("prompt_tokens") or 0)
        total_completion += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        total_calls += int(usage.get("llm_calls") or 0)

    for bucket in by_type.values():
        count = max(int(bucket["count"]), 1)
        for key in metric_keys:
            bucket["metrics"][key] /= count

    completed = len(cases)
    q_count = max(completed, 1)
    overall = {
        key: average(values)
        for key, values in overall_values.items()
    }
    task_averaged_accuracy = average([
        average(values) for values in type_acc.values()
    ])
    summary: dict[str, Any] = {
        "total_cases": completed,
        "completed_cases": completed,
        "total_samples": completed,
        "total_questions": completed,
        "overall": overall,
        "overall_f1": overall["f1"],
        "overall_bleu": overall["bleu"],
        "overall_rouge_l": overall["rouge_l"],
        "overall_accuracy": overall["accuracy"],
        "overall_llm_score": overall["accuracy"],
        "task_averaged_accuracy": task_averaged_accuracy,
        "abstention_count": abstention_total,
        "non_abstention_count": non_abstention_total,
        "avg_latency_ms": total_latency / q_count,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "total_llm_calls": total_calls,
        "avg_prompt_tokens_per_qa": total_prompt / q_count,
        "avg_completion_tokens_per_qa": total_completion / q_count,
        "avg_llm_calls_per_qa": total_calls / q_count,
    }
    if abstention_total:
        summary["abstention_accuracy"] = abstention_correct / abstention_total
    return summary, by_type

# Evaluation Results

This directory stores benchmark results for session summarization and on-demand retrieval of hidden context.

## Reports

| File | Description |
|------|-------------|
| [REPORT.md](REPORT.md) | Full evaluation report (English) |
| [REPORT.zh_CN.md](REPORT.zh_CN.md) | Full evaluation report (Chinese) |

## Benchmark Summary

The current report combines three complementary evaluations:

- **MT-Bench-101**: used to study when session summarization is broadly beneficial or harmful
- **QMSum**: used to study whether `summary_ondemand` can recover details hidden by summary compression on meeting transcripts (~19K tokens)
- **LongMemEval**: used to study whether on-demand retrieval works on realistic multi-session user/assistant dialogues at extreme context lengths (~102K tokens)

## MT-Bench-101 Evaluation Summary

**Configuration**:
- Model: deepseek-v3.2
- Summary Trigger: Every 2 turns (`-events 2`)
- Total Cases: 917 (9 tasks)

**Key Results**:

| Metric | Value |
|--------|------:|
| Overall Prompt Savings | 24.47% |
| Overall Token Savings | 12.89% |
| Weighted Consistency | 0.853 |
| Pass^1 Rate | 92.3% |
| Negative Token Cases | 35.9% |

**Task Suitability**:

| Suitability | Tasks | Avg Turns | Prompt Savings |
|-------------|-------|----------:|---------------:|
| Highly Recommended | SI, PI, CM | 4.0+ | 28%~40% |
| Conditional | CC, IC, GR | 2.4~3.1 | 4%~10% |
| Not Recommended | SA, SC, TS | 2.0~3.0 | -0.5%~1% |

**Key Insights**:
1. Summarization works well for long dialogues (≥4 turns) with long prompts (>2000 tokens).
2. Summarization harms short dialogues (≤2 turns) due to overhead > compression gains.
3. Current `-events 2` setting is too aggressive for short dialogues.

## QMSum On-Demand Summary

Evaluates `summary_ondemand` on meeting transcripts where supporting evidence is hidden by summary compression.

**Configuration**:
- Dataset: `QMSum` — `test / ALL / specific / support_distance_from_end >= 80`
- Evaluated Cases: `189`
- Model: `gpt-4o-mini`
- Avg Long Context Tokens: ~19K

**Key Results**:

| Metric | Long Context | Summary | Summary + On-Demand |
|--------|-------------:|--------:|--------------------:|
| ROUGE-L | 0.1930 | 0.1516 | 0.1770 |
| F1 | 0.3132 | 0.2238 | 0.2774 |
| Avg Prompt Tokens | 18,986 | 888 | 3,857 |

**Key Insights**:
1. On-demand retrieval recovers ROUGE-L by +0.0255 over plain summary (123 wins, 62 losses).
2. Prompt savings remain large at 76.69% versus long context.
3. The dominant tool path is `session_search → session_load` (1.94 calls/case average).

## LongMemEval On-Demand Summary

Evaluates `summary_ondemand` on realistic multi-session user/assistant dialogues at extreme context lengths.

**Configuration**:
- Dataset: `LongMemEval` — `longmemeval_s_cleaned.json / single-session-user`
- Evaluated Cases: `70`
- Model: `gpt-4o-mini`
- Avg Long Context Tokens: ~102K

**Key Results**:

| Metric | Long Context | Summary | Summary + On-Demand |
|--------|-------------:|--------:|--------------------:|
| ROUGE-L | 0.1159 | 0.0440 | **0.1711** |
| F1 | 0.1210 | 0.0547 | **0.1762** |
| Avg Prompt Tokens | 102,085 | 4,352 | 8,518 |

**Key Insights**:
1. On-demand retrieval **surpasses** long context (ROUGE-L 0.1711 vs 0.1159) — the model gets lost in ~102K tokens.
2. Uses only 8.3% of the tokens while achieving 48% higher ROUGE-L than the full-context baseline.
3. Wins 49 of 70 cases versus summary, and 43 of 70 cases versus long context.
4. Average tool calls: 1.01 per case (mostly single `session_search` without `session_load`).

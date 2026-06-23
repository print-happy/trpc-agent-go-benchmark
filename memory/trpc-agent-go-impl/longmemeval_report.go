//
// Tencent is pleased to support the open source community by making
// trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/metrics"
	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/scenarios"
)

func saveLMERunResult(outputDir string, result *lmeRunResult) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Printf("marshal LongMemEval result: %v", err)
		return
	}
	for _, name := range []string{"results.json", "checkpoint.json"} {
		path := filepath.Join(outputDir, name)
		if err := os.WriteFile(path, data, 0644); err != nil {
			log.Printf("write %s: %v", path, err)
		}
	}
}

func loadLMERunResult(outputDir string) (*lmeRunResult, error) {
	data, err := os.ReadFile(filepath.Join(outputDir, "checkpoint.json"))
	if err != nil {
		return nil, err
	}
	var result lmeRunResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func saveLMECaseLog(outputDir string, cr *lmeCaseResult) {
	path := filepath.Join(outputDir, cr.QuestionID+".log")
	var b strings.Builder
	fmt.Fprintf(&b, "QuestionID: %s\nQuestionType: %s\n", cr.QuestionID, cr.QuestionType)
	fmt.Fprintf(&b, "QuestionDate: %s\nCorrect: %v\n", cr.QuestionDate, cr.Correct)
	fmt.Fprintf(&b, "Metrics: accuracy=%.4f f1=%.4f bleu=%.4f rouge_l=%.4f\n",
		cr.Metrics.Accuracy, cr.Metrics.F1, cr.Metrics.BLEU, cr.Metrics.ROUGEL)
	fmt.Fprintf(&b, "\nQuestion:\n%s\n\nExpected:\n%s\n\nPredicted:\n%s\n",
		cr.Question, cr.Expected, cr.Predicted)
	if cr.Retrieval != nil {
		fmt.Fprintf(&b, "\nRetrieval Hits: %d\n", len(cr.Retrieval.Hits))
		for i, hit := range cr.Retrieval.Hits {
			fmt.Fprintf(&b, "[%d] session=%s turn=%s score=%.4f\n%s\n",
				i+1, hit.SessionID, hit.TurnID, hit.Score, hit.Text)
		}
	}
	if len(cr.ToolSteps) > 0 {
		fmt.Fprintf(&b, "\nTool Steps:\n")
		for _, step := range cr.ToolSteps {
			fmt.Fprintf(&b, "Step %d tokens=%d\n", step.Step, step.TotalTokens)
			for _, tc := range step.ToolCalls {
				result := tc.Result
				if len(cr.QATrace) == 0 {
					result = truncateLME(result, 600)
				}
				fmt.Fprintf(&b, "- %s args=%s result=%s\n", tc.Name, tc.Args, result)
			}
		}
	}
	if len(cr.QATrace) > 0 {
		fmt.Fprintf(&b, "\nQA Conversation:\n")
		for i, msg := range cr.QATrace {
			fmt.Fprintf(&b, "\n[%d] role=%s", i+1, msg.Role)
			if msg.Name != "" {
				fmt.Fprintf(&b, " name=%s", msg.Name)
			}
			if msg.Step > 0 {
				fmt.Fprintf(&b, " step=%d", msg.Step)
			}
			fmt.Fprintf(&b, "\n")
			for _, tc := range msg.ToolCalls {
				fmt.Fprintf(&b, "tool_call name=%s args=%s\n", tc.Name, tc.Args)
			}
			if msg.Content != "" {
				fmt.Fprintf(&b, "%s\n", msg.Content)
			}
		}
	}
	if err := os.WriteFile(path, []byte(b.String()), 0644); err != nil {
		log.Printf("write case log %s: %v", path, err)
	}
}

func printLMESummary(result *lmeRunResult) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 72))
	fmt.Printf("LongMemEval Memory Results - %s\n", result.Metadata.Scenario)
	fmt.Println(strings.Repeat("=", 72))
	fmt.Printf("Cases: %d/%d\n", result.Summary.CompletedCases, result.Summary.TotalCases)
	fmt.Printf("Accuracy: %.4f | Task-Avg Accuracy: %.4f\n",
		result.Summary.Overall.Accuracy, result.Summary.TaskAveragedAccuracy)
	fmt.Printf("F1/BLEU/ROUGE-L: %.4f / %.4f / %.4f\n",
		result.Summary.Overall.F1, result.Summary.Overall.BLEU, result.Summary.Overall.ROUGEL)
	fmt.Printf("Tokens prompt/completion/total: %d / %d / %d\n",
		result.Summary.TotalPromptTokens,
		result.Summary.TotalCompletionTokens,
		result.Summary.TotalTokens)
	if result.Summary.Retrieval != nil && result.Summary.Retrieval.Count > 0 {
		if m, ok := result.Summary.Retrieval.Turn[10]; ok {
			fmt.Printf("Turn Retrieval@10 recall_all=%.4f ndcg=%.4f\n", m.RecallAll, m.NDCGAny)
		}
	}
	fmt.Println(strings.Repeat("=", 72))
}

func writeLMEReports(
	rootDir string,
	cfg lmeRunConfig,
	scenarioTypes []scenarios.ScenarioType,
) error {
	results := make([]*lmeRunResult, 0, len(scenarioTypes)+1)
	seen := make(map[scenarios.ScenarioType]struct{}, len(scenarioTypes))
	if !lmeScenarioSelected(scenarioTypes, scenarios.ScenarioLongContext) {
		path := filepath.Join(
			lmeScenarioDir(rootDir, scenarios.ScenarioLongContext, ""),
			"results.json",
		)
		if _, err := os.Stat(path); err == nil {
			result, err := readLMERunResult(path)
			if err != nil {
				return err
			}
			results = append(results, result)
			seen[scenarios.ScenarioLongContext] = struct{}{}
		}
	}
	for _, scenarioType := range scenarioTypes {
		if _, ok := seen[scenarioType]; ok {
			continue
		}
		backend := ""
		if scenarioType == scenarios.ScenarioSessionRecall {
			backend = "session_pgvector"
		}
		if scenarioType == scenarios.ScenarioAuto ||
			scenarioType == scenarios.ScenarioAutoDeepSearch {
			backend = "pgvector"
		}
		path := filepath.Join(lmeScenarioDir(rootDir, scenarioType, backend), "results.json")
		result, err := readLMERunResult(path)
		if err != nil {
			return err
		}
		results = append(results, result)
	}
	en := renderLMEReport(results, cfg, false)
	zh := renderLMEReport(results, cfg, true)
	enName := "REPORT.md"
	zhName := "REPORT.zh_CN.md"
	if err := os.WriteFile(filepath.Join(rootDir, enName), []byte(en), 0644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(rootDir, zhName), []byte(zh), 0644); err != nil {
		return err
	}
	return nil
}

func lmeScenarioSelected(
	scenarioTypes []scenarios.ScenarioType,
	target scenarios.ScenarioType,
) bool {
	for _, scenarioType := range scenarioTypes {
		if scenarioType == target {
			return true
		}
	}
	return false
}

func readLMERunResult(path string) (*lmeRunResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read result %s: %w", path, err)
	}
	var result lmeRunResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("parse result %s: %w", path, err)
	}
	return &result, nil
}

func renderLMEReport(results []*lmeRunResult, cfg lmeRunConfig, zh bool) string {
	var b strings.Builder
	if zh {
		b.WriteString("# LongMemEval Memory Benchmark 报告\n\n")
		fmt.Fprintf(&b, "数据集：`%s`，模型：`%s`，Embedding：`%s`。\n\n",
			cfg.DatasetPath, cfg.ModelName, cfg.EmbedModelName)
		b.WriteString("## 总体结果\n\n")
		b.WriteString("| 场景 | 后端 | 样本 | Accuracy | F1 | BLEU | ROUGE-L | Prompt/QA | Calls/QA |\n")
	} else {
		b.WriteString("# LongMemEval Memory Benchmark Report\n\n")
		fmt.Fprintf(&b, "Dataset: `%s`; model: `%s`; embedding: `%s`.\n\n",
			cfg.DatasetPath, cfg.ModelName, cfg.EmbedModelName)
		b.WriteString("## Overall Results\n\n")
		b.WriteString("| Scenario | Backend | Cases | Accuracy | F1 | BLEU | ROUGE-L | Prompt/QA | Calls/QA |\n")
	}
	b.WriteString("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, result := range results {
		backend := result.Metadata.MemoryBackend
		if backend == "" {
			backend = "-"
		}
		fmt.Fprintf(&b, "| %s | %s | %d/%d | %.4f | %.4f | %.4f | %.4f | %.0f | %.2f |\n",
			result.Metadata.Scenario,
			backend,
			result.Summary.CompletedCases,
			result.Summary.TotalCases,
			result.Summary.Overall.Accuracy,
			result.Summary.Overall.F1,
			result.Summary.Overall.BLEU,
			result.Summary.Overall.ROUGEL,
			result.Summary.AvgPromptTokensPerQA,
			result.Summary.AvgLLMCallsPerQA,
		)
	}
	appendLMECostReport(&b, results, zh)
	if zh {
		b.WriteString("\n## 按问题类型\n\n")
	} else {
		b.WriteString("\n## By Question Type\n\n")
	}
	for _, result := range results {
		fmt.Fprintf(&b, "### %s\n\n", result.Metadata.Scenario)
		b.WriteString("| Type | Count | Accuracy | F1 | BLEU | ROUGE-L |\n")
		b.WriteString("| --- | ---: | ---: | ---: | ---: | ---: |\n")
		types := make([]string, 0, len(result.ByType))
		for t := range result.ByType {
			types = append(types, t)
		}
		sort.Strings(types)
		for _, t := range types {
			m := result.ByType[t]
			fmt.Fprintf(&b, "| %s | %d | %.4f | %.4f | %.4f | %.4f |\n",
				t, m.Count, m.Metrics.Accuracy, m.Metrics.F1, m.Metrics.BLEU, m.Metrics.ROUGEL)
		}
		b.WriteString("\n")
	}
	if zh {
		b.WriteString("## 检索指标\n\n")
	} else {
		b.WriteString("## Retrieval Metrics\n\n")
	}
	for _, result := range results {
		if result.Summary.Retrieval == nil || result.Summary.Retrieval.Count == 0 {
			continue
		}
		fmt.Fprintf(&b, "### %s\n\n", result.Metadata.Scenario)
		b.WriteString("| Level | K | Recall Any | Recall All | NDCG Any |\n")
		b.WriteString("| --- | ---: | ---: | ---: | ---: |\n")
		appendRetrievalRows(&b, "turn", result.Summary.Retrieval.Turn)
		appendRetrievalRows(&b, "session_from_turn", result.Summary.Retrieval.SessionFromTurn)
		b.WriteString("\n")
	}
	if zh {
		b.WriteString("## 公平性说明\n\n")
		b.WriteString("- 官方 yes/no judge accuracy 是主指标；F1/BLEU/ROUGE 是确定性辅助指标。\n")
		b.WriteString("- judge 输出无法严格解析为 yes/no 时评测中止，不做兜底补分。\n")
		b.WriteString("- 单样本失败会写 checkpoint 并中止；再次运行 `-resume` 可继续。\n")
	} else {
		b.WriteString("## Fairness Notes\n\n")
		b.WriteString("- Official yes/no judge accuracy is the primary metric; F1/BLEU/ROUGE are deterministic auxiliary metrics.\n")
		b.WriteString("- Judge output must parse as exact yes/no; there is no fallback scoring.\n")
		b.WriteString("- A case failure writes a checkpoint and stops; rerun with `-resume` to continue.\n")
	}
	return b.String()
}

func appendLMECostReport(
	b *strings.Builder,
	results []*lmeRunResult,
	zh bool,
) {
	hasCost := false
	for _, result := range results {
		if result != nil && result.Cost != nil {
			hasCost = true
			break
		}
	}
	if !hasCost {
		return
	}
	if zh {
		b.WriteString("\n## 模型调用成本总览\n\n")
	} else {
		b.WriteString("\n## Model Call Cost Summary\n\n")
	}
	b.WriteString("| Scenario | LLM Calls | LLM Tokens | Embedding Calls | Embedding Requests | Embedding Cache Hits | Embedding Tokens | Note |\n")
	b.WriteString("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
	for _, result := range results {
		cost := result.Cost
		if cost == nil {
			cost = newLMECostTracker().snapshot()
		}
		note := ""
		if cost.Partial {
			note = cost.PartialReason
		}
		fmt.Fprintf(
			b,
			"| %s | %d | %s | %d | %d | %d | %s | %s |\n",
			result.Metadata.Scenario,
			cost.LLM.Total.Calls,
			lmeCostTokensLabel(cost.LLM.Total),
			cost.Embedding.Total.Calls,
			cost.Embedding.Total.Requests,
			cost.Embedding.Total.CacheHits,
			lmeCostTokensLabel(cost.Embedding.Total),
			note,
		)
	}
	if zh {
		b.WriteString("\n## 分阶段模型成本\n\n")
	} else {
		b.WriteString("\n## Model Cost By Phase\n\n")
	}
	b.WriteString("| Scenario | Modality | Phase | Calls | Requests | Cache Hits | Prompt | Completion | Total | Cached | Tokens Known |\n")
	b.WriteString("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
	for _, result := range results {
		if result.Cost == nil {
			continue
		}
		appendLMECostPhaseRow(b, result, "llm", "memory_build", result.Cost.LLM.MemoryBuild)
		appendLMECostPhaseRow(b, result, "llm", "qa", result.Cost.LLM.QA)
		appendLMECostPhaseRow(b, result, "llm", "judge", result.Cost.LLM.Judge)
		appendLMECostPhaseRow(b, result, "embedding", "memory_build", result.Cost.Embedding.MemoryBuild)
		appendLMECostPhaseRow(b, result, "embedding", "qa_retrieval", result.Cost.Embedding.QARetrieval)
	}
	b.WriteString("\n")
}

func appendLMECostPhaseRow(
	b *strings.Builder,
	result *lmeRunResult,
	modality string,
	phase string,
	bucket lmeCostBucket,
) {
	fmt.Fprintf(
		b,
		"| %s | %s | %s | %d | %d | %d | %d | %d | %d | %d | %t |\n",
		result.Metadata.Scenario,
		modality,
		phase,
		bucket.Calls,
		bucket.Requests,
		bucket.CacheHits,
		bucket.PromptTokens,
		bucket.CompletionTokens,
		bucket.TotalTokens,
		bucket.CachedTokens,
		bucket.TokensKnown,
	)
}

func lmeCostTokensLabel(bucket lmeCostBucket) string {
	value := fmt.Sprintf("%d", bucket.TotalTokens)
	if !bucket.TokensKnown {
		return value + "?"
	}
	return value
}

func appendRetrievalRows(
	b *strings.Builder,
	level string,
	values metrics.RetrievalMetricsAtK,
) {
	keys := make([]int, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		m := values[k]
		fmt.Fprintf(b, "| %s | %d | %.4f | %.4f | %.4f |\n",
			level, k, m.RecallAny, m.RecallAll, m.NDCGAny)
	}
}

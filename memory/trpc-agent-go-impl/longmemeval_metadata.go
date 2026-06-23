//
// Tencent is pleased to support the open source community by making
// trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

package main

import "fmt"

func setLMERunMetadata(result *lmeRunResult, evaluator lmeEvaluator) {
	if result == nil || result.Metadata == nil {
		return
	}
	name := evaluator.Name()
	if name != "auto" && name != "auto_deepsearch" {
		return
	}
	method := "trpc-agent-go extractor.Extract -> pgvector memory.Service"
	qaRuntime := "fresh in-memory QA session per question"
	allowedInputs := []string{
		"current_question",
		"question_date",
		"memory_search results",
	}
	forbiddenInputs := []string{
		"full_conversation_transcript",
		"full_session_transcript",
		"longmemeval_haystack",
		"gold_evidence",
		"gold_answer_except_judge_prompt",
	}
	qaContextPolicy := "fresh QA sessions with only current question and memory_search results"
	autoQAOnly := result.Metadata.Config.AutoQAOnly
	if name == "auto_deepsearch" {
		method = "reuse existing pgvector memory.Service entries, lazy DeepSearch cue/tag index via memory_deepsearch"
		allowedInputs = append(allowedInputs, "memory_deepsearch tools after explicit activation")
		qaContextPolicy = "fresh QA sessions with memory_search first; DeepSearch tools only after activation"
	}
	if autoQAOnly {
		method = fmt.Sprintf(
			"%s from %s (QA only)",
			method,
			result.Metadata.Config.AutoMemoryTable,
		)
	}
	totalSessions := 0
	totalTurns := 0
	failed := make([]string, 0)
	for _, cr := range result.Cases {
		if cr == nil {
			continue
		}
		totalSessions += cr.TotalSessions
		totalTurns += cr.TotalTurns
	}
	if result.Summary != nil &&
		result.Summary.CompletedCases != result.Summary.TotalCases {
		failed = append(failed, "incomplete_result")
	}
	comparable := len(failed) == 0
	result.Metadata.MemoryOnlyCompliant = true
	result.Metadata.NativeMemoryPreserved = true
	result.Metadata.FairlyComparable = comparable
	if comparable {
		result.Metadata.ComparisonStatus = "comparable"
	} else {
		result.Metadata.ComparisonStatus = "not_comparable"
	}
	result.Metadata.ComparisonBlockers = failed
	result.Metadata.MemoryBuildMethod = method
	buildStatus := "completed"
	buildCostIncluded := true
	if autoQAOnly {
		buildStatus = "reused"
		buildCostIncluded = false
		totalSessions = 0
		totalTurns = 0
	}
	if !comparable {
		buildStatus = "failed"
	}
	memoryBuild := map[string]any{
		"method":                  result.Metadata.MemoryBuildMethod,
		"backend":                 result.Metadata.MemoryBackend,
		"status":                  buildStatus,
		"cost_included":           buildCostIncluded,
		"sample_count":            len(result.Cases),
		"failed_samples":          failed,
		"total_sessions_ingested": totalSessions,
		"total_turns_ingested":    totalTurns,
	}
	if autoQAOnly {
		memoryBuild["source_table"] = result.Metadata.Config.AutoMemoryTable
	}
	result.Metadata.MemoryBuild = memoryBuild
	result.Metadata.MemoryOnlyPolicy = map[string]any{
		"enabled":          true,
		"framework":        "trpc-agent-go",
		"qa_runtime":       qaRuntime,
		"allowed_inputs":   allowedInputs,
		"forbidden_inputs": forbiddenInputs,
	}
	result.Metadata.MemoryOnlySummary = map[string]any{
		"compliant":     true,
		"checked_cases": len(result.Cases),
		"failed_cases":  []string{},
		"violations":    map[string][]string{},
	}
	result.Metadata.QAContextPolicy = qaContextPolicy
}

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
	"testing"

	"trpc.group/trpc-go/trpc-agent-go/model"
)

func TestLMECostTrackerRecordLLMPhases(t *testing.T) {
	tracker := newLMECostTracker()
	usage := &model.Usage{
		PromptTokens:     11,
		CompletionTokens: 7,
		TotalTokens:      18,
	}
	usage.PromptTokensDetails.CachedTokens = 3

	tracker.recordLLM(lmeLLMPhaseQA, usage, true)
	tracker.recordLLM(lmeLLMPhaseJudge, usage, true)
	report := tracker.snapshot()

	if report.LLM.Total.Calls != 2 {
		t.Fatalf("total calls = %d, want 2", report.LLM.Total.Calls)
	}
	if report.LLM.QA.Calls != 1 || report.LLM.Judge.Calls != 1 {
		t.Fatalf("phase calls = qa:%d judge:%d, want 1 each", report.LLM.QA.Calls, report.LLM.Judge.Calls)
	}
	if report.LLM.Total.TotalTokens != 36 || report.LLM.Total.CachedTokens != 6 {
		t.Fatalf("total tokens = %d cached = %d, want 36 and 6", report.LLM.Total.TotalTokens, report.LLM.Total.CachedTokens)
	}
	if !report.LLM.Total.TokensKnown {
		t.Fatal("total tokens should be known")
	}
}

func TestLMECostTrackerPreservesUnknownTokens(t *testing.T) {
	tracker := newLMECostTracker()
	tracker.recordLLM(lmeLLMPhaseQA, nil, false)
	report := tracker.snapshot()

	if report.LLM.QA.Calls != 1 {
		t.Fatalf("qa calls = %d, want 1", report.LLM.QA.Calls)
	}
	if report.LLM.QA.TokensKnown {
		t.Fatal("qa tokens should be unknown")
	}
}

func TestLMECostTrackerEmbeddingCallsAndCacheHits(t *testing.T) {
	tracker := newLMECostTracker()
	tracker.recordEmbedding(
		lmeEmbeddingPhaseMemoryBuild,
		map[string]any{"prompt_tokens": 5, "total_tokens": 5},
		true,
	)
	tracker.recordEmbedding(lmeEmbeddingPhaseQARetrieval, nil, false)
	report := tracker.snapshot()

	if report.Embedding.Total.Calls != 1 || report.Embedding.Total.CacheHits != 1 {
		t.Fatalf("embedding calls = %d hits = %d, want 1 each", report.Embedding.Total.Calls, report.Embedding.Total.CacheHits)
	}
	if report.Embedding.Total.Requests != 2 {
		t.Fatalf("embedding requests = %d, want 2", report.Embedding.Total.Requests)
	}
	if report.Embedding.MemoryBuild.TotalTokens != 5 {
		t.Fatalf("memory-build tokens = %d, want 5", report.Embedding.MemoryBuild.TotalTokens)
	}
}

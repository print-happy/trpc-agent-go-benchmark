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
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/metrics"
	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/scenarios"
	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/event"
	"trpc.group/trpc-go/trpc-agent-go/memory"
	"trpc.group/trpc-go/trpc-agent-go/memory/deepsearch"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/session"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

func lmeScenarioDir(rootDir string, scenario scenarios.ScenarioType, backend string) string {
	if backend == "" {
		return filepath.Join(rootDir, string(scenario))
	}
	return filepath.Join(rootDir, fmt.Sprintf("%s_%s", scenario, backend))
}

func truncateLME(text string, limit int) string {
	text = strings.TrimSpace(text)
	if limit <= 0 || len(text) <= limit {
		return text
	}
	return text[:limit-3] + "..."
}

func parseCSV(raw string) []string {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func averageFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total float64
	for _, v := range values {
		total += v
	}
	return total / float64(len(values))
}

func metricsBool(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

func (a *lmeRetrievalAccumulator) add(cr *lmeCaseResult) {
	if cr == nil {
		return
	}
	if cr.Retrieval == nil {
		return
	}
	if cr.IsAbstention {
		a.skippedAbstention++
		return
	}
	if len(cr.Retrieval.CorrectTurns) == 0 {
		a.skippedNoTarget++
		return
	}
	if len(cr.Retrieval.TurnMetrics) == 0 {
		return
	}
	a.count++
	addRetrievalMetrics(a.turn, cr.Retrieval.TurnMetrics)
	addRetrievalMetrics(a.sessionFromTurn, cr.Retrieval.SessionMetrics)
}

func (a *lmeRetrievalAccumulator) summary() *lmeRetrievalSummary {
	if a.count == 0 && a.skippedAbstention == 0 && a.skippedNoTarget == 0 {
		return nil
	}
	return &lmeRetrievalSummary{
		Count:             a.count,
		SkippedAbstention: a.skippedAbstention,
		SkippedNoTarget:   a.skippedNoTarget,
		Turn:              averageRetrievalMetrics(a.turn, a.count),
		SessionFromTurn:   averageRetrievalMetrics(a.sessionFromTurn, a.count),
	}
}

type lmeRetrievalAccumulator struct {
	count             int
	skippedAbstention int
	skippedNoTarget   int
	turn              map[int]metrics.RetrievalMetrics
	sessionFromTurn   map[int]metrics.RetrievalMetrics
}

func newLMERetrievalAccumulator() *lmeRetrievalAccumulator {
	return &lmeRetrievalAccumulator{
		turn:            make(map[int]metrics.RetrievalMetrics),
		sessionFromTurn: make(map[int]metrics.RetrievalMetrics),
	}
}

func addRetrievalMetrics(
	dst map[int]metrics.RetrievalMetrics,
	src metrics.RetrievalMetricsAtK,
) {
	for k, v := range src {
		cur := dst[k]
		cur.RecallAny += v.RecallAny
		cur.RecallAll += v.RecallAll
		cur.NDCGAny += v.NDCGAny
		dst[k] = cur
	}
}

func averageRetrievalMetrics(
	src map[int]metrics.RetrievalMetrics,
	count int,
) metrics.RetrievalMetricsAtK {
	out := make(metrics.RetrievalMetricsAtK, len(src))
	if count == 0 {
		return out
	}
	for k, v := range src {
		v.RecallAny /= float64(count)
		v.RecallAll /= float64(count)
		v.NDCGAny /= float64(count)
		out[k] = v
	}
	return out
}

func (s *lmeNoAutoMemoryService) AddMemory(
	ctx context.Context,
	userKey memory.UserKey,
	mem string,
	topics []string,
	opts ...memory.AddOption,
) error {
	return s.inner.AddMemory(ctx, userKey, mem, topics, opts...)
}

func (s *lmeNoAutoMemoryService) UpdateMemory(
	ctx context.Context,
	key memory.Key,
	mem string,
	topics []string,
	opts ...memory.UpdateOption,
) error {
	return s.inner.UpdateMemory(ctx, key, mem, topics, opts...)
}

func (s *lmeNoAutoMemoryService) DeleteMemory(ctx context.Context, key memory.Key) error {
	return s.inner.DeleteMemory(ctx, key)
}

func (s *lmeNoAutoMemoryService) ClearMemories(
	ctx context.Context,
	userKey memory.UserKey,
) error {
	return s.inner.ClearMemories(ctx, userKey)
}

func (s *lmeNoAutoMemoryService) ReadMemories(
	ctx context.Context,
	userKey memory.UserKey,
	limit int,
) ([]*memory.Entry, error) {
	return s.inner.ReadMemories(ctx, userKey, limit)
}

func (s *lmeNoAutoMemoryService) SearchMemories(
	ctx context.Context,
	userKey memory.UserKey,
	query string,
	opts ...memory.SearchOption,
) ([]*memory.Entry, error) {
	return s.inner.SearchMemories(ctx, userKey, query, opts...)
}

func (s *lmeNoAutoMemoryService) EnsureIndex(
	ctx context.Context,
	userKey memory.UserKey,
) error {
	deepSearchSvc, err := s.deepSearchService()
	if err != nil {
		return err
	}
	return deepSearchSvc.EnsureIndex(ctx, userKey)
}

func (s *lmeNoAutoMemoryService) IndexDocuments(
	ctx context.Context,
	req deepsearch.IndexRequest,
) error {
	deepSearchSvc, err := s.deepSearchService()
	if err != nil {
		return err
	}
	return deepSearchSvc.IndexDocuments(ctx, req)
}

func (s *lmeNoAutoMemoryService) SearchCues(
	ctx context.Context,
	req deepsearch.CueSearchRequest,
) (*deepsearch.CueSearchResult, error) {
	deepSearchSvc, err := s.deepSearchService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.SearchCues(ctx, req)
}

func (s *lmeNoAutoMemoryService) ExpandTags(
	ctx context.Context,
	req deepsearch.TagExpandRequest,
) (*deepsearch.TagExpandResult, error) {
	deepSearchSvc, err := s.deepSearchService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.ExpandTags(ctx, req)
}

func (s *lmeNoAutoMemoryService) LoadContents(
	ctx context.Context,
	req deepsearch.ContentLoadRequest,
) (*deepsearch.ContentLoadResult, error) {
	deepSearchSvc, err := s.deepSearchService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.LoadContents(ctx, req)
}

func (s *lmeNoAutoMemoryService) DeleteDocuments(
	ctx context.Context,
	req deepsearch.DeleteRequest,
) error {
	deepSearchSvc, err := s.deepSearchService()
	if err != nil {
		return err
	}
	return deepSearchSvc.DeleteDocuments(ctx, req)
}

func (s *lmeNoAutoMemoryService) EdgesByTag(
	ctx context.Context,
	req deepsearch.EdgesByTagRequest,
) (*deepsearch.EdgesByTagResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.EdgesByTag(ctx, req)
}

func (s *lmeNoAutoMemoryService) QueryConversationTime(
	ctx context.Context,
	req deepsearch.QueryConversationTimeRequest,
) (*deepsearch.QueryResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.QueryConversationTime(ctx, req)
}

func (s *lmeNoAutoMemoryService) QueryEventKeywords(
	ctx context.Context,
	req deepsearch.QueryEventKeywordsRequest,
) (*deepsearch.QueryResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.QueryEventKeywords(ctx, req)
}

func (s *lmeNoAutoMemoryService) QueryEventContext(
	ctx context.Context,
	req deepsearch.QueryEventContextRequest,
) (*deepsearch.QueryResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.QueryEventContext(ctx, req)
}

func (s *lmeNoAutoMemoryService) QueryPersonalInformation(
	ctx context.Context,
	req deepsearch.QueryPersonalInformationRequest,
) (*deepsearch.QueryResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.QueryPersonalInformation(ctx, req)
}

func (s *lmeNoAutoMemoryService) QueryPersonalAspect(
	ctx context.Context,
	req deepsearch.QueryPersonalAspectRequest,
) (*deepsearch.QueryResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.QueryPersonalAspect(ctx, req)
}

func (s *lmeNoAutoMemoryService) QueryTopicEvents(
	ctx context.Context,
	req deepsearch.QueryTopicEventsRequest,
) (*deepsearch.QueryResult, error) {
	deepSearchSvc, err := s.deepSearchQueryService()
	if err != nil {
		return nil, err
	}
	return deepSearchSvc.QueryTopicEvents(ctx, req)
}

func (s *lmeNoAutoMemoryService) deepSearchService() (deepsearch.Service, error) {
	deepSearchSvc, ok := s.inner.(deepsearch.Service)
	if !ok || deepSearchSvc == nil {
		return nil, fmt.Errorf("memory deepsearch service is not available")
	}
	return deepSearchSvc, nil
}

func (s *lmeNoAutoMemoryService) deepSearchQueryService() (deepsearch.QueryService, error) {
	deepSearchSvc, ok := s.inner.(deepsearch.QueryService)
	if !ok || deepSearchSvc == nil {
		return nil, fmt.Errorf("memory deepsearch query service is not available")
	}
	return deepSearchSvc, nil
}

func (s *lmeNoAutoMemoryService) Tools() []tool.Tool {
	return s.inner.Tools()
}

func (s *lmeNoAutoMemoryService) EnqueueAutoMemoryJob(
	_ context.Context,
	_ *session.Session,
) error {
	return nil
}

func (s *lmeNoAutoMemoryService) Close() error {
	return nil
}

func (lmeSeedAgent) Run(
	ctx context.Context,
	invocation *agent.Invocation,
) (<-chan *event.Event, error) {
	ch := make(chan *event.Event, 1)
	go func() {
		defer close(ch)
		if invocation == nil {
			return
		}
		_ = event.EmitEvent(ctx, ch, event.NewResponseEvent(
			invocation.InvocationID,
			lmeSeedAgentName,
			&model.Response{
				Done: true,
				Choices: []model.Choice{{
					Message: model.NewAssistantMessage("OK."),
				}},
			},
		))
	}()
	return ch, nil
}

func (lmeSeedAgent) Tools() []tool.Tool { return nil }

func (lmeSeedAgent) Info() agent.Info {
	return agent.Info{Name: lmeSeedAgentName, Description: "LongMemEval seed agent."}
}

func (lmeSeedAgent) SubAgents() []agent.Agent { return nil }

func (lmeSeedAgent) FindSubAgent(_ string) agent.Agent { return nil }

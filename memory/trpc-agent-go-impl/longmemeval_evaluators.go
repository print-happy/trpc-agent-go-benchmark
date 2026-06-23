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
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/dataset"
	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/metrics"
	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/scenarios"
	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/agent/llmagent"
	"trpc.group/trpc-go/trpc-agent-go/event"
	"trpc.group/trpc-go/trpc-agent-go/memory"
	"trpc.group/trpc-go/trpc-agent-go/memory/extractor"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/runner"
	"trpc.group/trpc-go/trpc-agent-go/session"
	sessioninmemory "trpc.group/trpc-go/trpc-agent-go/session/inmemory"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

type lmeEvaluator interface {
	Name() string
	Evaluate(
		ctx context.Context,
		inst *dataset.LongMemEvalInstance,
	) (*lmeCaseResult, error)
	Close() error
}

func (e *lmeLongContextEvaluator) Name() string { return "long_context" }

func (e *lmeLongContextEvaluator) Close() error { return nil }

func (e *lmeLongContextEvaluator) Evaluate(
	ctx context.Context,
	inst *dataset.LongMemEvalInstance,
) (*lmeCaseResult, error) {
	start := time.Now()
	prompt, err := buildLMELongContextPrompt(inst)
	if err != nil {
		return nil, err
	}
	mr, err := runLMEModelWithRetry(ctx, e.llm, []model.Message{
		model.NewUserMessage(prompt),
	}, e.cfg.AnswerMaxTokens, e.cfg.MaxRetries)
	if err != nil {
		return nil, err
	}
	return buildLMECaseResult(
		ctx, e.llm, e.cfg, inst, strings.TrimSpace(mr.Text),
		time.Since(start), modelUsageToScenarioUsage(mr.Usage), mr.RetryCount, nil, nil,
		nil,
	)
}

func (e *lmeSessionRecallEvaluator) Name() string { return "session_recall" }

func (e *lmeSessionRecallEvaluator) Close() error {
	if e.svc == nil {
		return nil
	}
	return e.svc.Close()
}

func (e *lmeSessionRecallEvaluator) Evaluate(
	ctx context.Context,
	inst *dataset.LongMemEvalInstance,
) (*lmeCaseResult, error) {
	start := time.Now()
	userID := inst.QuestionID
	sessionIDs, err := seedLMESessions(
		ctx,
		e.svc,
		lmeAppSessionRecall,
		userID,
		inst,
		e.cfg.SessionRecallUserOnly,
	)
	if err != nil {
		return nil, err
	}
	defer cleanupLMESessions(context.Background(), e.svc, lmeAppSessionRecall, userID, sessionIDs)
	retrieval, err := e.search(ctx, userID, inst)
	if err != nil {
		return nil, err
	}
	prompt := buildLMESessionRecallPrompt(inst, retrieval.Hits)
	mr, err := runLMEModelWithRetry(ctx, e.llm, []model.Message{
		model.NewUserMessage(prompt),
	}, e.cfg.AnswerMaxTokens, e.cfg.MaxRetries)
	if err != nil {
		return nil, err
	}
	return buildLMECaseResult(
		ctx, e.llm, e.cfg, inst, strings.TrimSpace(mr.Text),
		time.Since(start), modelUsageToScenarioUsage(mr.Usage), mr.RetryCount, retrieval, nil,
		nil,
	)
}

func (e *lmeSessionRecallEvaluator) search(
	ctx context.Context,
	userID string,
	inst *dataset.LongMemEvalInstance,
) (*lmeRetrievalTrace, error) {
	searchable, ok := e.svc.(session.SearchableService)
	if !ok {
		return nil, fmt.Errorf("session service does not implement SearchableService")
	}
	req := session.EventSearchRequest{
		Query: strings.TrimSpace(inst.Question),
		UserKey: session.UserKey{
			AppName: lmeAppSessionRecall,
			UserID:  userID,
		},
		MaxResults: e.cfg.SessionRecallResults,
		MinScore:   e.cfg.SessionRecallMinScore,
		Roles:      []model.Role{model.RoleUser},
		SearchMode: session.SearchModeHybrid,
	}
	results, err := searchable.SearchEvents(ctx, req)
	if err != nil {
		return nil, err
	}
	trace := &lmeRetrievalTrace{
		Query:      req.Query,
		MaxResults: req.MaxResults,
		MinScore:   req.MinScore,
		SearchMode: req.SearchMode,
		Hits:       make([]lmeRetrievalHit, 0, len(results)),
	}
	rankedTurns := make([]string, 0, len(results))
	for _, result := range results {
		sessionID := strings.TrimPrefix(result.SessionKey.SessionID, "seed-")
		turnID := dataset.LongMemEvalTurnIDFromEventExtensions(result.Event.Extensions)
		if turnID == "" {
			turnID = result.Event.ID
		}
		rankedTurns = append(rankedTurns, turnID)
		trace.Hits = append(trace.Hits, lmeRetrievalHit{
			SessionID:   sessionID,
			TurnID:      turnID,
			EventID:     result.Event.ID,
			Role:        result.Role,
			Text:        result.Text,
			Score:       result.Score,
			DenseScore:  result.DenseScore,
			SparseScore: result.SparseScore,
		})
	}
	correctTurns := inst.EvidenceTurnIDs()
	corpusTurns := lmeCorpusTurnIDs(inst)
	trace.CorrectTurns = correctTurns
	if !inst.IsAbstention() && len(correctTurns) > 0 {
		trace.TurnMetrics = metrics.EvaluateRetrieval(
			rankedTurns,
			correctTurns,
			corpusTurns,
			lmeRetrievalCutoffs,
		)
		trace.SessionMetrics = metrics.EvaluateRetrievalTurnToSession(
			rankedTurns,
			correctTurns,
			corpusTurns,
			lmeRetrievalCutoffs,
		)
	}
	return trace, nil
}

func searchLMESessionEvents(
	ctx context.Context,
	svc session.Service,
	appName, userID string,
	inst *dataset.LongMemEvalInstance,
	cfg lmeRunConfig,
) (*lmeRetrievalTrace, error) {
	searchable, ok := svc.(session.SearchableService)
	if !ok {
		return nil, fmt.Errorf("session service does not implement SearchableService")
	}
	req := session.EventSearchRequest{
		Query: strings.TrimSpace(inst.Question),
		UserKey: session.UserKey{
			AppName: appName,
			UserID:  userID,
		},
		MaxResults: cfg.SessionRecallResults,
		MinScore:   cfg.SessionRecallMinScore,
		Roles:      []model.Role{model.RoleUser},
		SearchMode: session.SearchModeHybrid,
	}
	results, err := searchable.SearchEvents(ctx, req)
	if err != nil {
		return nil, err
	}
	trace := &lmeRetrievalTrace{
		Query:      req.Query,
		MaxResults: req.MaxResults,
		MinScore:   req.MinScore,
		SearchMode: req.SearchMode,
		Hits:       make([]lmeRetrievalHit, 0, len(results)),
	}
	rankedTurns := make([]string, 0, len(results))
	for _, result := range results {
		sessionID := strings.TrimPrefix(result.SessionKey.SessionID, "seed-")
		turnID := dataset.LongMemEvalTurnIDFromEventExtensions(result.Event.Extensions)
		if turnID == "" {
			turnID = result.Event.ID
		}
		rankedTurns = append(rankedTurns, turnID)
		trace.Hits = append(trace.Hits, lmeRetrievalHit{
			SessionID:   sessionID,
			TurnID:      turnID,
			EventID:     result.Event.ID,
			Role:        result.Role,
			Text:        result.Text,
			Score:       result.Score,
			DenseScore:  result.DenseScore,
			SparseScore: result.SparseScore,
		})
	}
	correctTurns := inst.EvidenceTurnIDs()
	corpusTurns := lmeCorpusTurnIDs(inst)
	trace.CorrectTurns = correctTurns
	if !inst.IsAbstention() && len(correctTurns) > 0 {
		trace.TurnMetrics = metrics.EvaluateRetrieval(
			rankedTurns,
			correctTurns,
			corpusTurns,
			lmeRetrievalCutoffs,
		)
		trace.SessionMetrics = metrics.EvaluateRetrievalTurnToSession(
			rankedTurns,
			correctTurns,
			corpusTurns,
			lmeRetrievalCutoffs,
		)
	}
	return trace, nil
}

func (e *lmeAutoEvaluator) Name() string {
	if e.name != "" {
		return e.name
	}
	return "auto"
}

func (e *lmeAutoEvaluator) Close() error {
	if e.mem == nil {
		return nil
	}
	return e.mem.Close()
}

func (e *lmeAutoEvaluator) Evaluate(
	ctx context.Context,
	inst *dataset.LongMemEvalInstance,
) (*lmeCaseResult, error) {
	ctx = withLMECostTracker(ctx, e.cost)
	start := time.Now()
	userKey := memory.UserKey{AppName: lmeAppAuto, UserID: inst.QuestionID}
	if e.cfg.AutoQAOnly {
		if err := requireLMEAutoMemories(ctx, e.mem, userKey); err != nil {
			return nil, err
		}
	} else {
		if err := e.mem.ClearMemories(ctx, userKey); err != nil {
			return nil, fmt.Errorf("clear memories: %w", err)
		}
		if err := seedLMEAutoMemories(
			ctx,
			e.extractor,
			e.mem,
			e.cfg,
			lmeAppAuto,
			userKey.UserID,
			inst,
		); err != nil {
			return nil, err
		}
	}
	qaMem := &lmeNoAutoMemoryService{inner: e.mem}
	agent := e.newQAAgent(e.qaTools(qaMem))
	qaRunner := runner.NewRunner(
		lmeAppAuto,
		agent,
		runner.WithSessionService(sessioninmemory.NewSessionService()),
		runner.WithMemoryService(qaMem),
	)
	defer qaRunner.Close()
	msg := model.NewUserMessage(
		fmt.Sprintf("Current Date: %s\nQuestion: %s", inst.QuestionDate, inst.Question),
	)
	qaCtx := withLMEEmbeddingPhase(ctx, lmeEmbeddingPhaseQARetrieval)
	cr, err := runLMERunnerWithRetry(qaCtx, e.cfg.MaxRetries, func() (<-chan *event.Event, error) {
		return qaRunner.Run(qaCtx, userKey.UserID, "qa-"+inst.QuestionID, msg)
	})
	if err != nil {
		return nil, err
	}
	return buildLMECaseResult(
		ctx, e.judgeLLM, e.cfg, inst, strings.TrimSpace(cr.Text),
		time.Since(start), &cr.Usage, cr.RetryCount,
		nil, cr.Steps, lmeQAConversationTrace(e.cfg, msg, cr.Trace),
	)
}

func requireLMEAutoMemories(
	ctx context.Context,
	mem memory.Service,
	userKey memory.UserKey,
) error {
	entries, err := mem.ReadMemories(ctx, userKey, 1)
	if err != nil {
		return fmt.Errorf("read QA-only auto memories: %w", err)
	}
	if len(entries) == 0 {
		return fmt.Errorf(
			"QA-only auto memories missing for %s/%s",
			userKey.AppName,
			userKey.UserID,
		)
	}
	return nil
}

func (e *lmeAutoEvaluator) CostReport() *lmeCostReport {
	if e.cost == nil {
		return nil
	}
	return e.cost.snapshot()
}

func seedLMEAutoMemories(
	ctx context.Context,
	ext extractor.MemoryExtractor,
	mem memory.Service,
	cfg lmeRunConfig,
	appName string,
	userID string,
	inst *dataset.LongMemEvalInstance,
) error {
	userKey := memory.UserKey{AppName: appName, UserID: userID}
	for i := range inst.HaystackSessions {
		seedCtx := ctx
		if t, ok := parseLMETime(inst.HaystackDates[i]); ok {
			seedCtx = extractor.WithReferenceDate(seedCtx, t)
		}
		if err := extractLMEAutoSession(seedCtx, ext, mem, cfg, userKey, lmeSessionMessages(inst, i)); err != nil {
			return fmt.Errorf("extract session %s: %w", inst.HaystackSessionIDs[i], err)
		}
	}
	return nil
}

func extractLMEAutoSession(
	ctx context.Context,
	ext extractor.MemoryExtractor,
	mem memory.Service,
	cfg lmeRunConfig,
	userKey memory.UserKey,
	messages []model.Message,
) error {
	existing, err := mem.ReadMemories(ctx, userKey, lmeMemoryReadLimit)
	if err != nil {
		return fmt.Errorf("read existing memories: %w", err)
	}
	ops, retries, err := runLMEExtractWithRetry(
		ctx,
		cfg.MaxRetries,
		func() ([]*extractor.Operation, error) {
			return ext.Extract(ctx, messages, existing)
		},
	)
	if err != nil {
		return err
	}
	if retries > 0 {
		log.Printf("LongMemEval auto extraction succeeded after %d retries", retries)
	}
	for _, op := range ops {
		opCtx := withLMEEmbeddingPhase(ctx, lmeEmbeddingPhaseMemoryBuild)
		if err := applyLMEAutoOperation(opCtx, mem, userKey, op); err != nil {
			return err
		}
	}
	return nil
}

func applyLMEAutoOperation(
	ctx context.Context,
	mem memory.Service,
	userKey memory.UserKey,
	op *extractor.Operation,
) error {
	if op == nil {
		return nil
	}
	metadata := lmeOperationMetadata(op)
	switch op.Type {
	case extractor.OperationAdd:
		return mem.AddMemory(ctx, userKey, op.Memory, op.Topics, memory.WithMetadata(metadata))
	case extractor.OperationUpdate:
		key := memory.Key{
			AppName:  userKey.AppName,
			UserID:   userKey.UserID,
			MemoryID: op.MemoryID,
		}
		err := mem.UpdateMemory(ctx, key, op.Memory, op.Topics, memory.WithUpdateMetadata(metadata))
		if err == nil {
			return nil
		}
		if strings.Contains(strings.ToLower(err.Error()), "not found") {
			return mem.AddMemory(ctx, userKey, op.Memory, op.Topics, memory.WithMetadata(metadata))
		}
		return err
	case extractor.OperationDelete:
		key := memory.Key{
			AppName:  userKey.AppName,
			UserID:   userKey.UserID,
			MemoryID: op.MemoryID,
		}
		return mem.DeleteMemory(ctx, key)
	case extractor.OperationClear:
		return mem.ClearMemories(ctx, userKey)
	default:
		return fmt.Errorf("unknown memory operation type %q", op.Type)
	}
}

func lmeOperationMetadata(op *extractor.Operation) *memory.Metadata {
	kind := op.MemoryKind
	if kind == "" {
		kind = memory.KindFact
	}
	return &memory.Metadata{
		Kind:         kind,
		EventTime:    op.EventTime,
		Participants: op.Participants,
		Location:     op.Location,
	}
}

func (e *lmeAutoEvaluator) newQAAgent(tools []tool.Tool) agent.Agent {
	maxTokens := e.cfg.AnswerMaxTokens
	temp := 0.0
	instruction := e.qaInstruction()
	options := []llmagent.Option{
		llmagent.WithModel(e.qaLLM),
		llmagent.WithInstruction(instruction),
		llmagent.WithTools(tools),
		llmagent.WithGenerationConfig(model.GenerationConfig{
			Stream:      false,
			MaxTokens:   &maxTokens,
			Temperature: &temp,
		}),
		llmagent.WithMaxToolIterations(8),
	}
	return llmagent.New(lmeQAAgentName, options...)
}

func (e *lmeAutoEvaluator) qaInstruction() string {
	return `You are a memory retrieval assistant for LongMemEval.

Rules:
- Use memory_search before answering.
- Use short keyword queries with names, entities, dates, and topics from the question.
- Do not use kind filters.
- Answer only from retrieved memories.
- If the retrieved memories do not contain enough information, say that the information is not available.
- Output the direct answer only.`
}

func (e *lmeAutoEvaluator) qaTools(mem memory.Service) []tool.Tool {
	return mem.Tools()
}

func buildLMECaseResult(
	ctx context.Context,
	judge model.Model,
	cfg lmeRunConfig,
	inst *dataset.LongMemEvalInstance,
	predicted string,
	latency time.Duration,
	tokenUsage *scenarios.TokenUsage,
	retryCount int,
	retrieval *lmeRetrievalTrace,
	steps []lmeStepTrace,
	qaTrace []lmeMessageTrace,
) (*lmeCaseResult, error) {
	answerMetrics := metrics.CalculateAnswerMetrics(predicted, inst.Answer)
	correct, judgeUsage, judgeRetries, err := judgeLMEAnswer(ctx, judge, cfg, inst, predicted)
	if err != nil {
		return nil, err
	}
	answerMetrics.Accuracy = metricsBool(correct)
	if tokenUsage == nil {
		tokenUsage = &scenarios.TokenUsage{}
	}
	if judgeUsage != nil {
		tokenUsage.Add(*modelUsageToScenarioUsage(judgeUsage))
	}
	return &lmeCaseResult{
		QuestionID:    inst.QuestionID,
		QuestionType:  inst.QuestionType,
		Question:      inst.Question,
		QuestionDate:  inst.QuestionDate,
		Expected:      inst.Answer,
		Predicted:     predicted,
		IsAbstention:  inst.IsAbstention(),
		Correct:       correct,
		Metrics:       answerMetrics,
		LatencyMs:     latency.Milliseconds(),
		TokenUsage:    tokenUsage,
		RetryCount:    retryCount + judgeRetries,
		TotalTurns:    inst.TotalTurns(),
		TotalSessions: len(inst.HaystackSessions),
		Retrieval:     retrieval,
		ToolSteps:     steps,
		QATrace:       qaTrace,
	}, nil
}

func lmeQAConversationTrace(
	cfg lmeRunConfig,
	userMsg model.Message,
	events []lmeMessageTrace,
) []lmeMessageTrace {
	if !cfg.FullQALog {
		return nil
	}
	trace := make([]lmeMessageTrace, 0, len(events)+1)
	trace = append(trace, lmeMessageTrace{
		Role:    string(userMsg.Role),
		Content: userMsg.Content,
	})
	return append(trace, events...)
}

func judgeLMEAnswer(
	ctx context.Context,
	judge model.Model,
	cfg lmeRunConfig,
	inst *dataset.LongMemEvalInstance,
	predicted string,
) (bool, *model.Usage, int, error) {
	prompt, err := metrics.LongMemEvalJudgePrompt(
		inst.QuestionType,
		inst.Question,
		inst.Answer,
		predicted,
		inst.IsAbstention(),
	)
	if err != nil {
		return false, nil, 0, err
	}
	mr, err := runLMEModelWithRetry(ctx, judge, []model.Message{
		model.NewUserMessage(prompt),
	}, cfg.JudgeMaxTokens, cfg.MaxRetries)
	if err != nil {
		return false, nil, mr.RetryCount, fmt.Errorf("judge model: %w", err)
	}
	label, err := metrics.ParseLongMemEvalJudgeLabel(mr.Text)
	if err != nil {
		return false, mr.Usage, mr.RetryCount, err
	}
	return label, mr.Usage, mr.RetryCount, nil
}

func buildLMELongContextPrompt(inst *dataset.LongMemEvalInstance) (string, error) {
	history, err := lmeHistoryJSON(inst)
	if err != nil {
		return "", err
	}
	template := "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n%s\n\nCurrent Date: %s\nQuestion: %s\nAnswer:"
	return fmt.Sprintf(template, history, inst.QuestionDate, inst.Question), nil
}

func buildLMESessionRecallPrompt(
	inst *dataset.LongMemEvalInstance,
	hits []lmeRetrievalHit,
) string {
	var b strings.Builder
	b.WriteString("I will give you recalled history events between you and a user. ")
	b.WriteString("Please answer the question based on the recalled history only.\n\n")
	b.WriteString("Recalled History:\n")
	for i, hit := range hits {
		fmt.Fprintf(&b, "\n### Event %d\nSession ID: %s\nContent: %s\n", i+1, hit.SessionID, hit.Text)
	}
	fmt.Fprintf(&b, "\nCurrent Date: %s\nQuestion: %s\nAnswer:", inst.QuestionDate, inst.Question)
	return b.String()
}

func lmeHistoryJSON(inst *dataset.LongMemEvalInstance) (string, error) {
	type turn struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type sess struct {
		SessionID string `json:"session_id"`
		Date      string `json:"date"`
		Turns     []turn `json:"turns"`
	}
	out := make([]sess, 0, len(inst.HaystackSessions))
	for i, sessionTurns := range inst.HaystackSessions {
		s := sess{
			SessionID: inst.HaystackSessionIDs[i],
			Date:      inst.HaystackDates[i],
			Turns:     make([]turn, 0, len(sessionTurns)),
		}
		for _, t := range sessionTurns {
			s.Turns = append(s.Turns, turn{Role: t.Role, Content: t.Content})
		}
		out = append(out, s)
	}
	data, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("marshal LongMemEval history: %w", err)
	}
	return string(data), nil
}

func seedLMESessions(
	ctx context.Context,
	svc session.Service,
	appName, userID string,
	inst *dataset.LongMemEvalInstance,
	userOnly bool,
) ([]string, error) {
	if err := dataset.ValidateLongMemEvalEvidenceMapping(inst); err != nil {
		return nil, err
	}
	sessionIDs := make([]string, 0, len(inst.HaystackSessions))
	for i := range inst.HaystackSessions {
		sessionID := "seed-" + inst.HaystackSessionIDs[i]
		key := session.Key{AppName: appName, UserID: userID, SessionID: sessionID}
		_ = svc.DeleteSession(ctx, key)
		sess, err := svc.CreateSession(ctx, key, nil)
		if err != nil {
			return nil, fmt.Errorf("create session %s: %w", sessionID, err)
		}
		for turnIdx, turn := range inst.HaystackSessions[i] {
			if userOnly && turn.Role != "user" {
				continue
			}
			content := strings.TrimSpace(turn.Content)
			if content == "" {
				continue
			}
			role := model.RoleUser
			if turn.Role == "assistant" {
				role = model.RoleAssistant
			}
			msg := model.Message{
				Role:    role,
				Content: fmt.Sprintf("[SessionDate: %s] %s", inst.HaystackDates[i], content),
			}
			eventID := dataset.LongMemEvalTurnID(
				inst.HaystackSessionIDs[i],
				turnIdx,
				turn.HasAnswer,
			)
			evt := event.New(
				eventID,
				lmeSeedAgentName,
				event.WithResponse(&model.Response{
					Done: true,
					Choices: []model.Choice{{
						Message: msg,
					}},
				}),
				event.WithExtension(dataset.EventExtensionLongMemEvalTurnID, eventID),
			)
			evt.ID = eventID
			if err := svc.AppendEvent(ctx, sess, evt); err != nil {
				return nil, fmt.Errorf("append event %s: %w", eventID, err)
			}
		}
		sessionIDs = append(sessionIDs, sessionID)
	}
	return sessionIDs, nil
}

func lmeSessionMessages(inst *dataset.LongMemEvalInstance, sessionIdx int) []model.Message {
	turns := inst.HaystackSessions[sessionIdx]
	date := inst.HaystackDates[sessionIdx]
	msgs := make([]model.Message, 0, len(turns)+1)
	msgs = append(msgs, model.NewSystemMessage("SessionDate: "+date))
	for _, turn := range turns {
		content := strings.TrimSpace(turn.Content)
		if content == "" {
			continue
		}
		content = fmt.Sprintf("[SessionDate: %s] %s", date, content)
		role := model.RoleUser
		if turn.Role == "assistant" {
			role = model.RoleAssistant
		}
		msgs = append(msgs, model.Message{Role: role, Content: content})
	}
	return msgs
}

func cleanupLMESessions(
	ctx context.Context,
	svc session.Service,
	appName, userID string,
	sessionIDs []string,
) {
	for _, sessionID := range sessionIDs {
		_ = svc.DeleteSession(ctx, session.Key{
			AppName:   appName,
			UserID:    userID,
			SessionID: sessionID,
		})
	}
}

func lmeCorpusTurnIDs(inst *dataset.LongMemEvalInstance) []string {
	return dataset.LongMemEvalCorpusTurnIDs(inst)
}

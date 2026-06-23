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
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/dataset"
	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/metrics"
	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/scenarios"
	"trpc.group/trpc-go/trpc-agent-go/memory"
	"trpc.group/trpc-go/trpc-agent-go/memory/extractor"
	"trpc.group/trpc-go/trpc-agent-go/model"
	openaimodel "trpc.group/trpc-go/trpc-agent-go/model/openai"
	"trpc.group/trpc-go/trpc-agent-go/session"
)

const (
	lmeDatasetFormat                 = "longmemeval"
	lmeAppLongContext                = "memory-lme-long-context"
	lmeAppSessionRecall              = "memory-lme-session-recall"
	lmeAppAuto                       = "memory-lme-auto"
	lmeSeedAgentName, lmeQAAgentName = "memory-lme-seed", "memory-lme-agent"
	lmeDefaultAnswerMaxTokens        = 500
	lmeDefaultJudgeMaxTokens         = 10
	lmeDefaultMaxRetries             = 3
	lmeExtractionPollInterval        = 5 * time.Second
	lmeExtractionStableRounds        = 3
	lmeMemoryReadLimit               = 10000
)

var lmeRetrievalCutoffs = []int{1, 3, 5, 10, 30, 50}

type lmeRunConfig struct {
	ModelName              string        `json:"model_name"`
	EmbedModelName         string        `json:"embed_model_name,omitempty"`
	DatasetPath            string        `json:"dataset_path"`
	ManifestPath           string        `json:"manifest_path,omitempty"`
	QuestionTypes          []string      `json:"question_types,omitempty"`
	MaxTasks               int           `json:"max_tasks,omitempty"`
	SessionRecallResults   int           `json:"session_recall_results,omitempty"`
	SessionRecallMinScore  float64       `json:"session_recall_min_score,omitempty"`
	SessionRecallUserOnly  bool          `json:"session_recall_user_only,omitempty"`
	MaxContext             int           `json:"max_context"`
	MaxRetries             int           `json:"max_retries"`
	AnswerMaxTokens        int           `json:"answer_max_tokens"`
	JudgeMaxTokens         int           `json:"judge_max_tokens"`
	AutoExtractionWait     time.Duration `json:"auto_extraction_wait"`
	AutoQAOnly             bool          `json:"auto_qa_only,omitempty"`
	AutoMemoryTable        string        `json:"auto_memory_table,omitempty"`
	EmbeddingCacheEnabled  bool          `json:"embedding_cache_enabled,omitempty"`
	EmbeddingCachePath     string        `json:"embedding_cache_path,omitempty"`
	TransportRetryEnabled  bool          `json:"transport_retry_enabled"`
	TransportRetryStrategy string        `json:"transport_retry_strategy"`
	FullQALog              bool          `json:"full_qa_log,omitempty"`
}

type lmeMetadata struct {
	Framework             string         `json:"framework"`
	Version               string         `json:"version"`
	Timestamp             time.Time      `json:"timestamp"`
	DatasetFormat         string         `json:"dataset_format"`
	Scenario              string         `json:"scenario"`
	MemoryBackend         string         `json:"memory_backend,omitempty"`
	MemoryOnlyCompliant   bool           `json:"memory_only_compliant,omitempty"`
	NativeMemoryPreserved bool           `json:"native_memory_preserved,omitempty"`
	FairlyComparable      bool           `json:"fairly_comparable,omitempty"`
	ComparisonStatus      string         `json:"comparison_status,omitempty"`
	ComparisonBlockers    []string       `json:"comparison_blockers,omitempty"`
	MemoryBuildMethod     string         `json:"memory_build_method,omitempty"`
	MemoryBuild           map[string]any `json:"memory_build,omitempty"`
	MemoryOnlyPolicy      map[string]any `json:"memory_only_policy,omitempty"`
	MemoryOnlySummary     map[string]any `json:"memory_only_summary,omitempty"`
	QAContextPolicy       string         `json:"qa_context_policy,omitempty"`
	Config                lmeRunConfig   `json:"config"`
}

type lmeRunResult struct {
	Metadata  *lmeMetadata              `json:"metadata"`
	Cost      *lmeCostReport            `json:"cost,omitempty"`
	Summary   *lmeSummary               `json:"summary"`
	ByType    map[string]*lmeTypeMetric `json:"by_type"`
	Cases     []*lmeCaseResult          `json:"cases"`
	LastError *lmeRunError              `json:"last_error,omitempty"`
}

type lmeSummary struct {
	TotalCases            int                   `json:"total_cases"`
	CompletedCases        int                   `json:"completed_cases"`
	Overall               metrics.AnswerMetrics `json:"overall"`
	TaskAveragedAccuracy  float64               `json:"task_averaged_accuracy"`
	AbstentionAccuracy    float64               `json:"abstention_accuracy,omitempty"`
	AbstentionCount       int                   `json:"abstention_count,omitempty"`
	NonAbstentionCount    int                   `json:"non_abstention_count,omitempty"`
	TotalTimeMs           int64                 `json:"total_time_ms"`
	AvgLatencyMs          float64               `json:"avg_latency_ms"`
	TotalPromptTokens     int                   `json:"total_prompt_tokens"`
	TotalCompletionTokens int                   `json:"total_completion_tokens"`
	TotalTokens           int                   `json:"total_tokens"`
	TotalCachedTokens     int                   `json:"total_cached_tokens,omitempty"`
	TotalLLMCalls         int                   `json:"total_llm_calls"`
	AvgPromptTokensPerQA  float64               `json:"avg_prompt_tokens_per_qa"`
	AvgCompletionPerQA    float64               `json:"avg_completion_tokens_per_qa"`
	AvgLLMCallsPerQA      float64               `json:"avg_llm_calls_per_qa"`
	Retrieval             *lmeRetrievalSummary  `json:"retrieval,omitempty"`
}

type lmeTypeMetric struct {
	Count   int                   `json:"count"`
	Metrics metrics.AnswerMetrics `json:"metrics"`
}

type lmeRetrievalSummary struct {
	Count             int                         `json:"count"`
	SkippedAbstention int                         `json:"skipped_abstention"`
	SkippedNoTarget   int                         `json:"skipped_no_target"`
	Turn              metrics.RetrievalMetricsAtK `json:"turn,omitempty"`
	SessionFromTurn   metrics.RetrievalMetricsAtK `json:"session_from_turn,omitempty"`
}

type lmeRunError struct {
	QuestionID string `json:"question_id,omitempty"`
	Scenario   string `json:"scenario,omitempty"`
	Message    string `json:"message"`
}

type lmeCaseResult struct {
	QuestionID    string                `json:"question_id"`
	QuestionType  string                `json:"question_type"`
	Question      string                `json:"question"`
	QuestionDate  string                `json:"question_date"`
	Expected      string                `json:"expected"`
	Predicted     string                `json:"predicted"`
	IsAbstention  bool                  `json:"is_abstention"`
	Correct       bool                  `json:"correct"`
	Metrics       metrics.AnswerMetrics `json:"metrics"`
	LatencyMs     int64                 `json:"latency_ms"`
	TokenUsage    *scenarios.TokenUsage `json:"token_usage,omitempty"`
	RetryCount    int                   `json:"retry_count,omitempty"`
	TotalTurns    int                   `json:"total_turns"`
	TotalSessions int                   `json:"total_sessions"`
	Retrieval     *lmeRetrievalTrace    `json:"retrieval,omitempty"`
	ToolSteps     []lmeStepTrace        `json:"tool_steps,omitempty"`
	QATrace       []lmeMessageTrace     `json:"qa_trace,omitempty"`
}

type lmeRetrievalTrace struct {
	Query          string                      `json:"query"`
	MaxResults     int                         `json:"max_results"`
	MinScore       float64                     `json:"min_score,omitempty"`
	SearchMode     session.SearchMode          `json:"search_mode"`
	Hits           []lmeRetrievalHit           `json:"hits"`
	CorrectTurns   []string                    `json:"correct_turns,omitempty"`
	TurnMetrics    metrics.RetrievalMetricsAtK `json:"turn_metrics,omitempty"`
	SessionMetrics metrics.RetrievalMetricsAtK `json:"session_metrics,omitempty"`
}

type lmeRetrievalHit struct {
	SessionID   string     `json:"session_id"`
	TurnID      string     `json:"turn_id"`
	EventID     string     `json:"event_id"`
	Role        model.Role `json:"role"`
	Text        string     `json:"text"`
	Score       float64    `json:"score,omitempty"`
	DenseScore  float64    `json:"dense_score,omitempty"`
	SparseScore float64    `json:"sparse_score,omitempty"`
}

type lmeStepTrace struct {
	Step             int                `json:"step"`
	PromptTokens     int                `json:"prompt_tokens"`
	CompletionTokens int                `json:"completion_tokens"`
	TotalTokens      int                `json:"total_tokens"`
	CachedTokens     int                `json:"cached_tokens,omitempty"`
	ToolCalls        []lmeToolCallTrace `json:"tool_calls,omitempty"`
}

type lmeToolCallTrace struct {
	Name   string `json:"name"`
	Args   string `json:"args,omitempty"`
	Result string `json:"result,omitempty"`
}

type lmeMessageTrace struct {
	Step      int                `json:"step,omitempty"`
	Role      string             `json:"role"`
	Name      string             `json:"name,omitempty"`
	Content   string             `json:"content,omitempty"`
	ToolCalls []lmeToolCallTrace `json:"tool_calls,omitempty"`
}

type lmeModelResult struct {
	Text       string
	Usage      *model.Usage
	RetryCount int
}

type lmeCollectResult struct {
	Text       string
	Usage      scenarios.TokenUsage
	Steps      []lmeStepTrace
	Trace      []lmeMessageTrace
	RetryCount int
}

type lmeLongContextEvaluator struct {
	llm model.Model
	cfg lmeRunConfig
}

type lmeSessionRecallEvaluator struct {
	llm model.Model
	svc session.Service
	cfg lmeRunConfig
}

type lmeAutoEvaluator struct {
	name       string
	judgeLLM   model.Model
	qaLLM      model.Model
	extractor  extractor.MemoryExtractor
	mem        memory.Service
	cfg        lmeRunConfig
	cost       *lmeCostTracker
	deepSearch bool
}

type lmeNoAutoMemoryService struct {
	inner memory.Service
}

type lmeSeedAgent struct{}

func runLongMemEvalMemory(ctx context.Context) error {
	cfg, llm, err := buildLMERunConfigAndModel()
	if err != nil {
		return err
	}
	defer closeLMEEmbeddingCaches()
	instances, err := loadLMEInstances(cfg)
	if err != nil {
		return err
	}
	scenarioTypes := getLMEScenarios(*flagScenario)
	if err := validateLMEPrerequisites(scenarioTypes); err != nil {
		return err
	}
	rootDir := filepath.Join(*flagOutput, "longmemeval")
	if err := os.MkdirAll(rootDir, 0755); err != nil {
		return fmt.Errorf("create LongMemEval output dir: %w", err)
	}
	for _, scenarioType := range scenarioTypes {
		evaluator, backend, err := newLMEEvaluator(scenarioType, llm, cfg)
		if err != nil {
			return err
		}
		scenarioDir := lmeScenarioDir(rootDir, scenarioType, backend)
		if err := runLMEEvaluator(ctx, evaluator, cfg, instances, backend, scenarioDir); err != nil {
			_ = evaluator.Close()
			return err
		}
		if err := evaluator.Close(); err != nil {
			return fmt.Errorf("close %s evaluator: %w", evaluator.Name(), err)
		}
	}
	if err := writeLMEReports(rootDir, cfg, scenarioTypes); err != nil {
		return err
	}
	writeLMEEmbeddingCacheStats(rootDir)
	return nil
}

func validateLMEPrerequisites(scenarioTypes []scenarios.ScenarioType) error {
	for _, scenarioType := range scenarioTypes {
		if scenarioType != scenarios.ScenarioSessionRecall &&
			scenarioType != scenarios.ScenarioAuto &&
			scenarioType != scenarios.ScenarioAutoDeepSearch {
			continue
		}
		if getPGVectorDSN() == "" {
			return fmt.Errorf(
				"PGVECTOR_DSN or -pgvector-dsn is required before running LongMemEval %s",
				scenarioType,
			)
		}
	}
	return nil
}

func buildLMERunConfigAndModel() (lmeRunConfig, model.Model, error) {
	modelName := strings.TrimSpace(*flagModel)
	if modelName == "" {
		modelName = strings.TrimSpace(os.Getenv("LLM_NAME"))
	}
	if modelName == "" {
		return lmeRunConfig{}, nil, fmt.Errorf("LLM_NAME or -model is required for LongMemEval")
	}
	apiKey := strings.TrimSpace(os.Getenv("LLM_API_KEY"))
	if apiKey == "" {
		return lmeRunConfig{}, nil, fmt.Errorf("LLM_API_KEY is required for LongMemEval")
	}
	baseURL := strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	opts := []openaimodel.Option{openaimodel.WithAPIKey(apiKey)}
	if baseURL != "" {
		opts = append(opts, openaimodel.WithBaseURL(baseURL))
	}
	embedModelName := getLMEEmbedModelName()
	ensureLMEEmbeddingEnv(embedModelName)
	cfg := lmeRunConfig{
		ModelName:              modelName,
		EmbedModelName:         embedModelName,
		DatasetPath:            lmeDatasetPath(),
		ManifestPath:           strings.TrimSpace(*flagLMEManifest),
		QuestionTypes:          parseCSV(*flagLMEQuestionTypes),
		MaxTasks:               *flagMaxTasks,
		SessionRecallResults:   *flagVectorTopK,
		SessionRecallMinScore:  *flagSessionRecallMinScore,
		SessionRecallUserOnly:  *flagLMESessionRecallUserOnly,
		MaxContext:             *flagMaxContext,
		MaxRetries:             max(*flagLMEMaxRetries, 0),
		AnswerMaxTokens:        max(*flagLMEAnswerMaxTokens, 1),
		JudgeMaxTokens:         max(*flagLMEJudgeMaxTokens, 1),
		AutoExtractionWait:     *flagLMEExtractionWait,
		AutoQAOnly:             *flagLMEAutoQAOnly,
		EmbeddingCacheEnabled:  *flagLMEEmbeddingCache,
		TransportRetryEnabled:  *flagLMEMaxRetries > 0,
		TransportRetryStrategy: "fixed prompt, same model, retry transport/rate-limit errors only",
		FullQALog:              true,
	}
	if cfg.AutoQAOnly {
		cfg.AutoMemoryTable = tableNameWithSuffix(pgvectorTableDefaultBase)
	}
	if cfg.EmbeddingCacheEnabled {
		cfg.EmbeddingCachePath = lmeEmbeddingCachePath(embedModelName)
	}
	return cfg, openaimodel.New(modelName, opts...), nil
}

func getLMEEmbedModelName() string {
	if *flagEmbedModel != "" {
		return *flagEmbedModel
	}
	if env := os.Getenv("EMBED_MODEL_NAME"); env != "" {
		return env
	}
	return "text-embedding-3-small"
}

func ensureLMEEmbeddingEnv(embedModelName string) {
	if os.Getenv("EMBED_MODEL_NAME") == "" && embedModelName != "" {
		_ = os.Setenv("EMBED_MODEL_NAME", embedModelName)
	}
	if os.Getenv("OPENAI_EMBEDDING_API_KEY") != "" {
		return
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		_ = os.Setenv("OPENAI_EMBEDDING_API_KEY", os.Getenv("OPENAI_API_KEY"))
		return
	}
	if os.Getenv("LLM_API_KEY") != "" {
		_ = os.Setenv("OPENAI_EMBEDDING_API_KEY", os.Getenv("LLM_API_KEY"))
	}
}

func lmeDatasetPath() string {
	if strings.HasSuffix(*flagDataset, ".json") || strings.HasSuffix(*flagDataset, ".jsonl") {
		return *flagDataset
	}
	dataFile := *flagDataFile
	if dataFile == "locomo10.json" {
		dataFile = "longmemeval_s_cleaned.json"
	}
	return filepath.Join(*flagDataset, dataFile)
}

func loadLMEInstances(cfg lmeRunConfig) ([]*dataset.LongMemEvalInstance, error) {
	instances, err := dataset.LoadLongMemEval(cfg.DatasetPath)
	if err != nil {
		return nil, err
	}
	instances = dataset.FilterLongMemEval(instances, cfg.QuestionTypes)
	if cfg.ManifestPath != "" {
		manifest, err := dataset.LoadLongMemEvalManifest(cfg.ManifestPath)
		if err != nil {
			return nil, err
		}
		instances, err = dataset.FilterLongMemEvalByManifest(instances, manifest)
		if err != nil {
			return nil, err
		}
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("no LongMemEval cases remain after filtering")
	}
	if cfg.MaxTasks > 0 && cfg.MaxTasks < len(instances) {
		instances = instances[:cfg.MaxTasks]
	}
	log.Printf("Loaded %d LongMemEval cases from %s", len(instances), cfg.DatasetPath)
	log.Printf("Question types: %s", strings.Join(dataset.LongMemEvalQuestionTypes(instances), ", "))
	return instances, nil
}

func getLMEScenarios(raw string) []scenarios.ScenarioType {
	if raw == "all" {
		// Reports reuse an existing long_context result as the reference
		// baseline when present, so all must not rerun long_context.
		return []scenarios.ScenarioType{scenarios.ScenarioAuto}
	}
	allowed := map[string]scenarios.ScenarioType{
		"long_context":    scenarios.ScenarioLongContext,
		"auto":            scenarios.ScenarioAuto,
		"auto_deepsearch": scenarios.ScenarioAutoDeepSearch,
	}
	seen := make(map[string]struct{})
	var out []scenarios.ScenarioType
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		scenarioType, ok := allowed[part]
		if !ok {
			log.Fatalf(
				"LongMemEval supports long_context, auto, auto_deepsearch, all; got %s",
				part,
			)
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		out = append(out, scenarioType)
	}
	if len(out) == 0 {
		log.Fatalf("No LongMemEval scenarios selected")
	}
	return out
}

func newLMEEvaluator(
	scenarioType scenarios.ScenarioType,
	llm model.Model,
	cfg lmeRunConfig,
) (lmeEvaluator, string, error) {
	switch scenarioType {
	case scenarios.ScenarioLongContext:
		return &lmeLongContextEvaluator{llm: llm, cfg: cfg}, "", nil
	case scenarios.ScenarioSessionRecall:
		svc, err := createSessionRecallService(scenarios.Config{
			SessionEventLimit:     *flagSessionEventLimit,
			SessionRecallResults:  cfg.SessionRecallResults,
			SessionRecallMinScore: cfg.SessionRecallMinScore,
		})
		if err != nil {
			return nil, "", err
		}
		return &lmeSessionRecallEvaluator{llm: llm, svc: svc, cfg: cfg}, "session_pgvector", nil
	case scenarios.ScenarioAuto, scenarios.ScenarioAutoDeepSearch:
		cost := newLMECostTracker()
		memoryBuildLLM := newLMETrackedModel(
			llm,
			cost,
			lmeLLMPhaseMemoryBuild,
		)
		memCfg := memoryConfig{backend: "pgvector", mode: memoryModeNone}
		memOpts := memoryServiceOptions{
			vectorTopK: cfg.SessionRecallResults,
		}
		deepSearch := scenarioType == scenarios.ScenarioAutoDeepSearch
		if deepSearch {
			cfg.AutoQAOnly = true
			cfg.AutoMemoryTable = tableNameWithSuffix(pgvectorTableDefaultBase)
			memOpts.deepSearchModel = memoryBuildLLM
		}
		memSvc, err := createMemoryService(
			memCfg,
			memOpts,
		)
		if err != nil {
			return nil, "", err
		}
		name := "auto"
		if deepSearch {
			name = "auto_deepsearch"
		}
		return &lmeAutoEvaluator{
			name:     name,
			judgeLLM: newLMETrackedModel(llm, cost, lmeLLMPhaseJudge),
			qaLLM:    newLMETrackedModel(llm, cost, lmeLLMPhaseQA),
			extractor: extractor.NewExtractor(
				memoryBuildLLM,
				extractor.WithModelCallbacks(lmeExtractorModelCallbacks()),
			),
			mem:        memSvc,
			cfg:        cfg,
			cost:       cost,
			deepSearch: deepSearch,
		}, "pgvector", nil
	default:
		return nil, "", fmt.Errorf("unsupported LongMemEval scenario %s", scenarioType)
	}
}

func lmeExtractorModelCallbacks() *model.Callbacks {
	temp := 0.0
	return model.NewCallbacks().RegisterBeforeModel(
		func(_ context.Context, args *model.BeforeModelArgs) (*model.BeforeModelResult, error) {
			if args == nil || args.Request == nil {
				return nil, nil
			}
			args.Request.GenerationConfig.Stream = false
			if args.Request.GenerationConfig.Temperature == nil {
				args.Request.GenerationConfig.Temperature = &temp
			}
			return nil, nil
		},
	)
}

func runLMEEvaluator(
	ctx context.Context,
	evaluator lmeEvaluator,
	cfg lmeRunConfig,
	instances []*dataset.LongMemEvalInstance,
	backend string,
	outputDir string,
) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create scenario output dir: %w", err)
	}
	result := newLMERunResult(evaluator.Name(), backend, cfg, len(instances))
	completed := make(map[string]struct{})
	startTime := time.Now()
	var checkpointCost *lmeCostReport
	if *flagResume {
		if checkpoint, err := loadLMERunResult(outputDir); err == nil && checkpoint != nil {
			result = checkpoint
			if checkpoint.Cost != nil {
				checkpointCost = checkpoint.Cost
			} else if len(checkpoint.Cases) > 0 {
				checkpointCost = partialLMECostReport(fmt.Sprintf(
					"resumed checkpoint has %d completed case(s) without cost metadata",
					len(checkpoint.Cases),
				))
			}
			for _, cr := range result.Cases {
				completed[cr.QuestionID] = struct{}{}
			}
			log.Printf("Resumed %s with %d completed cases", evaluator.Name(), len(completed))
		}
	}
	for i, inst := range instances {
		if err := ctx.Err(); err != nil {
			result.LastError = &lmeRunError{
				Scenario: evaluator.Name(),
				Message:  err.Error(),
			}
			result.Summary.TotalTimeMs = time.Since(startTime).Milliseconds()
			setLMERunMetadata(result, evaluator)
			setLMERunCost(result, evaluator, checkpointCost)
			saveLMERunResult(outputDir, result)
			return fmt.Errorf("%s stopped: %w", evaluator.Name(), err)
		}
		if _, ok := completed[inst.QuestionID]; ok {
			log.Printf("[%d/%d] %s skipped from checkpoint", i+1, len(instances), inst.QuestionID)
			continue
		}
		log.Printf("[%d/%d] LongMemEval %s %s (%s)",
			i+1, len(instances), evaluator.Name(), inst.QuestionID, inst.QuestionType)
		caseResult, err := evaluator.Evaluate(ctx, inst)
		if err != nil {
			result.LastError = &lmeRunError{
				QuestionID: inst.QuestionID,
				Scenario:   evaluator.Name(),
				Message:    err.Error(),
			}
			result.Summary.TotalTimeMs = time.Since(startTime).Milliseconds()
			setLMERunMetadata(result, evaluator)
			setLMERunCost(result, evaluator, checkpointCost)
			saveLMERunResult(outputDir, result)
			return fmt.Errorf("%s %s: %w", evaluator.Name(), inst.QuestionID, err)
		}
		result.Cases = append(result.Cases, caseResult)
		result.LastError = nil
		aggregateLMERunResult(result, time.Since(startTime), len(instances))
		setLMERunMetadata(result, evaluator)
		setLMERunCost(result, evaluator, checkpointCost)
		saveLMERunResult(outputDir, result)
		saveLMECaseLog(outputDir, caseResult)
	}
	aggregateLMERunResult(result, time.Since(startTime), len(instances))
	setLMERunMetadata(result, evaluator)
	setLMERunCost(result, evaluator, checkpointCost)
	saveLMERunResult(outputDir, result)
	printLMESummary(result)
	return nil
}

func newLMERunResult(
	scenarioName, backend string,
	cfg lmeRunConfig,
	totalCases int,
) *lmeRunResult {
	return &lmeRunResult{
		Metadata: &lmeMetadata{
			Framework:     "trpc-agent-go",
			Version:       "1.0.0",
			Timestamp:     time.Now(),
			DatasetFormat: lmeDatasetFormat,
			Scenario:      scenarioName,
			MemoryBackend: backend,
			Config:        cfg,
		},
		Summary: &lmeSummary{TotalCases: totalCases},
		ByType:  make(map[string]*lmeTypeMetric),
		Cases:   make([]*lmeCaseResult, 0, totalCases),
	}
}

func setLMERunCost(
	result *lmeRunResult,
	evaluator lmeEvaluator,
	base *lmeCostReport,
) {
	reporter, ok := evaluator.(interface {
		CostReport() *lmeCostReport
	})
	if !ok {
		result.Cost = base
		return
	}
	result.Cost = mergeLMECostReports(base, reporter.CostReport())
}

func aggregateLMERunResult(result *lmeRunResult, elapsed time.Duration, totalCases int) {
	result.ByType = make(map[string]*lmeTypeMetric)
	var overall metrics.AnswerMetrics
	var totalUsage scenarios.TokenUsage
	var totalLatency int64
	var abstentionTotal, abstentionCorrect int
	var nonAbstentionTotal int
	typeAcc := make(map[string][]float64)
	retrievalAgg := newLMERetrievalAccumulator()
	for _, cr := range result.Cases {
		overall.Add(cr.Metrics)
		totalLatency += cr.LatencyMs
		if cr.TokenUsage != nil {
			totalUsage.Add(*cr.TokenUsage)
		}
		tm := result.ByType[cr.QuestionType]
		if tm == nil {
			tm = &lmeTypeMetric{}
			result.ByType[cr.QuestionType] = tm
		}
		tm.Count++
		tm.Metrics.Add(cr.Metrics)
		typeAcc[cr.QuestionType] = append(typeAcc[cr.QuestionType], cr.Metrics.Accuracy)
		if cr.IsAbstention {
			abstentionTotal++
			if cr.Correct {
				abstentionCorrect++
			}
		} else {
			nonAbstentionTotal++
		}
		retrievalAgg.add(cr)
	}
	n := float64(len(result.Cases))
	overall.Divide(n)
	for _, tm := range result.ByType {
		tm.Metrics.Divide(float64(tm.Count))
	}
	taskAvg := 0.0
	if len(typeAcc) > 0 {
		for _, values := range typeAcc {
			taskAvg += averageFloat64(values)
		}
		taskAvg /= float64(len(typeAcc))
	}
	qCount := max(len(result.Cases), 1)
	summary := &lmeSummary{
		TotalCases:            totalCases,
		CompletedCases:        len(result.Cases),
		Overall:               overall,
		TaskAveragedAccuracy:  taskAvg,
		AbstentionCount:       abstentionTotal,
		NonAbstentionCount:    nonAbstentionTotal,
		TotalTimeMs:           elapsed.Milliseconds(),
		AvgLatencyMs:          float64(totalLatency) / float64(qCount),
		TotalPromptTokens:     totalUsage.PromptTokens,
		TotalCompletionTokens: totalUsage.CompletionTokens,
		TotalTokens:           totalUsage.TotalTokens,
		TotalCachedTokens:     totalUsage.CachedTokens,
		TotalLLMCalls:         totalUsage.LLMCalls,
		AvgPromptTokensPerQA:  float64(totalUsage.PromptTokens) / float64(qCount),
		AvgCompletionPerQA:    float64(totalUsage.CompletionTokens) / float64(qCount),
		AvgLLMCallsPerQA:      float64(totalUsage.LLMCalls) / float64(qCount),
		Retrieval:             retrievalAgg.summary(),
	}
	if abstentionTotal > 0 {
		summary.AbstentionAccuracy = float64(abstentionCorrect) / float64(abstentionTotal)
	}
	result.Summary = summary
}

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
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/scenarios"
	"trpc.group/trpc-go/trpc-agent-go/knowledge/embedder"
	"trpc.group/trpc-go/trpc-agent-go/knowledge/embedder/openai"
	"trpc.group/trpc-go/trpc-agent-go/memory"
	"trpc.group/trpc-go/trpc-agent-go/memory/deepsearch"
	"trpc.group/trpc-go/trpc-agent-go/memory/extractor"
	"trpc.group/trpc-go/trpc-agent-go/memory/inmemory"
	memorymysql "trpc.group/trpc-go/trpc-agent-go/memory/mysql"
	memorypgvector "trpc.group/trpc-go/trpc-agent-go/memory/pgvector"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/session"
	sessionpgvector "trpc.group/trpc-go/trpc-agent-go/session/pgvector"
)

func getEmbedModelName() string {
	if *flagEmbedModel != "" {
		return *flagEmbedModel
	}
	if env := os.Getenv("EMBED_MODEL_NAME"); env != "" {
		return env
	}
	return "text-embedding-3-small"
}

const (
	envOpenAIBaseURL          = "OPENAI_BASE_URL"
	envOpenAIEmbeddingAPIKey  = "OPENAI_EMBEDDING_API_KEY"
	envOpenAIEmbeddingBaseURL = "OPENAI_EMBEDDING_BASE_URL"
)

func newEmbeddingEmbedder(modelName string) (embedder.Embedder, error) {
	opts := []openai.Option{
		openai.WithModel(modelName),
	}

	if apiKey := os.Getenv(envOpenAIEmbeddingAPIKey); apiKey != "" {
		opts = append(opts, openai.WithAPIKey(apiKey))
	}

	baseURL := os.Getenv(envOpenAIEmbeddingBaseURL)
	if baseURL == "" {
		baseURL = os.Getenv(envOpenAIBaseURL)
	}
	if baseURL != "" {
		opts = append(opts, openai.WithBaseURL(baseURL))
	}

	inner := openai.New(opts...)
	if !shouldUseLMEEmbeddingCache() {
		return newLMETrackedEmbedder(inner), nil
	}
	cache, err := newLMEEmbeddingCache(
		inner,
		modelName,
		lmeEmbeddingCachePath(modelName),
	)
	if err != nil {
		return nil, err
	}
	return cache, nil
}

func shouldUseLMEEmbeddingCache() bool {
	return *flagDatasetFormat == lmeDatasetFormat && *flagLMEEmbeddingCache
}

func lmeEmbeddingCachePath(modelName string) string {
	cacheDir := strings.TrimSpace(*flagLMEEmbeddingCacheDir)
	if cacheDir == "" {
		cacheDir = filepath.Join(*flagOutput, "longmemeval", ".cache")
	}
	return filepath.Join(cacheDir, fmt.Sprintf(
		"embeddings_%s",
		sanitizeCacheFileName(modelName),
	))
}

func sanitizeCacheFileName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "default"
	}
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	return b.String()
}

func getPGVectorDSN() string {
	if *flagPGVectorDSN != "" {
		return *flagPGVectorDSN
	}
	return os.Getenv("PGVECTOR_DSN")
}

func getMySQLDSN() string {
	if *flagMySQLDSN != "" {
		return *flagMySQLDSN
	}
	return os.Getenv("MYSQL_DSN")
}

func buildMemoryConfig(
	scenarioType scenarios.ScenarioType,
	backend string,
) memoryConfig {
	switch scenarioType {
	case scenarios.ScenarioAuto:
		return memoryConfig{
			backend: backend,
			mode:    memoryModeAuto,
		}
	case scenarios.ScenarioAgentic:
		return memoryConfig{
			backend: backend,
			mode:    memoryModeManual,
		}
	default:
		return memoryConfig{
			mode: memoryModeNone,
		}
	}
}

func buildMemoryServiceOptions(
	cfg memoryConfig,
	extractorModel model.Model,
) memoryServiceOptions {
	opts := memoryServiceOptions{vectorTopK: *flagVectorTopK}
	if cfg.mode != memoryModeAuto {
		return opts
	}
	opts.enableExtractor = true
	opts.extractorModel = extractorModel
	return opts
}

type memoryServiceOptions struct {
	enableExtractor bool
	extractorModel  model.Model
	deepSearchModel model.Model
	vectorTopK      int
}

func createMemoryService(
	cfg memoryConfig,
	opts memoryServiceOptions,
) (memory.Service, error) {
	switch cfg.backend {
	case "pgvector":
		return createPGVectorService(opts)
	case "mysql":
		return createMySQLService(opts)
	case "sqlite":
		return createSQLiteService(opts)
	case "sqlitevec":
		return createSQLiteVecService(opts)
	default:
		return createInMemoryService(opts), nil
	}
}

func createSessionRecallService(
	cfg scenarios.Config,
) (session.Service, error) {
	dsn := getPGVectorDSN()
	if dsn == "" {
		return nil, fmt.Errorf(
			"pgvector-dsn or PGVECTOR_DSN is required for session_recall scenario",
		)
	}
	embedModelName := getEmbedModelName()
	emb, err := newEmbeddingEmbedder(embedModelName)
	if err != nil {
		return nil, err
	}
	log.Printf(
		"Creating session recall pgvector service (embed_model=%s)",
		embedModelName,
	)
	return sessionpgvector.NewService(
		sessionpgvector.WithPostgresClientDSN(dsn),
		sessionpgvector.WithEmbedder(emb),
		sessionpgvector.WithIndexDimension(emb.GetDimensions()),
		sessionpgvector.WithSessionEventLimit(cfg.SessionEventLimit),
		sessionpgvector.WithMaxResults(cfg.SessionRecallResults),
		sessionpgvector.WithTablePrefix(
			tableNameWithSuffix(sessionRecallTableBase),
		),
		sessionpgvector.WithSoftDelete(false),
		sessionpgvector.WithSyncIndexing(true),
	)
}

func createPGVectorService(
	opts memoryServiceOptions,
) (memory.Service, error) {
	dsn := getPGVectorDSN()
	if dsn == "" {
		return nil, fmt.Errorf(
			"pgvector-dsn or PGVECTOR_DSN is required for pgvector backend",
		)
	}
	embedModelName := getEmbedModelName()
	emb, err := newEmbeddingEmbedder(embedModelName)
	if err != nil {
		return nil, err
	}
	tableName := tableNameWithSuffix(pgvectorTableDefaultBase)
	var ext extractor.MemoryExtractor
	if opts.enableExtractor {
		log.Printf(
			"Creating pgvector memory service with extractor "+
				"(embed_model=%s)",
			embedModelName,
		)
		tableName = tableNameWithSuffix(pgvectorTableAutoBase)
		ext = extractor.NewExtractor(opts.extractorModel)
	} else {
		log.Printf(
			"Creating pgvector memory service (embed_model=%s)",
			embedModelName,
		)
	}
	svcOpts := []memorypgvector.ServiceOpt{
		memorypgvector.WithPGVectorClientDSN(dsn),
		memorypgvector.WithEmbedder(emb),
		memorypgvector.WithMaxResults(opts.vectorTopK),
		memorypgvector.WithTableName(tableName),
		memorypgvector.WithExtractor(ext),
	}
	if opts.deepSearchModel != nil {
		svcOpts = append(
			svcOpts,
			memorypgvector.WithDeepSearch(
				opts.deepSearchModel,
				deepsearch.WithBatchSize(lmeDeepSearchBatchSize),
			),
		)
	}
	if opts.enableExtractor {
		svcOpts = append(svcOpts,
			memorypgvector.WithAsyncMemoryNum(autoMemoryAsyncWorkers),
			memorypgvector.WithMemoryQueueSize(autoMemoryQueueSize),
			memorypgvector.WithMemoryJobTimeout(autoMemoryJobTimeout),
		)
	}
	return memorypgvector.NewService(svcOpts...)
}

func createMySQLService(
	opts memoryServiceOptions,
) (memory.Service, error) {
	dsn := getMySQLDSN()
	if dsn == "" {
		return nil, fmt.Errorf(
			"mysql-dsn or MYSQL_DSN is required for mysql backend",
		)
	}

	tableName := tableNameWithSuffix(mysqlTableDefaultBase)
	var ext extractor.MemoryExtractor
	if opts.enableExtractor {
		log.Printf("Creating mysql memory service with extractor")
		tableName = tableNameWithSuffix(mysqlTableAutoBase)
		ext = extractor.NewExtractor(opts.extractorModel)
	} else {
		log.Printf("Creating mysql memory service")
	}

	svcOpts := []memorymysql.ServiceOpt{
		memorymysql.WithMySQLClientDSN(dsn),
		memorymysql.WithTableName(tableName),
		memorymysql.WithExtractor(ext),
	}
	if opts.enableExtractor {
		svcOpts = append(svcOpts,
			memorymysql.WithAsyncMemoryNum(autoMemoryAsyncWorkers),
			memorymysql.WithMemoryQueueSize(autoMemoryQueueSize),
			memorymysql.WithMemoryJobTimeout(autoMemoryJobTimeout),
		)
	}
	return memorymysql.NewService(svcOpts...)
}

func createInMemoryService(opts memoryServiceOptions) memory.Service {
	if opts.enableExtractor {
		log.Printf("Creating inmemory memory service with extractor")
		ext := extractor.NewExtractor(opts.extractorModel)
		return inmemory.NewMemoryService(
			inmemory.WithExtractor(ext),
			inmemory.WithAsyncMemoryNum(autoMemoryAsyncWorkers),
			inmemory.WithMemoryQueueSize(autoMemoryQueueSize),
			inmemory.WithMemoryJobTimeout(autoMemoryJobTimeout),
		)
	}
	return inmemory.NewMemoryService()
}

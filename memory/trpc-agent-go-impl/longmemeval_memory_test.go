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
	"testing"

	"trpc.group/trpc-go/trpc-agent-go/memory"
	"trpc.group/trpc-go/trpc-agent-go/memory/inmemory"
)

func TestRequireLMEAutoMemories(t *testing.T) {
	ctx := context.Background()
	service := inmemory.NewMemoryService()
	defer service.Close()
	userKey := memory.UserKey{
		AppName: lmeAppAuto,
		UserID:  "case-1",
	}

	if err := requireLMEAutoMemories(ctx, service, userKey); err == nil {
		t.Fatal("requireLMEAutoMemories() error = nil, want missing memory error")
	}
	if err := service.AddMemory(ctx, userKey, "User likes yoga.", []string{"yoga"}); err != nil {
		t.Fatalf("AddMemory() error = %v", err)
	}
	if err := requireLMEAutoMemories(ctx, service, userKey); err != nil {
		t.Fatalf("requireLMEAutoMemories() error = %v", err)
	}
}

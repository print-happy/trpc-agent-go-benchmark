//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//
//

package dataset

import (
	"encoding/json"
	"fmt"
	"os"
)

// LongMemEvalManifest stores a fixed LongMemEval case subset.
type LongMemEvalManifest struct {
	CaseIDs []string `json:"case_ids"`
}

// LoadLongMemEvalManifest loads a fixed case-id manifest.
func LoadLongMemEvalManifest(path string) (*LongMemEvalManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read LongMemEval manifest %s: %w", path, err)
	}
	var manifest LongMemEvalManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("parse LongMemEval manifest %s: %w", path, err)
	}
	if len(manifest.CaseIDs) == 0 {
		return nil, fmt.Errorf("LongMemEval manifest %s has no case_ids", path)
	}
	return &manifest, nil
}

// FilterLongMemEvalByManifest filters instances by fixed case IDs and preserves
// manifest order.
func FilterLongMemEvalByManifest(
	instances []*LongMemEvalInstance,
	manifest *LongMemEvalManifest,
) ([]*LongMemEvalInstance, error) {
	if manifest == nil {
		return instances, nil
	}
	byID := make(map[string]*LongMemEvalInstance, len(instances))
	for _, inst := range instances {
		if inst == nil {
			continue
		}
		byID[inst.QuestionID] = inst
	}
	filtered := make([]*LongMemEvalInstance, 0, len(manifest.CaseIDs))
	for _, id := range manifest.CaseIDs {
		inst := byID[id]
		if inst == nil {
			return nil, fmt.Errorf("LongMemEval manifest case_id %s not found in dataset", id)
		}
		filtered = append(filtered, inst)
	}
	return filtered, nil
}

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
	"reflect"
	"testing"
)

func TestLongMemEvalEvidenceMapping(t *testing.T) {
	inst := &LongMemEvalInstance{
		QuestionID:         "case-1",
		QuestionType:       "single-session-user",
		HaystackSessionIDs: []string{"answer_s1"},
		HaystackSessions: [][]LongMemEvalTurn{
			{
				{Role: "user", Content: "where did we go?", HasAnswer: true},
				{Role: "assistant", Content: "Kyoto"},
				{Role: "user", Content: "thanks"},
			},
		},
	}

	if got := inst.EvidenceTurnIDs(); !reflect.DeepEqual(got, []string{"answer_s1_1"}) {
		t.Fatalf("EvidenceTurnIDs() = %v", got)
	}
	if got := LongMemEvalCorpusTurnIDs(inst); !reflect.DeepEqual(got, []string{"answer_s1_1", "noans_s1_3"}) {
		t.Fatalf("LongMemEvalCorpusTurnIDs() = %v", got)
	}
	if err := ValidateLongMemEvalEvidenceMapping(inst); err != nil {
		t.Fatalf("ValidateLongMemEvalEvidenceMapping() error = %v", err)
	}
}

func TestLongMemEvalTurnIDFromEventExtensions(t *testing.T) {
	raw, err := json.Marshal("answer_s1_1")
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	got := LongMemEvalTurnIDFromEventExtensions(map[string]json.RawMessage{
		EventExtensionLongMemEvalTurnID: raw,
	})

	if got != "answer_s1_1" {
		t.Fatalf("LongMemEvalTurnIDFromEventExtensions() = %q", got)
	}
	if got := LongMemEvalTurnIDFromEventExtensions(nil); got != "" {
		t.Fatalf("LongMemEvalTurnIDFromEventExtensions(nil) = %q", got)
	}
}

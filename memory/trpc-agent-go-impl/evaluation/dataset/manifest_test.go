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
	"strings"
	"testing"
)

func TestFilterLongMemEvalByManifest(t *testing.T) {
	instances := []*LongMemEvalInstance{
		{QuestionID: "a"},
		{QuestionID: "b"},
		{QuestionID: "c"},
	}

	filtered, err := FilterLongMemEvalByManifest(instances, &LongMemEvalManifest{
		CaseIDs: []string{"c", "a"},
	})

	if err != nil {
		t.Fatalf("FilterLongMemEvalByManifest() error = %v", err)
	}
	if len(filtered) != 2 {
		t.Fatalf("len(filtered) = %d", len(filtered))
	}
	if filtered[0].QuestionID != "c" || filtered[1].QuestionID != "a" {
		t.Fatalf("filtered order = %s, %s", filtered[0].QuestionID, filtered[1].QuestionID)
	}
}

func TestFilterLongMemEvalByManifest_MissingID(t *testing.T) {
	_, err := FilterLongMemEvalByManifest(
		[]*LongMemEvalInstance{{QuestionID: "a"}},
		&LongMemEvalManifest{CaseIDs: []string{"missing"}},
	)

	if err == nil {
		t.Fatal("FilterLongMemEvalByManifest() expected error")
	}
	if !strings.Contains(err.Error(), "missing") {
		t.Fatalf("error = %v", err)
	}
}

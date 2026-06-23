//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//
//

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/dataset"
)

var (
	flagDataset = flag.String("dataset", "", "LongMemEval dataset JSON path")
	flagOutput  = flag.String("output", "", "Output manifest JSON path")
	flagTypes   = flag.String(
		"types",
		"multi-session,temporal-reasoning,knowledge-update,single-session-user,single-session-assistant,single-session-preference",
		"Question types in selection priority order",
	)
	flagPerType = flag.Int("per-type", 2, "Maximum cases per question type")
)

func main() {
	flag.Parse()
	if *flagDataset == "" || *flagOutput == "" {
		fmt.Fprintln(os.Stderr, "-dataset and -output are required")
		os.Exit(1)
	}
	instances, err := dataset.LoadLongMemEval(*flagDataset)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load dataset: %v\n", err)
		os.Exit(1)
	}
	manifest := dataset.LongMemEvalManifest{
		CaseIDs: selectCaseIDs(instances, parseTypes(*flagTypes), *flagPerType),
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal manifest: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*flagOutput, append(data, '\n'), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write manifest: %v\n", err)
		os.Exit(1)
	}
}

func parseTypes(raw string) []string {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func selectCaseIDs(
	instances []*dataset.LongMemEvalInstance,
	types []string,
	perType int,
) []string {
	if perType <= 0 {
		perType = 1
	}
	var ids []string
	for _, typ := range types {
		count := 0
		for _, inst := range instances {
			if inst == nil || inst.QuestionType != typ {
				continue
			}
			ids = append(ids, inst.QuestionID)
			count++
			if count >= perType {
				break
			}
		}
	}
	return ids
}

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
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"trpc.group/trpc-go/trpc-agent-go-benchmark/memory/trpc-agent-go-impl/evaluation/scenarios"
	"trpc.group/trpc-go/trpc-agent-go/event"
	"trpc.group/trpc-go/trpc-agent-go/memory/extractor"
	"trpc.group/trpc-go/trpc-agent-go/model"
)

func runLMEModelWithRetry(
	ctx context.Context,
	llm model.Model,
	messages []model.Message,
	maxTokens int,
	maxRetries int,
) (lmeModelResult, error) {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		res, err := runLMEModelOnce(ctx, llm, messages, maxTokens)
		if err == nil {
			res.RetryCount = attempt
			return res, nil
		}
		lastErr = err
		if !isLMETransportError(err) || attempt == maxRetries {
			return lmeModelResult{RetryCount: attempt}, err
		}
		log.Printf("LongMemEval transport retry %d/%d after error: %v", attempt+1, maxRetries, err)
		if sleepErr := lmeRetrySleep(ctx, attempt); sleepErr != nil {
			return lmeModelResult{RetryCount: attempt}, sleepErr
		}
	}
	return lmeModelResult{RetryCount: maxRetries}, lastErr
}

func runLMEExtractWithRetry(
	ctx context.Context,
	maxRetries int,
	extract func() ([]*extractor.Operation, error),
) ([]*extractor.Operation, int, error) {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		ops, err := extract()
		if err == nil {
			return ops, attempt, nil
		}
		lastErr = err
		if !isLMETransportError(err) || attempt == maxRetries {
			return nil, attempt, err
		}
		log.Printf("LongMemEval extraction transport retry %d/%d after error: %v", attempt+1, maxRetries, err)
		if sleepErr := lmeRetrySleep(ctx, attempt); sleepErr != nil {
			return nil, attempt, sleepErr
		}
	}
	return nil, maxRetries, lastErr
}

func runLMEModelOnce(
	ctx context.Context,
	llm model.Model,
	messages []model.Message,
	maxTokens int,
) (lmeModelResult, error) {
	temp := 0.0
	req := &model.Request{
		Messages: messages,
		GenerationConfig: model.GenerationConfig{
			Stream:      false,
			MaxTokens:   &maxTokens,
			Temperature: &temp,
		},
	}
	ch, err := llm.GenerateContent(ctx, req)
	if err != nil {
		return lmeModelResult{}, err
	}
	var b strings.Builder
	var usage *model.Usage
	for resp := range ch {
		if resp == nil {
			continue
		}
		if resp.Error != nil {
			return lmeModelResult{}, errors.New(resp.Error.Message)
		}
		if len(resp.Choices) > 0 {
			b.WriteString(resp.Choices[0].Message.Content)
		}
		if resp.Usage != nil {
			usage = resp.Usage
		}
	}
	text := strings.TrimSpace(b.String())
	if text == "" {
		return lmeModelResult{}, fmt.Errorf("model returned empty response")
	}
	return lmeModelResult{Text: text, Usage: usage}, nil
}

func runLMERunnerWithRetry(
	ctx context.Context,
	maxRetries int,
	run func() (<-chan *event.Event, error),
) (lmeCollectResult, error) {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		ch, err := run()
		if err == nil {
			var res lmeCollectResult
			res, err = collectLMEEvents(ch)
			if err == nil {
				res.RetryCount = attempt
				return res, nil
			}
		}
		lastErr = err
		if !isLMETransportError(err) || attempt == maxRetries {
			return lmeCollectResult{RetryCount: attempt}, err
		}
		log.Printf("LongMemEval runner transport retry %d/%d after error: %v", attempt+1, maxRetries, err)
		if sleepErr := lmeRetrySleep(ctx, attempt); sleepErr != nil {
			return lmeCollectResult{RetryCount: attempt}, sleepErr
		}
	}
	return lmeCollectResult{RetryCount: maxRetries}, lastErr
}

func collectLMEEvents(ch <-chan *event.Event) (lmeCollectResult, error) {
	var res lmeCollectResult
	step := 0
	var pending []lmeToolCallTrace
	for ev := range ch {
		if ev == nil {
			continue
		}
		if ev.Error != nil {
			return res, errors.New(ev.Error.Message)
		}
		if ev.Response == nil {
			if ev.IsRunnerCompletion() {
				break
			}
			continue
		}
		if ev.Response.Usage != nil {
			res.Usage.PromptTokens += ev.Response.Usage.PromptTokens
			res.Usage.CompletionTokens += ev.Response.Usage.CompletionTokens
			res.Usage.TotalTokens += ev.Response.Usage.TotalTokens
			res.Usage.CachedTokens += ev.Response.Usage.PromptTokensDetails.CachedTokens
			res.Usage.LLMCalls++
		}
		if len(ev.Response.Choices) == 0 {
			continue
		}
		msg := ev.Response.Choices[0].Message
		hasToolCalls := len(msg.ToolCalls) > 0
		if hasToolCalls {
			step++
			st := lmeStepTrace{Step: step}
			if ev.Response.Usage != nil {
				st.PromptTokens = ev.Response.Usage.PromptTokens
				st.CompletionTokens = ev.Response.Usage.CompletionTokens
				st.TotalTokens = ev.Response.Usage.TotalTokens
				st.CachedTokens = ev.Response.Usage.PromptTokensDetails.CachedTokens
			}
			pending = make([]lmeToolCallTrace, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				pending = append(pending, lmeToolCallTrace{
					Name: tc.Function.Name,
					Args: string(tc.Function.Arguments),
				})
			}
			st.ToolCalls = pending
			res.Steps = append(res.Steps, st)
			res.Trace = append(res.Trace, lmeMessageTrace{
				Step:      step,
				Role:      string(msg.Role),
				Content:   msg.Content,
				ToolCalls: cloneLMEToolCalls(pending),
			})
		}
		if ev.Response.Object == model.ObjectTypeToolResponse && msg.Role == model.RoleTool {
			matched := false
			for i := range pending {
				if pending[i].Result == "" {
					pending[i].Result = msg.Content
					matched = true
					break
				}
			}
			if !matched && len(res.Steps) > 0 {
				last := &res.Steps[len(res.Steps)-1]
				last.ToolCalls = append(last.ToolCalls, lmeToolCallTrace{
					Name:   msg.ToolName,
					Result: msg.Content,
				})
			}
			res.Trace = append(res.Trace, lmeMessageTrace{
				Step:    step,
				Role:    string(msg.Role),
				Name:    msg.ToolName,
				Content: msg.Content,
			})
		}
		if msg.Role == model.RoleAssistant && msg.Content != "" {
			res.Text = msg.Content
			if !hasToolCalls {
				res.Trace = append(res.Trace, lmeMessageTrace{
					Step:    step,
					Role:    string(msg.Role),
					Content: msg.Content,
				})
			}
		}
		if ev.IsRunnerCompletion() {
			break
		}
	}
	res.Text = strings.TrimSpace(res.Text)
	if res.Text == "" {
		return res, fmt.Errorf("runner returned empty response")
	}
	return res, nil
}

func cloneLMEToolCalls(calls []lmeToolCallTrace) []lmeToolCallTrace {
	if len(calls) == 0 {
		return nil
	}
	out := make([]lmeToolCallTrace, len(calls))
	copy(out, calls)
	return out
}

func isLMETransportError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "429") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "too many requests") ||
		strings.Contains(msg, "server_busy") ||
		strings.Contains(msg, "timeout") ||
		strings.Contains(msg, "connection") ||
		strings.Contains(msg, "connection reset") ||
		strings.Contains(msg, "temporary") ||
		strings.Contains(msg, "\"code\":\"4029\"")
}

func lmeRetrySleep(ctx context.Context, attempt int) error {
	d := time.Duration(1<<attempt) * time.Second
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func modelUsageToScenarioUsage(u *model.Usage) *scenarios.TokenUsage {
	if u == nil {
		return &scenarios.TokenUsage{}
	}
	return &scenarios.TokenUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		CachedTokens:     u.PromptTokensDetails.CachedTokens,
		LLMCalls:         1,
	}
}

func parseLMETime(raw string) (time.Time, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{
		time.RFC3339,
		"2006-01-02 15:04:05",
		"2006-01-02",
		"2 January 2006",
		"January 2, 2006",
		"Jan 2, 2006",
		"2 Jan 2006",
		"January 2006",
		"Jan 2006",
		"2006-01",
		"2006",
	} {
		if t, err := time.Parse(layout, raw); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

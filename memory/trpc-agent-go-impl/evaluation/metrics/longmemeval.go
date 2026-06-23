//
// Tencent is pleased to support the open source community by making
// trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

package metrics

import (
	"fmt"
	"math"
	"strings"
)

// AnswerMetrics holds deterministic answer-quality metrics.
type AnswerMetrics struct {
	F1       float64 `json:"f1"`
	BLEU     float64 `json:"bleu"`
	ROUGE1   float64 `json:"rouge_1"`
	ROUGE2   float64 `json:"rouge_2"`
	ROUGEL   float64 `json:"rouge_l"`
	Accuracy float64 `json:"accuracy"`
}

// CalculateAnswerMetrics computes deterministic text metrics.
func CalculateAnswerMetrics(prediction, groundTruth string) AnswerMetrics {
	return AnswerMetrics{
		F1:     CalculateF1(prediction, groundTruth),
		BLEU:   CalculateBLEU(prediction, groundTruth),
		ROUGE1: CalculateROUGE1(prediction, groundTruth),
		ROUGE2: CalculateROUGE2(prediction, groundTruth),
		ROUGEL: CalculateROUGEL(prediction, groundTruth),
	}
}

// Add merges another AnswerMetrics into the receiver.
func (m *AnswerMetrics) Add(other AnswerMetrics) {
	m.F1 += other.F1
	m.BLEU += other.BLEU
	m.ROUGE1 += other.ROUGE1
	m.ROUGE2 += other.ROUGE2
	m.ROUGEL += other.ROUGEL
	m.Accuracy += other.Accuracy
}

// Divide divides all metrics by n.
func (m *AnswerMetrics) Divide(n float64) {
	if n == 0 {
		return
	}
	m.F1 /= n
	m.BLEU /= n
	m.ROUGE1 /= n
	m.ROUGE2 /= n
	m.ROUGEL /= n
	m.Accuracy /= n
}

// LongMemEvalJudgePrompt returns the official LongMemEval QA judge prompt.
func LongMemEvalJudgePrompt(
	task, question, answer, response string,
	abstention bool,
) (string, error) {
	if abstention {
		template := "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: %s\n\nExplanation: %s\n\nModel Response: %s\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
		return fmt.Sprintf(template, question, answer, response), nil
	}
	switch task {
	case "single-session-user", "single-session-assistant", "multi-session":
		template := "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: %s\n\nCorrect Answer: %s\n\nModel Response: %s\n\nIs the model response correct? Answer yes or no only."
		return fmt.Sprintf(template, question, answer, response), nil
	case "temporal-reasoning":
		template := "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: %s\n\nCorrect Answer: %s\n\nModel Response: %s\n\nIs the model response correct? Answer yes or no only."
		return fmt.Sprintf(template, question, answer, response), nil
	case "knowledge-update":
		template := "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: %s\n\nCorrect Answer: %s\n\nModel Response: %s\n\nIs the model response correct? Answer yes or no only."
		return fmt.Sprintf(template, question, answer, response), nil
	case "single-session-preference":
		template := "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: %s\n\nRubric: %s\n\nModel Response: %s\n\nIs the model response correct? Answer yes or no only."
		return fmt.Sprintf(template, question, answer, response), nil
	default:
		return "", fmt.Errorf("unsupported LongMemEval question type %q", task)
	}
}

// ParseLongMemEvalJudgeLabel parses an official yes/no judge response strictly.
func ParseLongMemEvalJudgeLabel(response string) (bool, error) {
	normalized := strings.TrimSpace(strings.ToLower(response))
	normalized = strings.Trim(normalized, ".! \n\t\r")
	switch normalized {
	case "yes":
		return true, nil
	case "no":
		return false, nil
	default:
		return false, fmt.Errorf("judge response is not an exact yes/no: %q", response)
	}
}

// RetrievalMetrics stores recall and NDCG at a cutoff.
type RetrievalMetrics struct {
	RecallAny float64 `json:"recall_any"`
	RecallAll float64 `json:"recall_all"`
	NDCGAny   float64 `json:"ndcg_any"`
}

// RetrievalMetricsAtK stores retrieval metrics keyed by cutoff.
type RetrievalMetricsAtK map[int]RetrievalMetrics

// EvaluateRetrieval computes LongMemEval retrieval metrics at each cutoff.
func EvaluateRetrieval(
	rankedIDs, correctIDs []string,
	corpusIDs []string,
	cutoffs []int,
) RetrievalMetricsAtK {
	out := make(RetrievalMetricsAtK, len(cutoffs))
	for _, k := range cutoffs {
		out[k] = evaluateRetrievalAtK(rankedIDs, correctIDs, corpusIDs, k)
	}
	return out
}

// EvaluateRetrievalTurnToSession evaluates turn retrieval after session folding.
func EvaluateRetrievalTurnToSession(
	rankedTurnIDs, correctTurnIDs []string,
	corpusTurnIDs []string,
	cutoffs []int,
) RetrievalMetricsAtK {
	rankedSessions := make([]string, 0, len(rankedTurnIDs))
	for _, id := range rankedTurnIDs {
		rankedSessions = append(rankedSessions, stripTurnID(id))
	}
	correctSessions := uniqueStringsMap(correctTurnIDs, stripTurnID)
	corpusSessions := make([]string, 0, len(corpusTurnIDs))
	for _, id := range corpusTurnIDs {
		corpusSessions = append(corpusSessions, stripTurnID(id))
	}
	out := make(RetrievalMetricsAtK, len(cutoffs))
	for _, k := range cutoffs {
		effectiveK := effectiveUniqueCutoff(rankedSessions, k)
		out[k] = evaluateRetrievalAtK(
			rankedSessions,
			correctSessions,
			corpusSessions,
			effectiveK,
		)
	}
	return out
}

func evaluateRetrievalAtK(
	rankedIDs, correctIDs []string,
	corpusIDs []string,
	k int,
) RetrievalMetrics {
	if k < 0 {
		k = 0
	}
	if k > len(rankedIDs) {
		k = len(rankedIDs)
	}
	correct := make(map[string]struct{}, len(correctIDs))
	for _, id := range correctIDs {
		correct[id] = struct{}{}
	}
	recalled := make(map[string]struct{}, k)
	for _, id := range rankedIDs[:k] {
		recalled[id] = struct{}{}
	}
	hits := 0
	for id := range correct {
		if _, ok := recalled[id]; ok {
			hits++
		}
	}
	var recallAll float64
	if len(correct) == 0 {
		recallAll = 1
	} else {
		recallAll = boolToFloat(hits == len(correct))
	}
	return RetrievalMetrics{
		RecallAny: boolToFloat(hits > 0),
		RecallAll: recallAll,
		NDCGAny:   ndcg(rankedIDs, correct, corpusIDs, k),
	}
}

func ndcg(
	rankedIDs []string,
	correct map[string]struct{},
	corpusIDs []string,
	k int,
) float64 {
	relevance := make([]float64, 0, len(rankedIDs))
	for _, id := range rankedIDs {
		_, ok := correct[id]
		relevance = append(relevance, boolToFloat(ok))
	}
	actual := dcg(relevance, k)
	ideal := make([]float64, 0, len(corpusIDs))
	for _, id := range corpusIDs {
		_, ok := correct[id]
		ideal = append(ideal, boolToFloat(ok))
	}
	sortDescFloat64(ideal)
	idealDCG := dcg(ideal, k)
	if idealDCG == 0 {
		return 0
	}
	return actual / idealDCG
}

func dcg(relevances []float64, k int) float64 {
	if k > len(relevances) {
		k = len(relevances)
	}
	if k <= 0 {
		return 0
	}
	score := relevances[0]
	for i := 1; i < k; i++ {
		score += relevances[i] / math.Log2(float64(i+2))
	}
	return score
}

func effectiveUniqueCutoff(ids []string, k int) int {
	effectiveK := k
	for effectiveK <= len(ids) {
		seen := make(map[string]struct{}, effectiveK)
		for _, id := range ids[:effectiveK] {
			seen[id] = struct{}{}
		}
		if len(seen) >= k {
			return effectiveK
		}
		effectiveK++
	}
	return len(ids)
}

func uniqueStringsMap(ids []string, mapper func(string) string) []string {
	seen := make(map[string]struct{}, len(ids))
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		mapped := mapper(id)
		if _, ok := seen[mapped]; ok {
			continue
		}
		seen[mapped] = struct{}{}
		out = append(out, mapped)
	}
	return out
}

func stripTurnID(id string) string {
	idx := strings.LastIndex(id, "_")
	if idx <= 0 {
		return id
	}
	return id[:idx]
}

func sortDescFloat64(values []float64) {
	for i := 0; i < len(values); i++ {
		for j := i + 1; j < len(values); j++ {
			if values[j] > values[i] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
}

func boolToFloat(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

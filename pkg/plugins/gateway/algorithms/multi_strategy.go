/*
Copyright 2025 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package routingalgorithms

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// multiStrategyRouter implements soft scoring across multiple routing strategies.
//
// Given a routing-strategy header value like "least-request,least-kv-cache", it:
//  1. Resolves each named strategy to a PodScorer.
//  2. For each pod, calls ScorePod on every scorer.
//  3. Normalises the raw scores across all pods using min-max normalisation so that
//     each sub-strategy contributes a value in [0, 1] regardless of its original scale.
//  4. Computes a weighted sum of the normalised sub-scores (equal weights by default).
//  5. Sorts pods by ascending composite score and routes to the best pod.
//
// This avoids the hard lexicographic isolation of repeated-bucket-sort approaches:
// a pod that scores badly on one dimension can still be preferred overall if it
// excels on the other dimensions.
type multiStrategyRouter struct {
	scorers  []namedScorer
	fallback types.Router
}

type namedScorer struct {
	name   types.RoutingAlgorithm
	scorer types.PodScorer
	weight float64
}

// scoredPod associates a pod with its computed composite score.
type scoredPod struct {
	pod   *v1.Pod
	score float64
}

// newMultiStrategyRouter constructs a multiStrategyRouter from a comma-separated list of strategy names.
// Each strategy must be registered in the RouterManager and must implement PodScorer.
// Strategies that do not implement PodScorer are silently skipped with a warning.
// ctx is the routing context for the current request; it is forwarded to each RouterProviderFunc
// so that context-aware provider functions can initialise correctly.
func newMultiStrategyRouter(rm *RouterManager, strategies []string, ctx *types.RoutingContext) (*multiStrategyRouter, error) {
	var scorers []namedScorer
	for _, s := range strategies {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		alg := types.RoutingAlgorithm(s)
		provider, ok := rm.routerFactory[alg]
		if !ok {
			return nil, fmt.Errorf("unknown routing strategy %q in multi-strategy chain", s)
		}
		// Instantiate the router using the caller's routing context.
		router, err := provider(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to instantiate router %q: %w", s, err)
		}
		scorer, ok := router.(types.PodScorer)
		if !ok {
			klog.Warningf("multi-strategy: router %q does not implement PodScorer; skipping in composite score", s)
			continue
		}
		scorers = append(scorers, namedScorer{name: alg, scorer: scorer, weight: 1.0})
	}

	if len(scorers) == 0 {
		return nil, fmt.Errorf("no strategies in %v implement PodScorer; cannot build multi-strategy router", strategies)
	}

	return &multiStrategyRouter{
		scorers:  scorers,
		fallback: RandomRouter,
	}, nil
}

// Route implements types.Router.
func (r *multiStrategyRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	pods := readyPodList.All()
	if len(pods) == 0 {
		return "", fmt.Errorf("no pods to forward request")
	}

	ranked := r.rankPods(ctx, pods)
	if len(ranked) == 0 {
		// Fallback to random if ranking failed for all pods.
		klog.Warningf("multi-strategy: ranking produced no results for requestID=%s; falling back to random", ctx.RequestID)
		targetPod, err := utils.SelectRandomPod(pods, rand.Intn)
		if err != nil {
			return "", err
		}
		ctx.SetTargetPod(targetPod)
		return ctx.TargetAddress(), nil
	}

	// The first pod in the sorted list is the best candidate.
	best := ranked[0]
	klog.V(4).InfoS("multi-strategy selected pod",
		"requestID", ctx.RequestID,
		"pod", best.pod.Name,
		"compositeScore", best.score,
	)
	ctx.SetTargetPod(best.pod)
	return ctx.TargetAddress(), nil
}

// rankPods scores every pod with each sub-scorer, normalises per sub-scorer, and
// returns the pods sorted by ascending weighted-sum composite score.
func (r *multiStrategyRouter) rankPods(ctx *types.RoutingContext, pods []*v1.Pod) []scoredPod {
	n := len(pods)
	if n == 0 {
		return nil
	}

	// rawScores[scorerIdx][podIdx] = raw score from scorer i for pod j.
	rawScores := make([][]float64, len(r.scorers))
	for i, s := range r.scorers {
		rawScores[i] = make([]float64, n)
		for j, pod := range pods {
			rawScores[i][j] = s.scorer.ScorePod(ctx, pod)
		}
	}

	// Normalise each scorer's scores with min-max to [0, 1].
	normScores := make([][]float64, len(r.scorers))
	for i := range r.scorers {
		normScores[i] = minMaxNormalise(rawScores[i])
	}

	// Compute total weight.
	var totalWeight float64
	for _, s := range r.scorers {
		totalWeight += s.weight
	}
	if totalWeight == 0 {
		totalWeight = 1
	}

	// Build composite scores.
	result := make([]scoredPod, n)
	for j, pod := range pods {
		var composite float64
		for i, s := range r.scorers {
			composite += (s.weight / totalWeight) * normScores[i][j]
		}
		result[j] = scoredPod{pod: pod, score: composite}
	}

	// Sort ascending (lower is better).
	sort.Slice(result, func(a, b int) bool {
		return result[a].score < result[b].score
	})

	return result
}

// minMaxNormalise rescales values to [0, 1].
// If all values are identical the function returns a zero-filled slice (no preference).
// Values equal to math.MaxFloat64 (sentinel for "no data") are treated as the maximum.
func minMaxNormalise(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}

	minVal := math.MaxFloat64
	maxVal := -math.MaxFloat64
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	out := make([]float64, len(values))
	span := maxVal - minVal
	if span == 0 {
		// All identical; return zeros so all pods are equally preferred.
		return out
	}
	for i, v := range values {
		out[i] = (v - minVal) / span
	}
	return out
}

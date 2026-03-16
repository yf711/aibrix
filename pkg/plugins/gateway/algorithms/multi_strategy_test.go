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
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makePod creates a pod with the given name and IP that is always Ready.
func makePod(name, ip string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
		Status: v1.PodStatus{
			PodIP: ip,
			Conditions: []v1.PodCondition{
				{Type: v1.PodReady, Status: v1.ConditionTrue},
			},
		},
	}
}

// makePodList wraps a slice of pods in a PodArray.
func makePodList(pods ...*v1.Pod) types.PodList {
	return &utils.PodArray{Pods: pods}
}

// routingCtx builds a RoutingContext with the given model name.
func routingCtx(model string) *types.RoutingContext {
	return types.NewRoutingContext(context.Background(), RouterNotSet, model, "", "test-id", "")
}

// ---------------------------------------------------------------------------
// minMaxNormalise
// ---------------------------------------------------------------------------

func TestMinMaxNormalise(t *testing.T) {
	t.Run("all identical values produce zeros", func(t *testing.T) {
		out := minMaxNormalise([]float64{5, 5, 5})
		for _, v := range out {
			assert.Equal(t, 0.0, v)
		}
	})

	t.Run("normal case", func(t *testing.T) {
		out := minMaxNormalise([]float64{0, 5, 10})
		assert.InDelta(t, 0.0, out[0], 1e-9)
		assert.InDelta(t, 0.5, out[1], 1e-9)
		assert.InDelta(t, 1.0, out[2], 1e-9)
	})

	t.Run("MaxFloat64 sentinel treated as max", func(t *testing.T) {
		out := minMaxNormalise([]float64{0, 10, math.MaxFloat64})
		assert.InDelta(t, 0.0, out[0], 1e-9)
		assert.InDelta(t, 1.0, out[2], 1e-9)
		// middle value should be between 0 and 1
		assert.Greater(t, out[1], 0.0)
		assert.Less(t, out[1], 1.0)
	})

	t.Run("single element", func(t *testing.T) {
		out := minMaxNormalise([]float64{42})
		assert.Equal(t, 0.0, out[0])
	})

	t.Run("empty slice", func(t *testing.T) {
		out := minMaxNormalise(nil)
		assert.Nil(t, out)
	})
}

// ---------------------------------------------------------------------------
// RouterManager.Validate – multi-strategy paths
// ---------------------------------------------------------------------------

func TestValidateMultiStrategy(t *testing.T) {
	rm := NewRouterManager()
	// Register two fake strategies.
	rm.routerFactory[types.RoutingAlgorithm("alpha")] = func(_ *types.RoutingContext) (types.Router, error) {
		return &fakeScorer{score: 1}, nil
	}
	rm.routerFactory[types.RoutingAlgorithm("beta")] = func(_ *types.RoutingContext) (types.Router, error) {
		return &fakeScorer{score: 2}, nil
	}

	t.Run("single valid strategy", func(t *testing.T) {
		alg, ok := rm.Validate("alpha")
		assert.True(t, ok)
		assert.Equal(t, types.RoutingAlgorithm("alpha"), alg)
	})

	t.Run("two valid strategies", func(t *testing.T) {
		alg, ok := rm.Validate("alpha,beta")
		assert.True(t, ok)
		assert.Equal(t, types.RoutingAlgorithm("alpha,beta"), alg)
	})

	t.Run("one unknown strategy returns false", func(t *testing.T) {
		_, ok := rm.Validate("alpha,unknown")
		assert.False(t, ok)
	})

	t.Run("empty string returns false", func(t *testing.T) {
		_, ok := rm.Validate("")
		assert.False(t, ok)
	})

	t.Run("spaces around names are tolerated", func(t *testing.T) {
		alg, ok := rm.Validate(" alpha , beta ")
		assert.True(t, ok)
		assert.Equal(t, types.RoutingAlgorithm(" alpha , beta "), alg)
	})
}

// ---------------------------------------------------------------------------
// RouterManager.Select – multi-strategy path
// ---------------------------------------------------------------------------

func TestSelectMultiStrategy(t *testing.T) {
	rm := NewRouterManager()
	rm.routerFactory[types.RoutingAlgorithm("alpha")] = func(_ *types.RoutingContext) (types.Router, error) {
		return &fakeScorer{score: 1}, nil
	}
	rm.routerFactory[types.RoutingAlgorithm("beta")] = func(_ *types.RoutingContext) (types.Router, error) {
		return &fakeScorer{score: 2}, nil
	}

	t.Run("returns multiStrategyRouter for comma list", func(t *testing.T) {
		ctx := routingCtx("m")
		ctx.Algorithm = types.RoutingAlgorithm("alpha,beta")
		router, err := rm.Select(ctx)
		require.NoError(t, err)
		_, ok := router.(*multiStrategyRouter)
		assert.True(t, ok, "expected *multiStrategyRouter")
	})

	t.Run("all-unknown-scorer list returns error", func(t *testing.T) {
		// Register a strategy that does NOT implement PodScorer.
		rm.routerFactory[types.RoutingAlgorithm("noScorer")] = func(_ *types.RoutingContext) (types.Router, error) {
			return &noScorerRouter{}, nil
		}
		ctx := routingCtx("m")
		ctx.Algorithm = types.RoutingAlgorithm("noScorer,noScorer")
		_, err := rm.Select(ctx)
		assert.Error(t, err)
	})
}

// ---------------------------------------------------------------------------
// multiStrategyRouter.Route
// ---------------------------------------------------------------------------

func TestMultiStrategyRoute(t *testing.T) {
	t.Run("selects pod with lowest composite score", func(t *testing.T) {
		// Pod A: scorer1=10, scorer2=50  → composite after norm should be higher
		// Pod B: scorer1=1,  scorer2=5   → composite should be lower → selected
		podA := makePod("podA", "10.0.0.1")
		podB := makePod("podB", "10.0.0.2")

		router := &multiStrategyRouter{
			scorers: []namedScorer{
				{name: "s1", scorer: &perPodScorer{scores: map[string]float64{"podA": 10, "podB": 1}}, weight: 1},
				{name: "s2", scorer: &perPodScorer{scores: map[string]float64{"podA": 50, "podB": 5}}, weight: 1},
			},
		}

		ctx := routingCtx("model")
		addr, err := router.Route(ctx, makePodList(podA, podB))
		require.NoError(t, err)
		assert.Contains(t, addr, "10.0.0.2", "podB should be selected")
	})

	t.Run("soft scoring avoids hard lexicographic trap", func(t *testing.T) {
		// Reproduces the colleague's counter-example:
		//   C: least-request=0, load=99  (busy even though low queue)
		//   D: least-request=1, load=10  (slightly more queued but much less loaded)
		//
		// Hard bucket-sort on least-request first would always prefer C.
		// Soft scoring with equal weights:
		//   norm(least-request): C=0/(1-0)=0, D=(1-0)/(1-0)=1  → C is better here
		//   norm(load):          C=(99-10)/(99-10)=1, D=(10-10)/(99-10)=0 → D is better here
		//   composite C = (0 + 1) / 2 = 0.5
		//   composite D = (1 + 0) / 2 = 0.5  → tie
		//
		// To ensure D wins, give load a 2× weight:
		//   composite C = (1*0 + 2*1) / 3 = 0.667
		//   composite D = (1*1 + 2*0) / 3 = 0.333  → D wins
		podC := makePod("podC", "10.0.0.3")
		podD := makePod("podD", "10.0.0.4")

		router := &multiStrategyRouter{
			scorers: []namedScorer{
				{name: "least-request", scorer: &perPodScorer{scores: map[string]float64{"podC": 0, "podD": 1}}, weight: 1},
				{name: "load", scorer: &perPodScorer{scores: map[string]float64{"podC": 99, "podD": 10}}, weight: 2},
			},
		}

		ctx := routingCtx("model")
		addr, err := router.Route(ctx, makePodList(podC, podD))
		require.NoError(t, err)
		assert.Contains(t, addr, "10.0.0.4", "podD should win due to weighted soft scoring even though podC has fewer queued requests")
	})

	t.Run("single scorer delegates correctly", func(t *testing.T) {
		podA := makePod("podA", "10.0.0.1")
		podB := makePod("podB", "10.0.0.2")

		router := &multiStrategyRouter{
			scorers: []namedScorer{
				{name: "s1", scorer: &perPodScorer{scores: map[string]float64{"podA": 100, "podB": 1}}, weight: 1},
			},
		}
		ctx := routingCtx("model")
		addr, err := router.Route(ctx, makePodList(podA, podB))
		require.NoError(t, err)
		assert.Contains(t, addr, "10.0.0.2")
	})

	t.Run("all max sentinel falls back to random", func(t *testing.T) {
		podA := makePod("podA", "10.0.0.1")
		podB := makePod("podB", "10.0.0.2")

		router := &multiStrategyRouter{
			scorers: []namedScorer{
				{name: "s1", scorer: &perPodScorer{scores: map[string]float64{
					"podA": math.MaxFloat64, "podB": math.MaxFloat64}}, weight: 1},
			},
			fallback: RandomRouter,
		}
		ctx := routingCtx("model")
		addr, err := router.Route(ctx, makePodList(podA, podB))
		require.NoError(t, err)
		assert.NotEmpty(t, addr)
	})

	t.Run("no pods returns error", func(t *testing.T) {
		router := &multiStrategyRouter{
			scorers: []namedScorer{
				{name: "s1", scorer: &perPodScorer{scores: map[string]float64{}}, weight: 1},
			},
		}
		ctx := routingCtx("model")
		_, err := router.Route(ctx, makePodList())
		assert.Error(t, err)
	})
}

// ---------------------------------------------------------------------------
// PodScorer implementations on existing routers
// ---------------------------------------------------------------------------

func TestLeastRequestScorePod(t *testing.T) {
	model := "test-model"
	c := cache.NewWithPodsMetricsForTest(
		[]*v1.Pod{makePod("p1", "1.1.1.1"), makePod("p2", "2.2.2.2")},
		model,
		map[string]map[string]metrics.MetricValue{
			"p1": {metrics.RealtimeNumRequestsRunning: &metrics.SimpleMetricValue{Value: 3}},
			"p2": {metrics.RealtimeNumRequestsRunning: &metrics.SimpleMetricValue{Value: 7}},
		},
	)
	r := leastRequestRouter{cache: c}
	ctx := routingCtx(model)

	scoreP1 := r.ScorePod(ctx, makePod("p1", "1.1.1.1"))
	scoreP2 := r.ScorePod(ctx, makePod("p2", "2.2.2.2"))
	assert.Equal(t, 3.0, scoreP1)
	assert.Equal(t, 7.0, scoreP2)
	assert.Less(t, scoreP1, scoreP2, "p1 should have a lower (better) score")
}

func TestLeastKvCacheScorePod(t *testing.T) {
	model := "test-model"
	c := cache.NewWithPodsMetricsForTest(
		[]*v1.Pod{makePod("p1", "1.1.1.1"), makePod("p2", "2.2.2.2")},
		model,
		map[string]map[string]metrics.MetricValue{
			"p1": {
				metrics.KVCacheUsagePerc:  &metrics.SimpleMetricValue{Value: 0.2},
				metrics.CPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.1},
			},
			"p2": {
				metrics.KVCacheUsagePerc:  &metrics.SimpleMetricValue{Value: 0.7},
				metrics.CPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.2},
			},
		},
	)
	r := leastKvCacheRouter{cache: c}
	ctx := routingCtx(model)

	scoreP1 := r.ScorePod(ctx, makePod("p1", "1.1.1.1"))
	scoreP2 := r.ScorePod(ctx, makePod("p2", "2.2.2.2"))
	assert.InDelta(t, 0.3, scoreP1, 1e-9)
	assert.InDelta(t, 0.9, scoreP2, 1e-9)
	assert.Less(t, scoreP1, scoreP2)
}

func TestLeastUtilScorePod(t *testing.T) {
	model := "test-model"
	c := cache.NewWithPodsMetricsForTest(
		[]*v1.Pod{makePod("p1", "1.1.1.1"), makePod("p2", "2.2.2.2")},
		model,
		map[string]map[string]metrics.MetricValue{
			"p1": {metrics.EngineUtilization: &metrics.SimpleMetricValue{Value: 0.4}},
			"p2": {metrics.EngineUtilization: &metrics.SimpleMetricValue{Value: 0.8}},
		},
	)
	r := leastUtilRouter{cache: c}
	ctx := routingCtx(model)

	scoreP1 := r.ScorePod(ctx, makePod("p1", "1.1.1.1"))
	scoreP2 := r.ScorePod(ctx, makePod("p2", "2.2.2.2"))
	assert.InDelta(t, 0.4, scoreP1, 1e-9)
	assert.InDelta(t, 0.8, scoreP2, 1e-9)
	assert.Less(t, scoreP1, scoreP2)
}

func TestLeastBusyTimeScorePod(t *testing.T) {
	model := "test-model"
	c := cache.NewWithPodsMetricsForTest(
		[]*v1.Pod{makePod("p1", "1.1.1.1"), makePod("p2", "2.2.2.2")},
		model,
		map[string]map[string]metrics.MetricValue{
			"p1": {metrics.GPUBusyTimeRatio: &metrics.SimpleMetricValue{Value: 0.3}},
			"p2": {metrics.GPUBusyTimeRatio: &metrics.SimpleMetricValue{Value: 0.9}},
		},
	)
	r := leastBusyTimeRouter{cache: c}
	ctx := routingCtx(model)

	scoreP1 := r.ScorePod(ctx, makePod("p1", "1.1.1.1"))
	scoreP2 := r.ScorePod(ctx, makePod("p2", "2.2.2.2"))
	assert.Less(t, scoreP1, scoreP2)
}

func TestLeastGpuCacheScorePod(t *testing.T) {
	model := "test-model"
	c := cache.NewWithPodsMetricsForTest(
		[]*v1.Pod{makePod("p1", "1.1.1.1"), makePod("p2", "2.2.2.2")},
		model,
		map[string]map[string]metrics.MetricValue{
			"p1": {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.1}},
			"p2": {metrics.GPUCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.6}},
		},
	)
	r := leastGpuCacheRouter{cache: c}
	ctx := routingCtx(model)

	scoreP1 := r.ScorePod(ctx, makePod("p1", "1.1.1.1"))
	scoreP2 := r.ScorePod(ctx, makePod("p2", "2.2.2.2"))
	assert.Less(t, scoreP1, scoreP2)
}

// MissingMetric returns MaxFloat64 sentinel when metric is unavailable.
func TestScorePodMissingMetric(t *testing.T) {
	c := &cache.Store{} // empty store → all metric lookups fail
	r := leastRequestRouter{cache: c}
	score := r.ScorePod(routingCtx("model"), makePod("p1", "1.1.1.1"))
	assert.Equal(t, math.MaxFloat64, score)
}

// TestLeastKvCacheScorePodPartialData verifies that ScorePod returns a partial sum
// when only one of the two cache metrics is available.
func TestLeastKvCacheScorePodPartialData(t *testing.T) {
	model := "test-model"
	// p1 has only GPU cache, no CPU cache
	c := cache.NewWithPodsMetricsForTest(
		[]*v1.Pod{makePod("p1", "1.1.1.1")},
		model,
		map[string]map[string]metrics.MetricValue{
			"p1": {
				metrics.KVCacheUsagePerc: &metrics.SimpleMetricValue{Value: 0.5},
				// CPUCacheUsagePerc intentionally absent
			},
		},
	)
	r := leastKvCacheRouter{cache: c}
	ctx := routingCtx(model)

	score := r.ScorePod(ctx, makePod("p1", "1.1.1.1"))
	// Should return only the GPU component (0.5), not MaxFloat64.
	assert.InDelta(t, 0.5, score, 1e-9)
}

// ---------------------------------------------------------------------------
// Integration: newMultiStrategyRouter error paths
// ---------------------------------------------------------------------------

func TestNewMultiStrategyRouterErrors(t *testing.T) {
	rm := NewRouterManager()
	rm.routerFactory[types.RoutingAlgorithm("ok")] = func(_ *types.RoutingContext) (types.Router, error) {
		return &fakeScorer{score: 1}, nil
	}

	t.Run("unknown strategy returns error", func(t *testing.T) {
		_, err := newMultiStrategyRouter(rm, []string{"ok", "unknown"}, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unknown")
	})

	t.Run("no PodScorer implementors returns error", func(t *testing.T) {
		rm.routerFactory[types.RoutingAlgorithm("noScorer")] = func(_ *types.RoutingContext) (types.Router, error) {
			return &noScorerRouter{}, nil
		}
		_, err := newMultiStrategyRouter(rm, []string{"noScorer"}, nil)
		assert.Error(t, err)
	})

	t.Run("valid scorer list succeeds", func(t *testing.T) {
		router, err := newMultiStrategyRouter(rm, []string{"ok"}, nil)
		require.NoError(t, err)
		assert.NotNil(t, router)
		assert.Len(t, router.scorers, 1)
	})
}

// ---------------------------------------------------------------------------
// Fake helpers
// ---------------------------------------------------------------------------

// fakeScorer implements both Router and PodScorer. All pods get the same fixed score.
type fakeScorer struct {
	score float64
}

func (f *fakeScorer) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	all := pods.All()
	if len(all) == 0 {
		return "", nil
	}
	ctx.SetTargetPod(all[0])
	return ctx.TargetAddress(), nil
}

func (f *fakeScorer) ScorePod(_ *types.RoutingContext, _ *v1.Pod) float64 {
	return f.score
}

// noScorerRouter implements Router but NOT PodScorer.
type noScorerRouter struct{}

func (n *noScorerRouter) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	all := pods.All()
	if len(all) == 0 {
		return "", nil
	}
	ctx.SetTargetPod(all[0])
	return ctx.TargetAddress(), nil
}

// perPodScorer returns a different score per pod name.
type perPodScorer struct {
	scores map[string]float64
}

func (p *perPodScorer) Route(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	all := pods.All()
	if len(all) == 0 {
		return "", nil
	}
	ctx.SetTargetPod(all[0])
	return ctx.TargetAddress(), nil
}

func (p *perPodScorer) ScorePod(_ *types.RoutingContext, pod *v1.Pod) float64 {
	if v, ok := p.scores[pod.Name]; ok {
		return v
	}
	return math.MaxFloat64
}

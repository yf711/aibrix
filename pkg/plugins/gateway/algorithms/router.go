/*
Copyright 2024 The Aibrix Team.

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
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/aibrix/pkg/types"
	"k8s.io/klog/v2"
)

const (
	RouterNotSet = ""
)

var (
	ErrInitTimeout           = errors.New("router initialization timeout")
	ErrFallbackNotSupported  = errors.New("router not support fallback")
	ErrFallbackNotRegistered = errors.New("fallback router not registered")
	defaultRM                = NewRouterManager()
)

type RouterManager struct {
	routerInited      context.Context
	routerDoneInit    context.CancelFunc
	routerFactory     map[types.RoutingAlgorithm]types.RouterProviderFunc
	routerConstructor map[types.RoutingAlgorithm]types.RouterProviderRegistrationFunc
	routerMu          sync.RWMutex
}

func NewRouterManager() *RouterManager {
	rm := &RouterManager{}
	rm.routerInited, rm.routerDoneInit = context.WithTimeout(context.Background(), 5*time.Second)
	rm.routerFactory = make(map[types.RoutingAlgorithm]types.RouterProviderFunc)
	rm.routerConstructor = make(map[types.RoutingAlgorithm]types.RouterProviderRegistrationFunc)
	return rm
}

// isMultiStrategy returns true when the algorithms string encodes a comma-separated
// list of two or more strategy names (e.g. "least-request,least-kv-cache").
func isMultiStrategy(algorithms string) bool {
	return strings.ContainsRune(algorithms, ',')
}

// Validate validates if user provided routing routers is supported by gateway.
// It accepts either a single strategy name or a comma-separated list of strategy names.
// For a comma-separated list, every named strategy must be individually registered.
func (rm *RouterManager) Validate(algorithms string) (types.RoutingAlgorithm, bool) {
	rm.routerMu.RLock()
	defer rm.routerMu.RUnlock()

	if !isMultiStrategy(algorithms) {
		if _, ok := rm.routerFactory[types.RoutingAlgorithm(algorithms)]; ok {
			return types.RoutingAlgorithm(algorithms), ok
		}
		return RouterNotSet, false
	}

	// Multi-strategy: every individual strategy must be registered.
	parts := strings.Split(algorithms, ",")
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if _, ok := rm.routerFactory[types.RoutingAlgorithm(p)]; !ok {
			return RouterNotSet, false
		}
	}
	return types.RoutingAlgorithm(algorithms), true
}
func Validate(algorithms string) (types.RoutingAlgorithm, bool) {
	return defaultRM.Validate(algorithms)
}

// Select returns the router for the algorithm stored in ctx.Algorithm.
// For comma-separated multi-strategy values a multiStrategyRouter is constructed and returned.
// Call Validate before this function to ensure expected behavior.
func (rm *RouterManager) Select(ctx *types.RoutingContext) (types.Router, error) {
	rm.routerMu.RLock()
	defer rm.routerMu.RUnlock()

	alg := string(ctx.Algorithm)

	if !isMultiStrategy(alg) {
		if provider, ok := rm.routerFactory[ctx.Algorithm]; ok {
			return provider(ctx)
		}
		klog.Warningf("Unsupported router strategy: %s, use %s instead.", ctx.Algorithm, RouterRandom)
		return RandomRouter, nil
	}

	// Build a multiStrategyRouter from the comma-separated list.
	parts := strings.Split(alg, ",")
	router, err := newMultiStrategyRouter(rm, parts, ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to build multi-strategy router: %w", err)
	}
	return router, nil
}
func Select(ctx *types.RoutingContext) (types.Router, error) {
	return defaultRM.Select(ctx)
}

func (rm *RouterManager) Register(algorithm types.RoutingAlgorithm, constructor types.RouterConstructor) {
	rm.routerMu.Lock()
	defer rm.routerMu.Unlock()
	rm.routerConstructor[algorithm] = func() types.RouterProviderFunc {
		router, err := constructor()
		if err != nil {
			klog.Errorf("Failed to construct router for %s: %v", algorithm, err)
			return nil
		}
		return func(_ *types.RoutingContext) (types.Router, error) {
			return router, nil
		}
	}
}
func Register(algorithm types.RoutingAlgorithm, constructor types.RouterConstructor) {
	defaultRM.Register(algorithm, constructor)
}

func (rm *RouterManager) RegisterProvider(algorithm types.RoutingAlgorithm, provider types.RouterProviderFunc) {
	rm.routerMu.Lock()
	defer rm.routerMu.Unlock()
	rm.routerFactory[algorithm] = provider
	klog.Infof("Registered router for %s", algorithm)
}
func RegisterProvider(algorithm types.RoutingAlgorithm, provider types.RouterProviderFunc) {
	defaultRM.RegisterProvider(algorithm, provider)
}

func (rm *RouterManager) SetFallback(router types.Router, fallback types.RoutingAlgorithm) error {
	r, ok := router.(types.FallbackRouter)
	if !ok {
		return ErrFallbackNotSupported
	}

	<-rm.routerInited.Done()
	initErr := rm.routerInited.Err()
	if initErr != context.Canceled {
		return fmt.Errorf("router did not initialized: %v", initErr)
	}

	rm.routerMu.RLock()
	defer rm.routerMu.RUnlock()

	if provider, ok := rm.routerFactory[fallback]; !ok {
		return ErrFallbackNotRegistered
	} else {
		r.SetFallback(fallback, provider)
	}
	return nil
}
func SetFallback(router types.Router, fallback types.RoutingAlgorithm) error {
	return defaultRM.SetFallback(router, fallback)
}

func (rm *RouterManager) Init() {
	rm.routerMu.Lock()
	defer rm.routerMu.Unlock()
	for algorithm, constructor := range rm.routerConstructor {
		rm.routerFactory[algorithm] = constructor()
		klog.Infof("Registered router for %s", algorithm)
	}
	rm.routerDoneInit()
}
func Init() {
	defaultRM.Init()
}

/*
Copyright 2025 The llm-d Authors.

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

package tokenization

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/daulet/tokenizers"
	lru "github.com/hashicorp/golang-lru/v2"
	"golang.org/x/sync/singleflight"
)

// tokenizersCacheSize is the size of the LRU cache for tokenizers.
// 1 tokenizer per base-model (NOT LoRAs).
const tokenizersCacheSize = 20

// Tokenizer interface defines the methods for tokenization.
type Tokenizer interface {
	// Encode tokenizes the input string and returns the token IDs and offsets.
	Encode(input, modelName string) ([]uint32, []tokenizers.Offset, error)
}

// HFTokenizerConfig holds the configuration for the HuggingFace tokenizer.
type HFTokenizerConfig struct {
	HuggingFaceToken   string            `json:"huggingFaceToken"`
	TokenizersCacheDir string            `json:"tokenizersCacheDir"` // Directory for caching tokenizers
	ModelPaths         map[string]string `json:"modelPaths"`         // Map of served-model-name to actual model path (e.g., "test-model" -> "/models/my-model")
}

// DefaultHFTokenizerConfig returns a default configuration for the HuggingFace
// tokenizer.
func DefaultHFTokenizerConfig() *HFTokenizerConfig {
	return &HFTokenizerConfig{
		HuggingFaceToken:   "",
		TokenizersCacheDir: getTokenizerCacheDir(),
		ModelPaths:         make(map[string]string),
	}
}

// CachedHFTokenizer is implements the Tokenizer interface using
// bindings to HuggingFace's rust tokenizer.
// The implementation wraps an LRU-cache for holding loaded per-model
// tokenizers.
type CachedHFTokenizer struct {
	cfg        tokenizers.TokenizerConfigOption
	cache      *lru.Cache[string, *tokenizers.Tokenizer]
	group      singleflight.Group
	modelPaths map[string]string // Map of served-model-name to actual model path
}

// NewCachedHFTokenizer creates a new instance of HFTokenizer with the provided configuration.
func NewCachedHFTokenizer(config *HFTokenizerConfig) (Tokenizer, error) {
	var cfg tokenizers.TokenizerConfigOption

	if config.TokenizersCacheDir != "" {
		cfg = tokenizers.WithCacheDir(config.TokenizersCacheDir)
	}
	if config.HuggingFaceToken != "" {
		cfg = tokenizers.WithAuthToken(config.HuggingFaceToken)
	}

	tokenizersCache, err := lru.New[string, *tokenizers.Tokenizer](tokenizersCacheSize)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tokenizer cache: %w", err)
	}

	// Initialize modelPaths map if not provided
	modelPaths := config.ModelPaths
	if modelPaths == nil {
		modelPaths = make(map[string]string)
	}

	return &CachedHFTokenizer{
		cfg:        cfg,
		cache:      tokenizersCache,
		modelPaths: modelPaths,
	}, nil
}

func (t *CachedHFTokenizer) getTokenizer(modelName string) (*tokenizers.Tokenizer, error) {
	tokenizer, ok := t.cache.Get(modelName)
	if !ok {
		result, err, shared := t.group.Do(modelName, func() (any, error) {
			return t.loadTokenizer(modelName)
		})
		if err != nil {
			return nil, err
		}

		tokenizer, ok = result.(*tokenizers.Tokenizer)
		if !ok {
			return nil, fmt.Errorf("unexpected tokenizer type from singleflight result")
		}

		if !shared {
			// Only add to cache if this goroutine actually loaded the tokenizer
			t.cache.Add(modelName, tokenizer)
		}
	}
	return tokenizer, nil
}

// loadTokenizer attempts to load a tokenizer from local path first,
// then falls back to HuggingFace if not found locally.
func (t *CachedHFTokenizer) loadTokenizer(modelName string) (*tokenizers.Tokenizer, error) {
	// Try to load from local path first
	localTokenizer, err := t.tryLoadFromLocal(modelName)
	if err == nil {
		return localTokenizer, nil
	}

	// Fallback to HuggingFace
	return tokenizers.FromPretrained(modelName, t.cfg)
}

// tryLoadFromLocal attempts to load a tokenizer from local filesystem.
// It checks multiple possible paths for the tokenizer.json file.
// Priority order:
// 1. Model path from ModelPaths mapping (e.g., /models/my-model/tokenizer.json)
// 2. Direct path interpretation of modelName
// 3. Cache directory paths
func (t *CachedHFTokenizer) tryLoadFromLocal(modelName string) (*tokenizers.Tokenizer, error) {
	var possiblePaths []string

	// PRIORITY 1: Check if we have a mapped model path for this served-model-name
	if modelPath, ok := t.modelPaths[modelName]; ok {
		// This is the actual model directory path (e.g., /models/my-model)
		possiblePaths = append(possiblePaths,
			// Check for tokenizer.json in the model directory
			filepath.Join(modelPath, "tokenizer.json"),
			// If modelPath itself is the tokenizer.json file
			modelPath,
		)
	}

	// PRIORITY 2: Try modelName as direct path (backward compatibility)
	possiblePaths = append(possiblePaths,
		// If modelName is already a path to tokenizer.json
		modelName,
		// If modelName is a directory containing tokenizer.json
		filepath.Join(modelName, "tokenizer.json"),
	)

	// PRIORITY 3: Check in the cache directory if configured
	if t.cfg != nil {
		cacheDir := getTokenizerCacheDir()
		possiblePaths = append(possiblePaths,
			filepath.Join(cacheDir, modelName, "tokenizer.json"),
			// HuggingFace cache directory structure
			filepath.Join(cacheDir, "models--"+modelName, "snapshots", "*", "tokenizer.json"),
		)
	}

	// Try each possible path
	for _, path := range possiblePaths {
		// Skip glob patterns for now, just check direct paths
		if filepath.Base(path) == "*" {
			continue
		}

		if _, err := os.Stat(path); err == nil {
			// File exists, try to load it
			tokenizer, err := tokenizers.FromFile(path)
			if err == nil {
				return tokenizer, nil
			}
		}
	}

	return nil, fmt.Errorf("tokenizer not found in local paths")
}

// Encode converts a string into token IDs.
func (t *CachedHFTokenizer) Encode(input, modelName string) ([]uint32, []tokenizers.Offset, error) {
	tokenizer, err := t.getTokenizer(modelName)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get tokenizer for model %q: %w", modelName, err)
	}

	encodeOptions := []tokenizers.EncodeOption{
		tokenizers.WithReturnTypeIDs(),
		tokenizers.WithReturnOffsets(),
	}

	resp := tokenizer.EncodeWithOptions(input, true, encodeOptions...)
	return resp.IDs, resp.Offsets, nil
}

// getTokenizerCacheDir returns the absolute path to the tokenizer cache directory relative to the project root.
func getTokenizerCacheDir() string {
	_, filename, _, _ := runtime.Caller(0) // this file
	base := filepath.Dir(filename)
	return filepath.Join(base, "..", "..", "bin")
}

// LoadModelPathsFromEnv reads model path mappings from environment variables.
// It looks for environment variables with the prefix "VLLM_MODEL_PATH_" followed by the served model name.
// For example: VLLM_MODEL_PATH_test_model=/models/my-model
// The served model name in the env var should use underscores, which will be converted to hyphens.
func LoadModelPathsFromEnv() map[string]string {
	const envPrefix = "VLLM_MODEL_PATH_"
	modelPaths := make(map[string]string)

	for _, env := range os.Environ() {
		// Check if the env var starts with our prefix
		if !strings.HasPrefix(env, envPrefix) {
			continue
		}

		// Split into key=value
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 {
			continue
		}

		// Extract the served model name from the env var name
		// VLLM_MODEL_PATH_test_model -> test_model
		envKey := parts[0]
		modelPath := parts[1]
		servedModelName := strings.TrimPrefix(envKey, envPrefix)

		// Convert underscores to hyphens for model name (common convention)
		servedModelName = strings.ReplaceAll(servedModelName, "_", "-")

		if servedModelName != "" && modelPath != "" {
			modelPaths[servedModelName] = modelPath
		}
	}

	return modelPaths
}

// SetModelPath adds or updates a model path mapping.
// This allows dynamically setting the path for a served model name.
func (t *CachedHFTokenizer) SetModelPath(servedModelName, modelPath string) {
	if t.modelPaths == nil {
		t.modelPaths = make(map[string]string)
	}
	t.modelPaths[servedModelName] = modelPath
}

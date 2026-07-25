// mock-llm：网关压测用的极简 LLM 上游 mock。
//
// 仅标准库、无第三方依赖，静态链接 + scratch 镜像，运行内存 ~5-10MB，
// 单请求 = 一个 net/http goroutine，吞吐可达数万 req/s，压测时不会成为网关瓶颈。
//
// 覆盖网关真实转发所需的 OpenAI 兼容子集：
//   POST /v1/chat/completions   非流式 + SSE 流式（stream=true）
//   POST /v1/embeddings         OpenAI 兼容
//   POST /v1/rerank             vLLM/Jina rerank 兼容（results[].index/relevance_score）
//   GET  /v1/models
//   GET  /healthz
//   GET  /metrics                Prometheus 文本格式计数
//
// 环境变量（全部可选）：
//   LISTEN_ADDR          监听地址（默认 :8080）
//   MODEL_NAME           对外模型名（默认 mock-llm）
//   LATENCY_MS           固定响应延迟（毫秒，默认 0）
//   LATENCY_JITTER_MS    延迟抖动区间（毫秒，默认 0，LATENCY_MS+[0,jitter)）
//   STREAM_CHUNK_MS      流式每块间隔（毫秒，默认 0）
//   ERROR_RATE           注入错误概率 0.0~1.0（默认 0）
//   ERROR_STATUS         注入错误状态码（默认 500）
//   COMPLETION_TOKENS    mock 回复长度（近似 token 数，默认 32）
//   EMBEDDING_DIM        嵌入向量维度（默认 1024）
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// ---------- 配置 ----------

var (
	modelName        = "mock-llm"
	latency          time.Duration
	latencyJitter    time.Duration
	streamChunkDelay time.Duration
	errorRate        float64
	errorStatus      int
	completionTokens int
	embeddingDim     int
	startedAt        = time.Now()
)

func envInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(name string, def float64) float64 {
	if v := os.Getenv(name); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func loadConfig() {
	if m := os.Getenv("MODEL_NAME"); m != "" {
		modelName = m
	}
	latency = time.Duration(envInt("LATENCY_MS", 0)) * time.Millisecond
	latencyJitter = time.Duration(envInt("LATENCY_JITTER_MS", 0)) * time.Millisecond
	streamChunkDelay = time.Duration(envInt("STREAM_CHUNK_MS", 0)) * time.Millisecond
	errorRate = envFloat("ERROR_RATE", 0.0)
	errorStatus = envInt("ERROR_STATUS", 500)
	completionTokens = envInt("COMPLETION_TOKENS", 32)
	embeddingDim = envInt("EMBEDDING_DIM", 1024)
}

// ---------- 计数（原子，manding /metrics 直出） ----------

var (
	cRequests = atomic.Uint64{}
	cBytes    = atomic.Uint64{}
	cErrors   = atomic.Uint64{}
)

// ---------- 通用工具 ----------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	cErrors.Add(1)
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": "mock_http_error"},
	})
}

func shouldInjectError() bool {
	return errorRate > 0 && rand.Float64() < errorRate
}

// applyLatency 在响应前统一注入延迟（LATENCY_MS + [0, LATENCY_JITTER_MS)）。
func applyLatency() {
	d := latency
	if latencyJitter > 0 {
		d += time.Duration(rand.Int64N(int64(latencyJitter)))
	}
	if d > 0 {
		time.Sleep(d)
	}
}

func readBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	return io.ReadAll(r.Body)
}

func reqID() string {
	return fmt.Sprintf("mock-%d-%d", time.Now().UnixMilli(), rand.Uint64()%1e6)
}

// lastUserText 取最后一条 role=user 且 content 为字符串的消息原文。
func lastUserText(messages []any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		m, ok := messages[i].(map[string]any)
		if !ok {
			continue
		}
		if m["role"] == "user" {
			if s, ok := m["content"].(string); ok {
				return s
			}
		}
	}
	return ""
}

// estTokens 粗略估算 token 数（中英文混排 ≈ 每 2 字符 1 token）。
func estTokens(s string) int {
	n := utf8.RuneCountInString(s)
	if n <= 0 {
		return 1
	}
	return n/2 + 1
}

// echoReply 以用户文本回显生成 mock 回复，长度受 COMPLETION_TOKENS 约束。
func echoReply(text string) string {
	if strings.TrimSpace(text) == "" {
		text = "这是一条 mock 回复。"
	}
	runes := []rune(text)
	max := completionTokens * 4 // 每 token 约 2 字符，留余量
	if len(runes) > max {
		runes = runes[:max]
	}
	return "mock[" + modelName + "]: " + string(runes)
}

// fakeVector 生成确定性伪随机嵌入向量（同一 seed 不同维度值稳定可复现）。
func fakeVector(seed int) []float64 {
	v := make([]float64, embeddingDim)
	// 固定首位分量按 seed 变化，便于区分批次/观测
	v[0] = float64(seed%200) / 200.0 - 0.5
	for i := 1; i < len(v); i++ {
		// 确定性：由 i+seed 派生，避免每次请求不同
		x := float64(((seed+1)*(i+3))%1000)/1000.0 - 0.5
		v[i] = float64(int(x*1000)) / 1000.0
	}
	return v
}

// ---------- 处理器 ----------

func handleChat(w http.ResponseWriter, r *http.Request) {
	cRequests.Add(1)
	body, err := readBody(w, r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	cBytes.Add(uint64(len(body)))

	var req struct {
		Model    string `json:"model"`
		Messages []any  `json:"messages"`
		Stream   bool   `json:"stream"`
		N        int    `json:"n"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if req.Model == "" {
		req.Model = modelName
	}
	if req.N < 1 {
		req.N = 1
	}
	if shouldInjectError() {
		writeErr(w, errorStatus, fmt.Sprintf("injected error status %d", errorStatus))
		return
	}

	userText := lastUserText(req.Messages)
	reply := echoReply(userText)
	usage := map[string]any{
		"prompt_tokens":     estTokens(lastUserText(req.Messages)),
		"completion_tokens": estTokens(reply),
		"total_tokens":      estTokens(lastUserText(req.Messages)) + estTokens(reply),
	}
	id := reqID()

	if req.Stream {
		applyLatency()
		serveSSE(w, req.Model, id, reply, usage)
		return
	}

	applyLatency()
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   req.Model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": reply},
			"finish_reason": "stop",
		}},
		"usage": usage,
	})
}

// serveSSE 把整条回复按小块逐条 data: 帧推送，末尾含 stop chunk + [DONE]。
func serveSSE(w http.ResponseWriter, model, id, reply string, usage map[string]any) {
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	runes := []rune(reply)
	const chunkSize = 16
	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		obj := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []any{map[string]any{
				"index":         0,
				"delta":         map[string]any{"content": string(runes[i:end])},
				"finish_reason": nil,
			}},
		}
		buf, _ := json.Marshal(obj)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", buf); err != nil {
			return
		}
		fl.Flush()
		if streamChunkDelay > 0 {
			time.Sleep(streamChunkDelay)
		}
	}
	// 终止 chunk（OpenAI 约定：delta 为空 + finish_reason=stop）
	stopObj := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": "stop",
		}},
	}
	if buf, err := json.Marshal(stopObj); err == nil {
		_, _ = fmt.Fprintf(w, "data: %s\n\n", buf)
		fl.Flush()
	}
	_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
	fl.Flush()
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	cRequests.Add(1)
	body, err := readBody(w, r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	cBytes.Add(uint64(len(body)))

	var req struct {
		Model string `json:"model"`
		Input any    `json:"input"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if req.Model == "" {
		req.Model = modelName
	}
	if shouldInjectError() {
		writeErr(w, errorStatus, fmt.Sprintf("injected error status %d", errorStatus))
		return
	}

	var texts []string
	switch v := req.Input.(type) {
	case string:
		texts = []string{v}
	case []any:
		for _, it := range v {
			if s, ok := it.(string); ok {
				texts = append(texts, s)
			}
		}
	}
	if len(texts) == 0 {
		writeErr(w, http.StatusBadRequest, "input required: string or []string")
		return
	}

	applyLatency()
	data := make([]map[string]any, 0, len(texts))
	total := 0
	for i, t := range texts {
		data = append(data, map[string]any{
			"object":    "embedding",
			"index":     i,
			"embedding": fakeVector(i),
		})
		total += estTokens(t)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"model":  req.Model,
		"usage":  map[string]any{"prompt_tokens": total, "total_tokens": total},
	})
}

func handleRerank(w http.ResponseWriter, r *http.Request) {
	cRequests.Add(1)
	body, err := readBody(w, r)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}
	cBytes.Add(uint64(len(body)))

	var req struct {
		Model     string `json:"model"`
		Query     string `json:"query"`
		Documents []any  `json:"documents"`
		TopN      int    `json:"top_n"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if req.Model == "" {
		req.Model = modelName
	}
	if shouldInjectError() {
		writeErr(w, errorStatus, fmt.Sprintf("injected error status %d", errorStatus))
		return
	}

	docs := make([]string, 0, len(req.Documents))
	for _, it := range req.Documents {
		switch v := it.(type) {
		case string:
			docs = append(docs, v)
		case map[string]any:
			if s, ok := v["text"].(string); ok {
				docs = append(docs, s)
			}
		}
	}
	if len(docs) == 0 {
		writeErr(w, http.StatusBadRequest, "documents required")
		return
	}
	if req.TopN <= 0 || req.TopN > len(docs) {
		req.TopN = len(docs)
	}

	applyLatency()
	results := make([]map[string]any, 0, req.TopN)
	for i := 0; i < req.TopN; i++ {
		score := 0.95 - float64(i)*0.07
		if score < 0.10 {
			score = 0.10
		}
		results = append(results, map[string]any{
			"index":           i,
			"relevance_score": score,
			"document":        docs[i],
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "model": req.Model})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []any{map[string]any{
			"id":       modelName,
			"object":   "model",
			"created":  startedAt.Unix(),
			"owned_by": "mock-llm",
		}},
	})
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func handleMetrics(w http.ResponseWriter, r *http.Request) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# HELP mockllm_requests_total Total HTTP requests served\n")
	fmt.Fprintf(w, "# TYPE mockllm_requests_total counter\n")
	fmt.Fprintf(w, "mockllm_requests_total %d\n", cRequests.Load())
	fmt.Fprintf(w, "# HELP mockllm_errors_total Total injected/HTTP errors\n")
	fmt.Fprintf(w, "# TYPE mockllm_errors_total counter\n")
	fmt.Fprintf(w, "mockllm_errors_total %d\n", cErrors.Load())
	fmt.Fprintf(w, "# HELP mockllm_ingress_bytes_total Total request body bytes\n")
	fmt.Fprintf(w, "# TYPE mockllm_ingress_bytes_total counter\n")
	fmt.Fprintf(w, "mockllm_ingress_bytes_total %d\n", cBytes.Load())
	fmt.Fprintf(w, "# HELP mockllm_uptime_seconds Uptime\n")
	fmt.Fprintf(w, "# TYPE mockllm_uptime_seconds gauge\n")
	fmt.Fprintf(w, "mockllm_uptime_seconds %d\n", int(time.Since(startedAt).Seconds()))
	fmt.Fprintf(w, "# HELP mockllm_goroutines Current goroutines\n")
	fmt.Fprintf(w, "# TYPE mockllm_goroutines gauge\n")
	fmt.Fprintf(w, "mockllm_goroutines %d\n", runtime.NumGoroutine())
	fmt.Fprintf(w, "# HELP mockllm_alloc_bytes Current heap alloc\n")
	fmt.Fprintf(w, "# TYPE mockllm_alloc_bytes gauge\n")
	fmt.Fprintf(w, "mockllm_alloc_bytes %d\n", ms.Alloc)
}

func main() {
	loadConfig()
	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", handleChat)
	mux.HandleFunc("POST /v1/embeddings", handleEmbeddings)
	mux.HandleFunc("POST /v1/rerank", handleRerank)
	mux.HandleFunc("GET /v1/models", handleModels)
	mux.HandleFunc("GET /healthz", handleHealthz)
	mux.HandleFunc("GET /metrics", handleMetrics)

	log.Printf(
		"mock-llm listening on %s model=%s error_rate=%.2f error_status=%d latency_ms=%d jitter_ms=%d chunk_ms=%d dim=%d",
		addr, modelName, errorRate, errorStatus, latency.Milliseconds(),
		latencyJitter.Milliseconds(), streamChunkDelay.Milliseconds(), embeddingDim,
	)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
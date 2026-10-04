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

// firstSystemText 取第一条 role=system 且 content 为字符串的消息原文。
func firstSystemText(messages []any) string {
	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if msg["role"] == "system" {
			if s, ok := msg["content"].(string); ok {
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

// intentReply 根据用户查询内容生成对应业务场景的 mock 回复
func intentReply(text string, systemPrompt string) string {
	lower := strings.ToLower(text)
	
	// 订单查询相关
	if strings.Contains(lower, "订单") || strings.Contains(lower, "order") {
		return `mock[mock-llm]: 根据您的查询，为您查询到以下订单信息：
订单号: WB202409010077
状态: 已发货
下单时间: 2024-09-01 14:30
商品: 无线蓝牙耳机 Pro × 1
金额: ¥299.00
物流单号: SF5555666677
预计送达: 2024-09-14`
	}
	
	// 物流查询相关
	if strings.Contains(lower, "物流") || strings.Contains(lower, "快递") || strings.Contains(lower, "到哪了") || strings.Contains(lower, "运单") {
		return `mock[mock-llm]: 物流跟踪信息 (SF5555666677)：
[2024-09-12 08:00] 已揽收 - 上海转运中心
[2024-09-12 14:30] 运输中 - 离开上海华东中转场
[2024-09-13 06:15] 到达 - 目的地城市分拨中心
[2024-09-13 09:30] 派送中 - 快递员正在派送
预计今日 18:00 前送达`
	}
	
	// 余额查询相关
	if strings.Contains(lower, "余额") || strings.Contains(lower, "钱包") || strings.Contains(lower, "积分") || strings.Contains(lower, "多少钱") {
		return `mock[mock-llm]: 账户余额查询结果：
可用余额: ¥1,258.50
冻结金额: ¥0.00
可用积分: 3,420 分
会员等级: 黄金会员`
	}
	
	// 优惠券查询相关
	if strings.Contains(lower, "优惠券") || strings.Contains(lower, "代金券") || strings.Contains(lower, "满减") || strings.Contains(lower, "优惠码") {
		return `mock[mock-llm]: 您可用的优惠券列表：
1. 满减券 ¥50 - 满 300 可用，有效期至 2024-10-31
2. 新用户专享 ¥20 - 无门槛，有效期至 2024-09-30
3. 品类券 85折 - 耳机品类专用，有效期至 2024-12-31
4. 生日专属 ¥100 - 满 500 可用，有效期至 2024-09-15`
	}
	
	// 退货退款相关
	if strings.Contains(lower, "退货") || strings.Contains(lower, "退款") || strings.Contains(lower, "不想要") {
		return `mock[mock-llm]: 已为您提交退货退款申请：
退货单号: TH202409120045
订单号: WB202409010077
退货商品: 无线蓝牙耳机 Pro
退款金额: ¥299.00
退款方式: 原路返回（预计 1-3 个工作日到账）
请在 7 天内寄出商品，快递单号请在退货详情页填写`
	}
	
	// 电商商品推荐场景：检测 system prompt 中的商品信息
	if strings.Contains(systemPrompt, "shop-agent电商平台") || strings.Contains(systemPrompt, "商品关系") || strings.Contains(systemPrompt, "product_relations") {
		return generateProductRecommendation(text, systemPrompt)
	}
	
	// 默认回显
	return echoReply(text)
}

// generateProductRecommendation 根据商品上下文生成推荐回复
func generateProductRecommendation(userQuery, systemPrompt string) string {
	// 提取商品信息
	productName := extractBetween(systemPrompt, "商品名称: ", "\n")
	brand := extractBetween(systemPrompt, "品牌: ", "\n")
	price := extractBetween(systemPrompt, "价格: ", "\n")
	description := extractBetween(systemPrompt, "描述: ", "\n")
	
	// 提取商品关系
	graphContext := extractBetween(systemPrompt, "<product_relations>", "</product_relations>")
	
	var recommendation strings.Builder
	recommendation.WriteString("mock[mock-llm]: ")
	
	if productName != "" {
		recommendation.WriteString("为您推荐: ")
		recommendation.WriteString(productName)
		if brand != "" {
			recommendation.WriteString(" (")
			recommendation.WriteString(brand)
			recommendation.WriteString(")")
		}
		recommendation.WriteString("\n")
		if price != "" {
			recommendation.WriteString("价格: ")
			recommendation.WriteString(price)
			recommendation.WriteString("\n")
		}
		if description != "" {
			recommendation.WriteString("特点: ")
			recommendation.WriteString(description)
			recommendation.WriteString("\n")
		}
	}
	
	// 基于商品关系生成搭配推荐
	if strings.Contains(graphContext, "兼容配件") {
		recommendation.WriteString("\n推荐搭配配件: ")
		accessories := extractBetween(graphContext, "兼容配件：", "\n")
		if accessories != "" {
			recommendation.WriteString(accessories)
		} else {
			recommendation.WriteString("蓝牙耳机充电仓、耳机收纳包")
		}
		recommendation.WriteString("\n")
	}
	
	if strings.Contains(graphContext, "同品牌") {
		recommendation.WriteString("同品牌推荐: ")
		sameBrand := extractBetween(graphContext, "同品牌的其他商品：", "\n")
		if sameBrand != "" {
			recommendation.WriteString(sameBrand)
		} else {
			recommendation.WriteString("有线耳机基础款、运动蓝牙耳机")
		}
		recommendation.WriteString("\n")
	}
	
	if strings.Contains(graphContext, "替代品") || strings.Contains(graphContext, "竞品") {
		recommendation.WriteString("替代方案: ")
		alternatives := extractBetween(graphContext, "替代品/竞品：", "\n")
		if alternatives != "" {
			recommendation.WriteString(alternatives)
		} else {
			recommendation.WriteString("某品牌降噪耳机、另一品牌运动耳机")
		}
	}
	
	return recommendation.String()
}

// extractBetween 提取两个标记之间的内容
func extractBetween(text, start, end string) string {
	startIdx := strings.Index(text, start)
	if startIdx == -1 {
		return ""
	}
	startIdx += len(start)
	endIdx := strings.Index(text[startIdx:], end)
	if endIdx == -1 {
		return strings.TrimSpace(text[startIdx:])
	}
	return strings.TrimSpace(text[startIdx : startIdx+endIdx])
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
	systemPrompt := firstSystemText(req.Messages)
	reply := intentReply(userText, systemPrompt)
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
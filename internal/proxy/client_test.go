package proxy

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/liguangsheng/wildtoken/internal/models"
)

func int32Value(t *testing.T, value *int32) any {
	t.Helper()
	if value == nil {
		return nil
	}
	return *value
}

func TestStreamingChatRequestIncludesUsageAndPreservesOptions(t *testing.T) {
	body := []byte(`{"model":"requested-model","stream":true,"stream_options":{"include_obfuscation":true}}`)

	prepared := PrepareUpstreamBody(body, ptrTo("upstream-model"), "chat/completions", nil)

	var decoded map[string]any
	if err := json.Unmarshal(prepared, &decoded); err != nil {
		t.Fatalf("decode prepared body: %v", err)
	}
	if decoded["model"] != "upstream-model" {
		t.Errorf("model = %v, want the forward model", decoded["model"])
	}
	options, ok := decoded["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options missing: %s", prepared)
	}
	if options["include_usage"] != true {
		t.Error("include_usage was not requested")
	}
	if options["include_obfuscation"] != true {
		t.Error("an existing stream option was dropped")
	}
}

func TestUsageOptionIsNotAddedToOtherOrNonStreamingRequests(t *testing.T) {
	for _, testCase := range []struct{ path, body string }{
		{"chat/completions", `{"model":"m","stream":false}`},
		{"responses", `{"model":"m","stream":true}`},
	} {
		prepared := PrepareUpstreamBody([]byte(testCase.body), nil, testCase.path, nil)
		var decoded map[string]any
		if err := json.Unmarshal(prepared, &decoded); err != nil {
			t.Fatalf("decode %s: %v", testCase.path, err)
		}
		if _, present := decoded["stream_options"]; present {
			t.Errorf("%s gained stream_options: %s", testCase.path, prepared)
		}
	}
}

func TestExtractsReasoningEffortFromOpenAIAndAnthropicRequests(t *testing.T) {
	for _, testCase := range []struct {
		body string
		want any
	}{
		{`{"reasoning_effort":"high"}`, "high"},
		{`{"reasoning":{"effort":"medium"}}`, "medium"},
		{`{"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"}}`, "xhigh"},
		{`{"thinking":{"type":"disabled"},"output_config":{"effort":"  high  "}}`, "high"},
		{`{"output_config":{"effort":"  "}}`, nil},
	} {
		effort := ExtractReasoningEffort([]byte(testCase.body))
		var got any
		if effort != nil {
			got = *effort
		}
		if got != testCase.want {
			t.Errorf("body %s gave %v, want %v", testCase.body, got, testCase.want)
		}
	}
}

func TestOpenAIReasoningEffortTakesPrecedenceOverAnthropicOutputConfig(t *testing.T) {
	body := []byte(`{"reasoning_effort":"low","reasoning":{"effort":"medium"},"output_config":{"effort":"high"}}`)
	effort := ExtractReasoningEffort(body)
	if effort == nil || *effort != "low" {
		t.Errorf("effort = %v, want low", effort)
	}
}

func TestBackfillsReasoningContentWhenThinkingModeActive(t *testing.T) {
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"","reasoning_content":"thinking","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},` +
		`{"role":"tool","content":"ok","tool_call_id":"c1"},` +
		`{"role":"assistant","content":"","tool_calls":[{"id":"c2","type":"function","function":{"name":"f","arguments":"{}"}}]},` +
		`{"role":"tool","content":"ok","tool_call_id":"c2"}` +
		`]}`)

	prepared := PrepareUpstreamBody(body, nil, "chat/completions", nil)

	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(prepared, &decoded); err != nil {
		t.Fatalf("decode prepared body: %v", err)
	}
	for i, message := range decoded.Messages {
		if messageRole(message) != "assistant" {
			continue
		}
		if _, ok := message["reasoning_content"]; !ok {
			t.Errorf("assistant message %d missing reasoning_content: %s", i, prepared)
		}
	}
	// The turn without its own reasoning gets an empty string, not the prior value.
	if string(decoded.Messages[3]["reasoning_content"]) != `""` {
		t.Errorf("backfilled reasoning_content = %s, want empty string",
			decoded.Messages[3]["reasoning_content"])
	}
}

func TestLeavesReasoningContentAloneWhenNoThinkingMode(t *testing.T) {
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},` +
		`{"role":"tool","content":"ok","tool_call_id":"c1"}` +
		`]}`)

	prepared := PrepareUpstreamBody(body, nil, "chat/completions", nil)

	if strings.Contains(string(prepared), "reasoning_content") {
		t.Errorf("reasoning_content added without thinking mode: %s", prepared)
	}
}

func TestStringifiesNonStringToolCallArguments(t *testing.T) {
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"calc","arguments":{"x":1}}}]},` +
		`{"role":"tool","content":"1","tool_call_id":"c1"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c2","type":"function","function":{"name":"ping","arguments":null}}]},` +
		`{"role":"tool","content":"pong","tool_call_id":"c2"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c3","type":"function","function":{"name":"blank"}}]},` +
		`{"role":"tool","content":"ok","tool_call_id":"c3"},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c4","type":"function","function":{"name":"kept","arguments":"{\"y\":2}"}}]}` +
		`]}`)

	prepared := PrepareUpstreamBody(body, nil, "chat/completions", nil)

	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(prepared, &decoded); err != nil {
		t.Fatalf("decode prepared body: %v", err)
	}
	var args []string
	for _, message := range decoded.Messages {
		if messageRole(message) != "assistant" {
			continue
		}
		var calls []struct {
			Function struct {
				Arguments string `json:"arguments"`
			} `json:"function"`
		}
		if err := json.Unmarshal(message["tool_calls"], &calls); err != nil {
			t.Fatalf("decode tool_calls: %v", err)
		}
		for _, call := range calls {
			args = append(args, call.Function.Arguments)
		}
	}
	want := `{"x":1}|{}|{}|{"y":2}`
	if got := strings.Join(args, "|"); got != want {
		t.Errorf("arguments = %q, want %q (prepared: %s)", got, want, prepared)
	}
}

func TestFlattensTextOnlyContentArrays(t *testing.T) {
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]},` +
		`{"role":"user","content":[{"type":"text"}]}` +
		`]}`)

	prepared := PrepareUpstreamBody(body, nil, "chat/completions", nil)

	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(prepared, &decoded); err != nil {
		t.Fatalf("decode prepared body: %v", err)
	}
	var flat string
	if err := json.Unmarshal(decoded.Messages[0]["content"], &flat); err != nil {
		t.Fatalf("text-only content should flatten to a string: %v", err)
	}
	if flat != "hello\nworld" {
		t.Errorf("flattened content = %q, want %q", flat, "hello\nworld")
	}
	if _, ok := decoded.Messages[1]["content"]; ok {
		t.Errorf("empty text-only content not dropped: %s", prepared)
	}
}

func TestPreservesTextPartsWithExtraFields(t *testing.T) {
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}` +
		`]}`)

	prepared := PrepareUpstreamBody(body, nil, "v1/messages", nil)

	if !strings.Contains(string(prepared), "cache_control") {
		t.Errorf("cache_control dropped by flattening: %s", prepared)
	}
}

func assertUsage(t *testing.T, usage TokenUsage, want map[string]any) {
	t.Helper()
	got := map[string]any{
		"prompt":            int32Value(t, usage.PromptTokens),
		"completion":        int32Value(t, usage.CompletionTokens),
		"total":             int32Value(t, usage.TotalTokens),
		"prompt_cached":     int32Value(t, usage.PromptCachedTokens),
		"cache_creation":    int32Value(t, usage.CacheCreationTokens),
		"completion_reason": int32Value(t, usage.CompletionReasoningTokens),
	}
	for key, expected := range want {
		if got[key] != expected {
			t.Errorf("%s = %v, want %v", key, got[key], expected)
		}
	}
}

func TestExtractsUsageFromCodexResponsesCompletionEvent(t *testing.T) {
	response := []byte(`data: {"type":"response.completed","response":{"usage":{"input_tokens":99424,"output_tokens":440,"total_tokens":99864,"input_tokens_details":{"cached_tokens":12000},"output_tokens_details":{"reasoning_tokens":128}}}}

`)

	// OpenAI Responses and Codex report input/output as already complete, with
	// the cache and reasoning fields as subsets.
	assertUsage(t, ExtractUsage(response, "text/event-stream"), map[string]any{
		"prompt": int32(99424), "completion": int32(440), "total": int32(99864),
		"prompt_cached": int32(12000), "cache_creation": nil,
		"completion_reason": int32(128),
	})
}

func TestOpenAIChatCompletionsUsageKeepsDetailsAsSubsets(t *testing.T) {
	// Official OpenAI Chat Completions:
	//   prompt_tokens includes cached_tokens
	//   completion_tokens includes reasoning_tokens
	//   total_tokens = prompt + completion, authoritative when present
	// Details must never be added into the main columns.
	body := []byte(`{"usage":{"prompt_tokens":2006,"completion_tokens":300,"total_tokens":2306,
        "prompt_tokens_details":{"cached_tokens":1920,"cache_write_tokens":80},
        "completion_tokens_details":{"reasoning_tokens":128}}}`)

	assertUsage(t, ExtractUsage(body, "application/json"), map[string]any{
		"prompt": int32(2006), "completion": int32(300), "total": int32(2306),
		"prompt_cached": int32(1920), "cache_creation": int32(80),
		"completion_reason": int32(128),
	})
}

func TestOpenAIResponsesUsageDoesNotDoubleCountCachedOrReasoning(t *testing.T) {
	body := []byte(`{"usage":{"input_tokens":75,"input_tokens_details":{"cached_tokens":0},
        "output_tokens":1186,"output_tokens_details":{"reasoning_tokens":1024},"total_tokens":1261}}`)

	assertUsage(t, ExtractUsage(body, "application/json"), map[string]any{
		"prompt": int32(75), "completion": int32(1186), "total": int32(1261),
		"prompt_cached": int32(0), "completion_reason": int32(1024),
	})
}

func TestAggregatesAnthropicStyleInputWithoutAddingThinking(t *testing.T) {
	// Official Anthropic:
	//   input  = input_tokens + cache_creation + cache_read
	//   output = output_tokens, with thinking already included
	//   total  = input + output
	// A top-level thinking_tokens, if a proxy emits one, is detail only.
	body := []byte(`{"usage":{"input_tokens":100,"output_tokens":40,
        "cache_creation_input_tokens":20,"cache_read_input_tokens":30,"thinking_tokens":15}}`)

	assertUsage(t, ExtractUsage(body, "application/json"), map[string]any{
		"prompt": int32(150), "completion": int32(40), "total": int32(190),
		"prompt_cached": int32(30), "cache_creation": int32(20),
		"completion_reason": int32(15),
	})
}

func TestNestedThinkingTokensAreBreakdownNotAddedTwice(t *testing.T) {
	// output_tokens_details.thinking_tokens is a subset of output_tokens.
	body := []byte(`{"usage":{"input_tokens":100,"output_tokens":503,
        "cache_creation_input_tokens":50,"cache_read_input_tokens":25,
        "output_tokens_details":{"thinking_tokens":312}}}`)

	assertUsage(t, ExtractUsage(body, "application/json"), map[string]any{
		"prompt": int32(175), "completion": int32(503), "total": int32(678),
		"completion_reason": int32(312),
	})
}

func TestRecognizesSSEProtocolTerminalEvents(t *testing.T) {
	for _, line := range []string{
		"data: [DONE]",
		"event: response.completed",
		`data: {"type":"response.failed"}`,
		"event: message_stop",
		"event: error",
	} {
		if !sseBytesLineIsTerminal([]byte(line)) {
			t.Errorf("%q was not recognized as terminal", line)
		}
	}

	if sseBytesLineIsTerminal([]byte("event: response.output_item.done")) {
		t.Error("a non-terminal event was treated as terminal")
	}
}

func TestBuildUpstreamURLAvoidsDuplicatingTheVersionSegment(t *testing.T) {
	for _, testCase := range []struct{ base, path, query, want string }{
		{"https://api.example.com", "responses", "", "https://api.example.com/v1/responses"},
		{"https://api.example.com/", "/responses", "", "https://api.example.com/v1/responses"},
		{"https://api.example.com/v1", "responses", "", "https://api.example.com/v1/responses"},
		{"https://api.example.com/v1/", "responses", "", "https://api.example.com/v1/responses"},
		{"https://api.example.com", "models", "limit=5", "https://api.example.com/v1/models?limit=5"},
	} {
		upstream := models.UpstreamRow{BaseURL: testCase.base}
		if got := BuildUpstreamURL(&upstream, testCase.path, testCase.query); got != testCase.want {
			t.Errorf("base %q path %q gave %q, want %q",
				testCase.base, testCase.path, got, testCase.want)
		}
	}
}

func upstreamWithHeaders(t *testing.T, baseURL, extraHeaders string) models.UpstreamRow {
	t.Helper()
	key := "upstream-secret"
	return models.UpstreamRow{
		Name:         "channel",
		BaseURL:      baseURL,
		APIKey:       &key,
		ExtraHeaders: extraHeaders,
	}
}

func TestAnthropicMessagesUsesUpstreamAPIKeyAndHidesTheDownstreamKey(t *testing.T) {
	downstream := http.Header{}
	downstream.Set("x-api-key", "downstream-secret")
	upstream := upstreamWithHeaders(t, "https://api.anthropic.com", "{}")

	headers, err := BuildForwardHeaders(downstream, &upstream, "messages")
	if err != nil {
		t.Fatalf("build headers: %v", err)
	}
	if headers["x-api-key"] != "upstream-secret" {
		t.Errorf("x-api-key = %q, want the channel key", headers["x-api-key"])
	}
	if headers["anthropic-version"] != "2023-06-01" {
		t.Errorf("anthropic-version = %q, want the default", headers["anthropic-version"])
	}
	if _, present := headers["authorization"]; present {
		t.Error("an Anthropic request carried an authorization header")
	}
}

func TestChannelHeadersOverrideDownstreamAndGeneratedCredentialsCaseInsensitively(t *testing.T) {
	downstream := http.Header{}
	downstream.Set("user-agent", "downstream-agent")
	downstream.Set("x-request-id", "request-123")
	downstream.Set("authorization", "downstream-secret")

	upstream := upstreamWithHeaders(t, "https://example.test", `{
        "UsEr-AgEnT": "channel-agent",
        "AUTHORIZATION": "Token channel-credential",
        "X-Trace-Id": "channel-trace",
        "X-Upstream-Request": "{client_header:X-Request-Id}",
        "X-Missing": "{client_header:X-Not-Present}"
    }`)

	headers, err := BuildForwardHeaders(downstream, &upstream, "responses")
	if err != nil {
		t.Fatalf("build headers: %v", err)
	}

	for name, want := range map[string]string{
		"user-agent":         "channel-agent",
		"authorization":      "Token channel-credential",
		"x-trace-id":         "channel-trace",
		"x-upstream-request": "request-123",
	} {
		if headers[name] != want {
			t.Errorf("%s = %q, want %q", name, headers[name], want)
		}
	}
	// A placeholder naming an absent header contributes nothing.
	if _, present := headers["x-missing"]; present {
		t.Error("an unresolved placeholder produced a header")
	}

	// Names are normalized, so no case-duplicate key can reach the upstream.
	authorizationKeys := 0
	for name := range headers {
		if strings.EqualFold(name, "authorization") {
			authorizationKeys++
		}
	}
	if authorizationKeys != 1 {
		t.Errorf("found %d authorization keys, want exactly 1", authorizationKeys)
	}
}

func TestHeaderOverrideValidationRejectsAmbiguousOrTransportHeaders(t *testing.T) {
	duplicate := map[string]string{"Authorization": "one", "authorization": "two"}
	err := ValidateHeaderOverrides(duplicate)
	if err == nil || !strings.Contains(err.Error(), "duplicate Header") {
		t.Errorf("duplicate names gave %v, want a duplicate-header error", err)
	}

	for _, overrides := range []map[string]string{
		{"Host": "example.test"},
		{"Connection": "keep-alive"},
		{"X-Test": "one\r\ntwo"},
		{"X-Test": "prefix-{client_header:X-Request-Id}"},
		{"X-Test": "{client_header:Authorization}"},
	} {
		if err := ValidateHeaderOverrides(overrides); err == nil {
			t.Errorf("overrides %v were accepted", overrides)
		}
	}

	// A well-formed override is still accepted.
	if err := ValidateHeaderOverrides(map[string]string{
		"X-Trace": "value", "X-Copy": "{client_header:X-Request-Id}",
	}); err != nil {
		t.Errorf("a valid override was rejected: %v", err)
	}
}

func TestConnectionNominatedHeadersAreNotForwardedOrReintroduced(t *testing.T) {
	downstream := http.Header{}
	downstream.Set("connection", "x-hop, keep-alive")
	downstream.Set("x-hop", "downstream-value")

	upstream := upstreamWithHeaders(t, "https://example.test", `{
        "X-Hop": "channel-value",
        "X-Remapped-Hop": "{client_header:X-Hop}",
        "X-End-To-End": "kept"
    }`)

	headers, err := BuildForwardHeaders(downstream, &upstream, "responses")
	if err != nil {
		t.Fatalf("build headers: %v", err)
	}

	for _, name := range []string{"connection", "x-hop", "x-remapped-hop"} {
		if _, present := headers[name]; present {
			t.Errorf("a connection-nominated header survived: %s", name)
		}
	}
	if headers["x-end-to-end"] != "kept" {
		t.Errorf("an end-to-end header was dropped: %v", headers)
	}
}

func TestResponseCaptureRetainsOnlyTheConfiguredPrefix(t *testing.T) {
	capture := newResponseCapture(5)
	capture.push([]byte("abc"))
	capture.push([]byte("defgh"))

	if string(capture.bytes) != "abcde" {
		t.Errorf("captured %q, want the first 5 bytes", capture.bytes)
	}
	if capture.byteLength != 8 {
		t.Errorf("byte length = %d, want the full 8", capture.byteLength)
	}
}

func TestObservationExtractsTerminalMetadataAfterTheSnapshotLimit(t *testing.T) {
	capture := newResponseCapture(8)
	observation := &sseObservation{}
	measure := func() int32 { return 1 }

	first := []byte(`data: {"type":"response.output_text.delta","delta":"hello"}` + "\n\n")
	terminal := []byte(`data: {"type":"response.completed","response":{"reasoning":{"effort":"high"},` +
		`"usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18,` +
		`"input_tokens_details":{"cached_tokens":3},"cache_creation_input_tokens":5,` +
		`"output_tokens_details":{"reasoning_tokens":2}}}}` + "\n\n")

	response := append(append([]byte{}, first...), terminal...)
	for start := 0; start < len(response); start += 3 {
		chunk := response[start:min(start+3, len(response))]
		capture.push(chunk)
		observation.observeChunk(chunk, measure)
	}

	// Metadata is still read from events that fall past the snapshot limit.
	if len(capture.bytes) != 8 {
		t.Errorf("captured %d bytes, want the 8-byte limit", len(capture.bytes))
	}
	if capture.byteLength != len(first)+len(terminal) {
		t.Errorf("byte length = %d, want the full stream length", capture.byteLength)
	}
	assertUsage(t, observation.usage, map[string]any{
		"prompt": int32(11), "completion": int32(7), "total": int32(18),
		"prompt_cached": int32(3), "cache_creation": int32(5),
		"completion_reason": int32(2),
	})
	if observation.responseReasoningEffort == nil || *observation.responseReasoningEffort != "high" {
		t.Errorf("effort = %v, want high", observation.responseReasoningEffort)
	}
	if observation.firstTokenMs == nil {
		t.Error("no first token was observed")
	}
	if !observation.terminalEventSeen {
		t.Error("the terminal event was not observed")
	}
}

func TestObservationDiscardsAndRecoversFromAnOversizedEventLine(t *testing.T) {
	observation := &sseObservation{}
	measure := func() int32 { return 1 }

	oversized := make([]byte, maxSSEEventBytes+1)
	for i := range oversized {
		oversized[i] = 'x'
	}
	observation.observeChunk(oversized, measure)
	if !observation.lineOverflow {
		t.Error("an oversized line did not trip the overflow guard")
	}
	if len(observation.lineBuf) != 0 {
		t.Error("an oversized line was buffered")
	}

	// The next newline ends the discarded line, and observation resumes.
	observation.observeChunk([]byte("\ndata: [DONE]\n\n"), measure)
	if observation.lineOverflow {
		t.Error("the overflow guard did not reset")
	}
	if len(observation.lineBuf) != 0 {
		t.Error("the line buffer was not cleared")
	}
	if !observation.terminalEventSeen {
		t.Error("the stream did not recover to observe the terminal event")
	}
}

func TestBufferedSSEBodiesAreWalkedWithoutCopyingThem(t *testing.T) {
	usageEvent := `data: {"type":"response.completed","response":{"usage":` +
		`{"input_tokens":11,"output_tokens":7,"total_tokens":18}}}`
	tokenEvent := `data: {"type":"response.output_text.delta","delta":"hi"}`

	// The body is walked in place now, so the line boundaries themselves are
	// what a rewrite could get wrong: a trailing newline, none at all, and the
	// CRLF an upstream behind a proxy tends to emit.
	for name, body := range map[string]string{
		"trailing newline":    tokenEvent + "\n\n" + usageEvent + "\n\n",
		"no trailing newline": tokenEvent + "\n\n" + usageEvent,
		"crlf":                tokenEvent + "\r\n\r\n" + usageEvent + "\r\n\r\n",
	} {
		t.Run(name, func(t *testing.T) {
			raw := []byte(body)
			assertUsage(t, ExtractUsage(raw, "text/event-stream"), map[string]any{
				"prompt": int32(11), "completion": int32(7), "total": int32(18),
			})
			if !HasVisibleToken(raw) {
				t.Error("a body carrying a delta reported no visible token")
			}
		})
	}

	// An empty body must not be mistaken for a line.
	if usage := ExtractUsage(nil, "text/event-stream"); usage.TotalTokens != nil {
		t.Errorf("an empty body reported usage %v", usage.TotalTokens)
	}
	if HasVisibleToken(nil) {
		t.Error("an empty body reported a visible token")
	}
}

func TestAnUnusableUsageFigureIsReportedAbsentRatherThanSubstituted(t *testing.T) {
	// The quota counter reads whatever lands here. A negative total is skipped
	// as "no usage" and buys free tokens; saturating to the maximum instead
	// would spend a token's entire budget on one malformed report and persist
	// it. Neither is a count anyone asked for.
	for name, body := range map[string]string{
		"beyond int32": `{"usage":{"prompt_tokens":1e30,"completion_tokens":5}}`,
		"negative":     `{"usage":{"prompt_tokens":-1,"completion_tokens":5}}`,
	} {
		t.Run(name, func(t *testing.T) {
			usage := ExtractUsage([]byte(body), "application/json")
			if usage.PromptTokens != nil {
				t.Errorf("prompt tokens = %d, want it reported absent", *usage.PromptTokens)
			}
			if usage.TotalTokens != nil && *usage.TotalTokens < 0 {
				t.Errorf("total tokens = %d, want no negative count", *usage.TotalTokens)
			}
		})
	}

	// A figure that does fit is still read normally.
	usage := ExtractUsage([]byte(`{"usage":{"prompt_tokens":11,"completion_tokens":7}}`),
		"application/json")
	if usage.PromptTokens == nil || *usage.PromptTokens != 11 {
		t.Errorf("prompt tokens = %v, want 11", usage.PromptTokens)
	}
}

func TestEffortMappingRewritesEveryShapeAnEffortCanTake(t *testing.T) {
	mappings := map[string]string{"max": "xhigh"}

	for _, testCase := range []struct{ name, body, want string }{
		{
			name: "openai chat completions",
			body: `{"model":"m","reasoning_effort":"max"}`,
			want: "xhigh",
		},
		{
			name: "responses api",
			body: `{"model":"m","reasoning":{"effort":"max"}}`,
			want: "xhigh",
		},
		{
			name: "anthropic messages",
			body: `{"model":"m","output_config":{"effort":"max"}}`,
			want: "xhigh",
		},
		{
			name: "the match ignores case and padding",
			body: `{"model":"m","output_config":{"effort":"  MAX  "}}`,
			want: "xhigh",
		},
	} {
		prepared := PrepareUpstreamBody([]byte(testCase.body), nil, "messages", mappings)
		effort := ExtractReasoningEffort(prepared)
		if effort == nil || *effort != testCase.want {
			t.Errorf("%s: effort = %v, want %s", testCase.name, effort, testCase.want)
		}
	}
}

func TestEffortMappingLeavesUnmappedAndUnconfiguredRequestsAlone(t *testing.T) {
	for _, testCase := range []struct {
		name     string
		mappings map[string]string
		body     string
	}{
		{"no mapping configured", nil, `{"reasoning_effort":"max"}`},
		{"effort is not in the table", map[string]string{"max": "xhigh"}, `{"reasoning_effort":"low"}`},
		{"request states no effort", map[string]string{"max": "xhigh"}, `{"model":"m"}`},
	} {
		prepared := PrepareUpstreamBody([]byte(testCase.body), nil, "messages", testCase.mappings)
		if string(prepared) != testCase.body {
			t.Errorf("%s: body was rewritten to %s, want it untouched", testCase.name, prepared)
		}
	}
}

// A rewrite must not cost the request the rest of the object it was found in:
// dropping thinking or budget_tokens would change what the upstream is asked to
// do, for the sake of translating one field beside them.
func TestEffortMappingKeepsTheSiblingsOfTheEffortItRewrites(t *testing.T) {
	body := []byte(`{"model":"m","thinking":{"type":"adaptive"},` +
		`"output_config":{"effort":"max","verbosity":"low"}}`)

	prepared := PrepareUpstreamBody(body, nil, "messages", map[string]string{"max": "xhigh"})

	var decoded map[string]any
	if err := json.Unmarshal(prepared, &decoded); err != nil {
		t.Fatalf("decode prepared body: %v", err)
	}
	outputConfig, ok := decoded["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("output_config was lost: %s", prepared)
	}
	if outputConfig["effort"] != "xhigh" {
		t.Errorf("effort = %v, want xhigh", outputConfig["effort"])
	}
	if outputConfig["verbosity"] != "low" {
		t.Errorf("a sibling of effort was dropped: %s", prepared)
	}
	thinking, ok := decoded["thinking"].(map[string]any)
	if !ok || thinking["type"] != "adaptive" {
		t.Errorf("the thinking block was disturbed: %s", prepared)
	}
}

// A client that states its effort twice must not reach the upstream with the two
// disagreeing, because which one the upstream reads is its own business.
func TestEffortMappingRewritesEveryEffortARequestStates(t *testing.T) {
	body := []byte(`{"reasoning_effort":"max","output_config":{"effort":"max"}}`)

	prepared := PrepareUpstreamBody(body, nil, "messages", map[string]string{"max": "xhigh"})

	var decoded map[string]any
	if err := json.Unmarshal(prepared, &decoded); err != nil {
		t.Fatalf("decode prepared body: %v", err)
	}
	if decoded["reasoning_effort"] != "xhigh" {
		t.Errorf("reasoning_effort = %v, want xhigh", decoded["reasoning_effort"])
	}
	outputConfig, _ := decoded["output_config"].(map[string]any)
	if outputConfig["effort"] != "xhigh" {
		t.Errorf("output_config.effort = %v, want xhigh", outputConfig["effort"])
	}
}

func TestEffortMappingsFromRowSkipsTheEmptyColumn(t *testing.T) {
	for _, stored := range []string{"", "  ", "{}", "not json"} {
		if mappings := EffortMappingsFromRow(stored); mappings != nil {
			t.Errorf("stored %q gave %v, want no mappings", stored, mappings)
		}
	}
	mappings := EffortMappingsFromRow(`{"max":"xhigh"}`)
	if mappings["max"] != "xhigh" {
		t.Errorf("mappings = %v, want max mapped to xhigh", mappings)
	}
}

// The log has to be able to say what the upstream was actually asked for, or a
// channel that rewrites max into xhigh leaves behind a record claiming the
// request ran at max — the one effort that never reached anyone.
func TestPrepareRequestRecordsBothTheRequestedAndTheForwardedEffort(t *testing.T) {
	body := []byte(`{"model":"m","output_config":{"effort":"max"}}`)

	for _, testCase := range []struct {
		name          string
		stored        string
		wantRequested string
		wantForwarded string
	}{
		{
			name:          "a channel that rewrites the effort records both sides",
			stored:        `{"max":"xhigh"}`,
			wantRequested: "max",
			wantForwarded: "xhigh",
		},
		{
			name:          "a channel that maps nothing forwards what it was given",
			stored:        `{}`,
			wantRequested: "max",
			wantForwarded: "max",
		},
		{
			name:          "a mapping that does not cover this effort leaves it alone",
			stored:        `{"high":"medium"}`,
			wantRequested: "max",
			wantForwarded: "max",
		},
	} {
		upstream := models.UpstreamRow{
			ID: 1, Name: "channel", BaseURL: "https://api.example.test",
			ExtraHeaders: "{}", EffortMappings: testCase.stored,
		}
		prepared, err := PrepareRequest(http.Header{}, &upstream, "POST", "messages",
			"", nil, body, 4096)
		if err != nil {
			t.Fatalf("%s: prepare: %v", testCase.name, err)
		}
		if prepared.ReasoningEffort == nil || *prepared.ReasoningEffort != testCase.wantRequested {
			t.Errorf("%s: requested effort = %v, want %s",
				testCase.name, prepared.ReasoningEffort, testCase.wantRequested)
		}
		if prepared.UpstreamReasoningEffort == nil ||
			*prepared.UpstreamReasoningEffort != testCase.wantForwarded {
			t.Errorf("%s: forwarded effort = %v, want %s",
				testCase.name, prepared.UpstreamReasoningEffort, testCase.wantForwarded)
		}
	}
}

// A request that names no effort must not gain one in the log just because the
// channel carries a mapping table.
func TestPrepareRequestLeavesBothEffortsUnsetWhenTheRequestNamesNone(t *testing.T) {
	upstream := models.UpstreamRow{
		ID: 1, Name: "channel", BaseURL: "https://api.example.test",
		ExtraHeaders: "{}", EffortMappings: `{"max":"xhigh"}`,
	}

	prepared, err := PrepareRequest(http.Header{}, &upstream, "POST", "messages",
		"", nil, []byte(`{"model":"m"}`), 4096)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if prepared.ReasoningEffort != nil || prepared.UpstreamReasoningEffort != nil {
		t.Errorf("efforts = %v / %v, want both unset",
			prepared.ReasoningEffort, prepared.UpstreamReasoningEffort)
	}
}

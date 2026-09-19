package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/liguangsheng/wildtoken/internal/apperr"
	"github.com/liguangsheng/wildtoken/internal/metrics"
	"github.com/liguangsheng/wildtoken/internal/models"
)

// attemptTimeout bounds one upstream attempt and reports whether it was the
// thing that ended it.
//
// The bound is on silence, not on total duration. A deadline over the whole
// attempt cut off streaming answers for the offence of being long, which is
// exactly what a reasoning model produces; each chunk that arrives restarts the
// clock, so what remains bounded is "the upstream stopped sending".
//
// Knowing whether the clock ran out is what separates a gateway timeout from a
// client that walked away, since both reach the read as a cancelled context.
type attemptTimeout struct {
	cancel  context.CancelFunc
	timer   *time.Timer
	window  time.Duration
	expired atomic.Bool
}

func newAttemptTimeout(cancel context.CancelFunc, window time.Duration) *attemptTimeout {
	timeout := &attemptTimeout{cancel: cancel, window: window}
	timeout.timer = time.AfterFunc(window, func() {
		// Recorded before cancelling, so a reader woken by the cancellation
		// always sees the reason for it.
		timeout.expired.Store(true)
		cancel()
	})
	return timeout
}

// extend restarts the clock after the upstream made progress.
func (t *attemptTimeout) extend() { t.timer.Reset(t.window) }

// Expired reports whether this timeout ended the attempt.
func (t *attemptTimeout) Expired() bool { return t.expired.Load() }

// stop releases the timer and the attempt's context.
func (t *attemptTimeout) stop() {
	t.timer.Stop()
	t.cancel()
}

// BuildUpstreamURL builds the full upstream URL for a proxied path.
func BuildUpstreamURL(upstream *models.UpstreamRow, path, queryParams string) string {
	base := strings.TrimRight(upstream.BaseURL, "/")
	suffix := strings.TrimLeft(path, "/")

	// A base that already ends in /v1 is not given a second one.
	target := base + "/v1/" + suffix
	if strings.HasSuffix(base, "/v1") {
		target = base + "/" + suffix
	}
	if queryParams != "" {
		target += "?" + queryParams
	}
	return target
}

// ExtractReasoningEffort reads the requested effort from an OpenAI- or
// Anthropic-compatible request body.
//
// It supports the top-level reasoning_effort (chat completions and the o-series),
// the nested reasoning.effort (Responses API style), and the nested
// output_config.effort (Anthropic Messages API style).
func ExtractReasoningEffort(body []byte) *string {
	var request jsonValue
	if err := json.Unmarshal(body, &request); err != nil {
		return nil
	}

	if effort, ok := formatEffort(request["reasoning_effort"]); ok {
		return &effort
	}
	for _, keys := range [][]string{{"reasoning", "effort"}, {"output_config", "effort"}} {
		if text, ok := valueAt(request, keys...).(string); ok {
			if trimmed := strings.TrimSpace(text); trimmed != "" {
				return &trimmed
			}
		}
	}
	return nil
}

// EffortMappingsFromRow decodes a channel's stored reasoning-effort rewrites.
//
// The empty and "{}" cases are answered without parsing, because this runs on
// every proxied request and almost no channel configures a rewrite.
func EffortMappingsFromRow(stored string) map[string]string {
	trimmed := strings.TrimSpace(stored)
	if trimmed == "" || trimmed == "{}" {
		return nil
	}
	var mappings map[string]string
	if err := json.Unmarshal([]byte(trimmed), &mappings); err != nil {
		return nil
	}
	return mappings
}

// rewriteEffort reads one stored effort value and returns the JSON to put back
// in its place, reporting false when the channel maps it to nothing or to what
// it already says.
//
// The lookup is on the lower-cased value because that is the shape the stored
// keys are normalized to; the replacement goes upstream exactly as written. It
// is always written as a string, which is what every effort field these APIs
// define accepts, even where the caller stated theirs as a number.
func rewriteEffort(raw json.RawMessage, mappings map[string]string) (json.RawMessage, bool) {
	var current any
	if err := json.Unmarshal(raw, &current); err != nil {
		return nil, false
	}
	formatted, ok := formatEffort(current)
	if !ok {
		return nil, false
	}
	replacement, ok := mappings[strings.ToLower(formatted)]
	if !ok || replacement == formatted {
		return nil, false
	}
	encoded, err := json.Marshal(replacement)
	if err != nil {
		return nil, false
	}
	return encoded, true
}

// applyEffortMappings rewrites every place a request states its reasoning
// effort, reporting whether it changed anything.
//
// All three locations are rewritten rather than only the first one found. Which
// of them an upstream reads is its own business, so leaving one still holding
// the downstream value would let the original effort through — and a request
// that states two different efforts is one no upstream promises to resolve the
// way the gateway happened to guess.
func applyEffortMappings(request map[string]json.RawMessage, mappings map[string]string) bool {
	if len(mappings) == 0 {
		return false
	}
	changed := false

	if raw, present := request["reasoning_effort"]; present {
		if replacement, ok := rewriteEffort(raw, mappings); ok {
			request["reasoning_effort"] = replacement
			changed = true
		}
	}

	for _, parent := range []string{"reasoning", "output_config"} {
		raw, present := request[parent]
		if !present {
			continue
		}
		// A non-object here is left alone: rewriting it would mean inventing a
		// shape the caller did not send. Decoding into raw messages keeps the
		// siblings of the effort — thinking, verbosity, summary — byte for byte.
		var nested map[string]json.RawMessage
		if err := json.Unmarshal(raw, &nested); err != nil {
			continue
		}
		effort, present := nested["effort"]
		if !present {
			continue
		}
		replacement, ok := rewriteEffort(effort, mappings)
		if !ok {
			continue
		}
		nested["effort"] = replacement
		reencoded, err := json.Marshal(nested)
		if err != nil {
			continue
		}
		request[parent] = reencoded
		changed = true
	}

	return changed
}

// PrepareUpstreamBody rewrites a JSON request body for its selected upstream.
//
// Streaming Chat Completions responses omit usage by default on many
// OpenAI-compatible upstreams. It is requested explicitly so the gateway can
// consistently record prompt, completion, and total token counts.
//
// effortMappings replaces the reasoning effort the caller asked for with the one
// the channel's upstream understands, so a downstream naming an effort its
// provider does not have is translated rather than refused.
func PrepareUpstreamBody(body []byte, forwardModel *string, path string,
	effortMappings map[string]string) []byte {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return body
	}

	changed := applyEffortMappings(request, effortMappings)

	if forwardModel != nil {
		var currentModel string
		if err := json.Unmarshal(request["model"], &currentModel); err == nil &&
			currentModel != *forwardModel {
			encoded, err := json.Marshal(*forwardModel)
			if err == nil {
				request["model"] = encoded
				changed = true
			}
		}
	}

	if strings.Trim(path, "/") == "chat/completions" && requestsStreaming(request) {
		streamOptions := map[string]json.RawMessage{}
		if raw, present := request["stream_options"]; present {
			if err := json.Unmarshal(raw, &streamOptions); err != nil {
				// A non-object stream_options is replaced rather than merged.
				streamOptions = map[string]json.RawMessage{}
				changed = true
			}
		}
		if !bytes.Equal(streamOptions["include_usage"], []byte("true")) {
			streamOptions["include_usage"] = json.RawMessage("true")
			changed = true
		}
		if encoded, err := json.Marshal(streamOptions); err == nil {
			request["stream_options"] = encoded
		}
	}
	normalizeMessages(request, &changed)

	if !changed {
		return body
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return body
	}
	return encoded
}

// normalizeMessages rewrites the messages array for upstream compatibility.
//
// Three fixes, applied in one decode/encode pass:
//
//  1. Flatten array-formatted text-only content into plain strings. Some
//     upstreams (e.g. futureppo.top) reject content: [{"type":"text"}] with
//     422 because the text field is missing or because they only accept
//     string content. Multimodal arrays (with image or other non-text parts)
//     are left untouched.
//
//  2. Backfill reasoning_content on assistant messages. When any assistant
//     turn carries reasoning_content, thinking mode is active and some
//     upstreams (novaai/deepseek) 400 with "reasoning_content in the thinking
//     mode must be passed back to the API" unless EVERY assistant message
//     includes the field. Clients drop it on turns that only emit tool_calls,
//     so an empty string is inserted to satisfy the round-trip requirement.
//
//  3. Stringify non-string tool_call function arguments. Some clients emit
//     arguments as a raw JSON object; strict upstreams (Cohere via
//     OpenRouter) 400 with "tool arguments must be a stringified JSON
//     object".
func normalizeMessages(request map[string]json.RawMessage, changed *bool) {
	rawMessages, ok := request["messages"]
	if !ok {
		return
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return
	}
	msgChanged := false

	thinkingMode := false
	for i := range messages {
		if messageRole(messages[i]) == "assistant" {
			if _, ok := messages[i]["reasoning_content"]; ok {
				thinkingMode = true
				break
			}
		}
	}

	for i := range messages {
		if flattenTextContent(messages[i]) {
			msgChanged = true
		}
		if stringifyToolCallArguments(messages[i]) {
			msgChanged = true
		}
		if thinkingMode && messageRole(messages[i]) == "assistant" {
			if _, ok := messages[i]["reasoning_content"]; !ok {
				messages[i]["reasoning_content"] = json.RawMessage(`""`)
				msgChanged = true
			}
		}
	}
	if msgChanged {
		encoded, err := json.Marshal(messages)
		if err == nil {
			request["messages"] = encoded
			*changed = true
		}
	}
}

func messageRole(message map[string]json.RawMessage) string {
	var role string
	_ = json.Unmarshal(message["role"], &role)
	return role
}

// stringifyToolCallArguments rewrites assistant tool_calls whose function
// arguments are not a JSON string. The OpenAI schema types arguments as a
// string, but some clients (e.g. opencode-review) emit a raw JSON object;
// strict upstreams (Cohere via OpenRouter) reject those with 400 "tool
// arguments must be a stringified JSON object". Objects, arrays, numbers and
// booleans are re-encoded as their JSON text; null becomes "{}". String
// values (including empty) and absent fields pass through untouched.
// Reports whether the message was modified.
func stringifyToolCallArguments(message map[string]json.RawMessage) bool {
	rawCalls, ok := message["tool_calls"]
	if !ok || len(rawCalls) == 0 || rawCalls[0] != '[' {
		return false
	}
	var calls []map[string]json.RawMessage
	if err := json.Unmarshal(rawCalls, &calls); err != nil {
		return false
	}
	changed := false
	for _, call := range calls {
		fnRaw, ok := call["function"]
		if !ok || len(fnRaw) == 0 || fnRaw[0] != '{' {
			continue
		}
		var fn map[string]json.RawMessage
		if err := json.Unmarshal(fnRaw, &fn); err != nil {
			continue
		}
		rawArgs, present := fn["arguments"]
		if present && len(rawArgs) > 0 && rawArgs[0] == '"' {
			continue // already a string
		}
		if !present || len(rawArgs) == 0 || bytes.Equal(rawArgs, []byte("null")) {
			fn["arguments"] = json.RawMessage(`"{}"`)
		} else if encoded, err := json.Marshal(string(rawArgs)); err == nil {
			fn["arguments"] = json.RawMessage(encoded)
		} else {
			continue
		}
		if encodedFn, err := json.Marshal(fn); err == nil {
			call["function"] = json.RawMessage(encodedFn)
			changed = true
		}
	}
	if !changed {
		return false
	}
	encodedCalls, err := json.Marshal(calls)
	if err != nil {
		return false
	}
	message["tool_calls"] = json.RawMessage(encodedCalls)
	return true
}

// flattenTextContent collapses a text-only content array on one message into a
// plain string, or drops it when empty and no tool_calls carry the turn. It
// reports whether the message was modified.
func flattenTextContent(message map[string]json.RawMessage) bool {
	rawContent, ok := message["content"]
	if !ok {
		return false
	}
	// Only arrays need flattening; strings and null pass through.
	if len(rawContent) == 0 || rawContent[0] != '[' {
		return false
	}
	var parts []map[string]json.RawMessage
	if err := json.Unmarshal(rawContent, &parts); err != nil {
		return false // not a content-part array, leave as-is
	}
	// ponytail: join with newline, empty parts contribute nothing
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		// Fields beyond type/text (cache_control, loss, …) carry semantics
		// flattening would silently drop; leave the array untouched.
		for key := range p {
			if key != "type" && key != "text" {
				return false
			}
		}
		var partType, text string
		_ = json.Unmarshal(p["type"], &partType)
		if partType != "text" {
			return false // multimodal array, leave untouched
		}
		_ = json.Unmarshal(p["text"], &text)
		if text != "" {
			texts = append(texts, text)
		}
	}
	flat := strings.Join(texts, "\n")
	// Some upstreams (futureppo.top) convert empty-string content into
	// [{"type":"text"}] then reject it with 422. Drop the content field
	// when it flattens to empty and there are no tool_calls to carry it.
	if flat == "" {
		if _, hasTC := message["tool_calls"]; !hasTC {
			delete(message, "content")
			return true
		}
	}
	encoded, err := json.Marshal(flat)
	if err != nil {
		return false
	}
	message["content"] = encoded
	return true
}

func requestsStreaming(request map[string]json.RawMessage) bool {
	var stream bool
	return json.Unmarshal(request["stream"], &stream) == nil && stream
}

// PreparedRequest is everything derived from one attempt's request: the URL,
// headers, upstream body, and their log snapshots.
//
// It is computed once and shared by the caller's abort-log fallback and the
// real upstream call, instead of each redoing the same JSON parsing and
// truncation work.
type PreparedRequest struct {
	URL            string
	ForwardHeaders map[string]string
	UpstreamBody   []byte
	// ReasoningEffort is what the caller asked for; UpstreamReasoningEffort is
	// what the upstream was actually sent, which differ once the channel's
	// effort mapping has rewritten one into the other.
	ReasoningEffort         *string
	UpstreamReasoningEffort *string
	DownstreamSnapshot      json.RawMessage
	UpstreamSnapshot        json.RawMessage
}

// PrepareRequest resolves one attempt against its selected upstream.
func PrepareRequest(downstreamHeaders http.Header, upstream *models.UpstreamRow,
	method, path, queryParams string, forwardModel *string, body []byte,
	logBodyMaxBytes int) (*PreparedRequest, error) {
	url := BuildUpstreamURL(upstream, path, queryParams)
	forwardHeaders, err := BuildForwardHeaders(downstreamHeaders, upstream, path)
	if err != nil {
		return nil, err
	}

	effortMappings := EffortMappingsFromRow(upstream.EffortMappings)
	upstreamBody := PrepareUpstreamBody(body, forwardModel, path, effortMappings)

	// Both efforts are logged: the one the caller asked for, and the one the
	// upstream was actually sent. Without the second, a channel that rewrites
	// "max" into "xhigh" leaves a log saying the request ran at "max", which is
	// the one thing that did not happen.
	//
	// The upstream value is re-read from the prepared body rather than inferred
	// from the mapping table, so it reports what was really sent. That read is
	// skipped when the channel maps nothing, because then nothing rewrote the
	// effort and the two are the same string.
	requestEffort := ExtractReasoningEffort(body)
	upstreamEffort := requestEffort
	if len(effortMappings) > 0 {
		upstreamEffort = ExtractReasoningEffort(upstreamBody)
	}

	return &PreparedRequest{
		URL:                     url,
		ForwardHeaders:          forwardHeaders,
		UpstreamBody:            upstreamBody,
		ReasoningEffort:         requestEffort,
		UpstreamReasoningEffort: upstreamEffort,
		DownstreamSnapshot: SnapshotRequest(method, url, forwardHeaders, body,
			logBodyMaxBytes),
		UpstreamSnapshot: SnapshotRequest(method, url, forwardHeaders, upstreamBody,
			logBodyMaxBytes),
	}, nil
}

// Response is a proxied upstream response. Body must always be closed.
type Response struct {
	Status  int
	Headers map[string]string
	Body    io.ReadCloser
}

// Deps are the shared services a proxied request needs.
type Deps struct {
	HTTPClient     *http.Client
	AutoWeight     *AutoWeightManager
	Metrics        *metrics.Runtime
	LogWriter      *LogWriter
	DefaultTimeout time.Duration
}

// RequestContext identifies the caller and the model for one proxied request.
type RequestContext struct {
	DownstreamTokenID   int64
	DownstreamTokenName string
	ClientType          string
	RequestModel        *string
	ForwardModel        *string
	Method              string
	Path                string
	LogBodyMaxBytes     int
}

// ProxyRequest forwards a request upstream, streaming SSE bodies as they arrive.
func ProxyRequest(ctx context.Context, deps Deps, policy AutoWeightPolicy,
	upstream *models.UpstreamRow, requestCtx RequestContext,
	prepared *PreparedRequest) (*Response, error) {
	start := time.Now()
	autoWeightEnabled := upstream.AutoWeightEnabled == 1

	timeout := deps.DefaultTimeout
	if upstream.TimeoutSeconds > 0 {
		timeout = time.Duration(upstream.TimeoutSeconds * float64(time.Second))
	}
	attemptCtx, cancel := context.WithCancel(ctx)
	attempt := newAttemptTimeout(cancel, timeout)

	request, err := buildUpstreamRequest(attemptCtx, requestCtx.Method, prepared)
	if err != nil {
		attempt.stop()

		// The channel's own configuration is what fails here — a base URL the
		// request builder will not accept. Charging it is what eventually takes
		// it out of routing; without that it keeps full weight and is chosen
		// again for every request it is going to fail in the same way.
		deps.AutoWeight.RecordFailure(upstream.ID, autoWeightEnabled, policy)

		// Logged here because the caller disarms its own fallback entry on any
		// error, trusting that the attempt logged itself. This was the one path
		// that did not, so the request left no trace at all.
		message := err.Error()
		statusCode := int32(502)
		entry := baseLogEntry(requestCtx, upstream, prepared)
		entry.StatusCode = &statusCode
		entry.DurationMs = elapsedMs(start)
		entry.Error = &message
		deps.LogWriter.Schedule(entry)

		// Returned unwrapped: buildUpstreamRequest already answers with an
		// upstream error, and wrapping it again repeated the prefix in both the
		// response and the log.
		return nil, err
	}

	response, err := deps.HTTPClient.Do(request)
	if err != nil {
		attempt.stop()

		// A client that walks away cancels this request, and the failure that
		// surfaces here looks like any other. It is not the channel's doing, so
		// it is reported as a client abort and left out of the health score.
		clientGone := !attempt.Expired() && ctx.Err() != nil

		statusCode := int32(502)
		switch {
		case clientGone:
			statusCode = 499
		case attempt.Expired():
			// The attempt's own clock ran out: a gateway timeout.
			statusCode = 504
		}
		if !clientGone {
			deps.AutoWeight.RecordFailure(upstream.ID, autoWeightEnabled, policy)
		}

		message := err.Error()
		entry := baseLogEntry(requestCtx, upstream, prepared)
		entry.StatusCode = &statusCode
		entry.DurationMs = elapsedMs(start)
		entry.Error = &message
		deps.LogWriter.Schedule(entry)

		return nil, apperr.Upstream(message)
	}

	responseHeaders := flattenHeaders(response.Header)
	contentType := responseHeaders["content-type"]
	status := response.StatusCode

	if status >= 200 && status < 300 && IsSSEContentType(contentType) {
		entry := baseLogEntry(requestCtx, upstream, prepared)
		entry.Stream = true
		statusCode := int32(status)
		entry.StatusCode = &statusCode

		stream := newSSEStream(ctx, response.Body, attempt, start, status, responseHeaders,
			requestCtx.LogBodyMaxBytes, entry, deps, policy, autoWeightEnabled, upstream.ID)
		return &Response{Status: status, Headers: responseHeaders, Body: stream}, nil
	}

	bodyBytes, streamedFirstTokenMs, err := readResponseBody(response.Body, start, attempt.extend)
	response.Body.Close()
	attempt.stop()
	if err != nil {
		clientGone := !attempt.Expired() && ctx.Err() != nil

		statusCode := int32(502)
		switch {
		case clientGone:
			statusCode = 499
		case attempt.Expired():
			statusCode = 504
		}
		if !clientGone {
			deps.AutoWeight.RecordFailure(upstream.ID, autoWeightEnabled, policy)
		}

		message := err.Error()
		entry := baseLogEntry(requestCtx, upstream, prepared)
		entry.StatusCode = &statusCode
		entry.DurationMs = elapsedMs(start)
		entry.Error = &message
		deps.LogWriter.Schedule(entry)
		return nil, apperr.Upstream(message)
	}

	if status >= 200 && status < 300 {
		deps.AutoWeight.RecordSuccess(upstream.ID, autoWeightEnabled, policy)
	} else {
		deps.AutoWeight.RecordFailure(upstream.ID, autoWeightEnabled, policy)
	}

	responseSnapshot := SnapshotResponse(status, responseHeaders, bodyBytes,
		requestCtx.LogBodyMaxBytes)
	usage := ExtractUsage(bodyBytes, contentType)
	isStream := bytes.HasPrefix(bodyBytes, []byte("data:")) ||
		strings.Contains(contentType, "event-stream")

	// A true streamed time-to-first-token is preferred; buffered detection is
	// only a fallback, and only for stream bodies.
	var firstTokenMs *int32
	if isStream {
		firstTokenMs = streamedFirstTokenMs
		if firstTokenMs == nil && HasVisibleToken(bodyBytes) {
			firstTokenMs = elapsedMs(start)
		}
	}

	entry := baseLogEntry(requestCtx, upstream, prepared)
	statusCode := int32(status)
	entry.StatusCode = &statusCode
	entry.Stream = isStream
	entry.ResponseReasoningEffort = ExtractResponseReasoningEffort(bodyBytes, contentType)
	entry.PromptTokens = usage.PromptTokens
	entry.CompletionTokens = usage.CompletionTokens
	entry.TotalTokens = usage.TotalTokens
	entry.PromptCachedTokens = usage.PromptCachedTokens
	entry.CacheCreationTokens = usage.CacheCreationTokens
	entry.CompletionReasoningTokens = usage.CompletionReasoningTokens
	entry.FirstTokenMs = firstTokenMs
	entry.DurationMs = elapsedMs(start)
	entry.UpstreamResponse = responseSnapshot
	entry.DownstreamResponse = responseSnapshot

	// Scheduled when the body is closed rather than here. Here is before the
	// response has reached the client at all, so a client that leaves during
	// delivery would be recorded as having received what the upstream sent.
	return &Response{
		Status:  status,
		Headers: responseHeaders,
		Body:    newBufferedStream(ctx, bodyBytes, entry, deps, statusCode),
	}, nil
}

func buildUpstreamRequest(ctx context.Context, method string, prepared *PreparedRequest) (*http.Request, error) {
	var body io.Reader
	if len(prepared.UpstreamBody) > 0 {
		body = bytes.NewReader(prepared.UpstreamBody)
	}

	request, err := http.NewRequestWithContext(ctx, method, prepared.URL, body)
	if err != nil {
		return nil, apperr.Upstream(err.Error())
	}
	for name, value := range prepared.ForwardHeaders {
		if containsFold(HopByHopHeaders, name) {
			continue
		}
		request.Header.Set(name, value)
	}
	return request, nil
}

// baseLogEntry fills the fields every attempt reports, regardless of outcome.
func baseLogEntry(requestCtx RequestContext, upstream *models.UpstreamRow,
	prepared *PreparedRequest) LogEntry {
	upstreamID := upstream.ID
	upstreamName := upstream.Name
	tokenID := requestCtx.DownstreamTokenID
	tokenName := requestCtx.DownstreamTokenName
	clientType := requestCtx.ClientType

	return LogEntry{
		Method:                  requestCtx.Method,
		Path:                    requestCtx.Path,
		DownstreamTokenID:       &tokenID,
		DownstreamTokenName:     &tokenName,
		ClientType:              &clientType,
		UpstreamID:              &upstreamID,
		UpstreamName:            &upstreamName,
		Model:                   requestCtx.ForwardModel,
		RequestModel:            requestCtx.RequestModel,
		UpstreamModel:           requestCtx.ForwardModel,
		ReasoningEffort:         prepared.ReasoningEffort,
		UpstreamReasoningEffort: prepared.UpstreamReasoningEffort,
		DownstreamRequest:       prepared.DownstreamSnapshot,
		UpstreamRequest:         prepared.UpstreamSnapshot,
	}
}

func elapsedMs(start time.Time) *int32 {
	measured := int32(time.Since(start).Milliseconds())
	return &measured
}

func flattenHeaders(headers http.Header) map[string]string {
	flattened := make(map[string]string, len(headers))
	for name, values := range headers {
		if len(values) > 0 {
			flattened[strings.ToLower(name)] = values[0]
		}
	}
	return flattened
}

// MaxUpstreamResponseBytes caps a buffered upstream response.
//
// The downstream request body is already bounded, but the response was not: a
// misbehaving or compromised channel could return a body large enough to exhaust
// the gateway's memory, and a handful of concurrent ones could do it outright.
// The limit is far above any real completion, so it only ever catches a channel
// that is not answering in good faith.
const MaxUpstreamResponseBytes = 128 << 20

// ErrUpstreamResponseTooLarge reports a buffered response that ran past the cap.
var ErrUpstreamResponseTooLarge = errors.New("upstream response exceeded the maximum buffered size")

// readResponseBody reads a full upstream body while recording the true
// time-to-first-token for SSE streams.
//
// progress is called for each chunk, so the attempt's clock measures silence
// from the upstream rather than the total time a long body takes to arrive.
func readResponseBody(body io.Reader, start time.Time, progress func()) ([]byte, *int32, error) {
	var collected bytes.Buffer
	observation := &sseObservation{}
	measure := func() int32 { return int32(time.Since(start).Milliseconds()) }

	buffer := make([]byte, 32*1024)
	for {
		read, err := body.Read(buffer)
		if read > 0 {
			if collected.Len()+read > MaxUpstreamResponseBytes {
				return nil, nil, ErrUpstreamResponseTooLarge
			}
			progress()
			chunk := buffer[:read]
			collected.Write(chunk)
			observation.observeChunk(chunk, measure)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
	}

	// The final partial line is observed too, keeping parity with buffered
	// detection.
	observation.finish(measure)
	return collected.Bytes(), observation.firstTokenMs, nil
}

package app

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"github.com/liguangsheng/wildtoken/internal/appstate"
	"github.com/liguangsheng/wildtoken/internal/authstate"
	"github.com/liguangsheng/wildtoken/internal/config"
	"github.com/liguangsheng/wildtoken/internal/db"
	"github.com/liguangsheng/wildtoken/internal/metrics"
	"github.com/liguangsheng/wildtoken/internal/models"
	"github.com/liguangsheng/wildtoken/internal/proxy"
	"github.com/liguangsheng/wildtoken/internal/quota"
	"github.com/liguangsheng/wildtoken/internal/ratelimit"
)

const (
	// requestDrainTimeout is how long shutdown waits for in-flight requests. A
	// streaming answer can outlive it, which is why the log writer has to be
	// safe to schedule onto afterwards.
	requestDrainTimeout = 15 * time.Second
	// logDrainTimeout bounds the wait for the log queue, so a database that has
	// stopped answering cannot hold the process open indefinitely.
	logDrainTimeout = 30 * time.Second
	// readHeaderTimeout bounds how long a caller may take to send its request
	// headers.
	readHeaderTimeout = 20 * time.Second
)

// ReadyInfo reports the bound port and console URL once the server is listening.
type ReadyInfo struct {
	Port     uint16
	AdminURL string
}

// Server owns a listening WildToken instance.
type Server struct {
	state      *appstate.State
	httpServer *http.Server
	listener   net.Listener
	logWriter  *proxy.LogWriter
	cancelJobs context.CancelFunc
	Ready      ReadyInfo
}

// New loads configuration, opens the database, and binds the listener.
//
// It does not serve; the caller decides when to start accepting, so a tray can
// learn the bound port first.
func New(ctx context.Context) (*Server, error) {
	settings, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("load configuration: %w", err)
	}

	database, err := openDatabase(ctx, settings.Database)
	if err != nil {
		return nil, err
	}
	if err := db.Init(ctx, database); err != nil {
		database.Close()
		return nil, fmt.Errorf("initialize schema: %w", err)
	}
	if err := db.CheckAutoVacuum(ctx, database); err != nil {
		slog.Warn("could not read the SQLite auto-vacuum mode", "error", err)
	}

	credential, err := loadOrBootstrapAdminCredential(ctx, database, settings.Admin.Token,
		settings.Server.Host)
	if err != nil {
		database.Close()
		return nil, err
	}
	// The startup token is bootstrap material only; it is never retained as a
	// fallback credential.
	settings.Admin.Token = ""

	throttle := authstate.NewThrottle()
	credentials, err := authstate.NewCredentials(credential, throttle)
	if err != nil {
		database.Close()
		return nil, err
	}

	runtimeMetrics := metrics.New()
	logStats, err := db.LoadLogStatsCache(ctx, database)
	if err != nil {
		database.Close()
		return nil, fmt.Errorf("load log statistics: %w", err)
	}

	jobsCtx, cancelJobs := context.WithCancel(context.Background())
	quotas := quota.NewTracker()
	logWriter := proxy.NewLogWriter(jobsCtx, database, runtimeMetrics, logStats,
		settings.Logging.LogQueueCapacity, quotas)

	state := &appstate.State{
		DB:                  database,
		Settings:            settings,
		AutoWeight:          proxy.NewAutoWeightManager(),
		Runtime:             appstate.NewSettingsStore(appstate.LoadRuntimeSettings(ctx, database)),
		Credentials:         credentials,
		Throttle:            throttle,
		Metrics:             runtimeMetrics,
		LogWriter:           logWriter,
		LogStats:            logStats,
		ModelsCache:         appstate.NewModelsListCache(),
		Routing:             proxy.NewRoutingCache(),
		TokenRateLimiter:    ratelimit.NewLimiter(),
		UpstreamRateLimiter: ratelimit.NewLimiter(),
		Quotas:              quotas,
		StartedAt:           time.Now(),
	}
	// The client reads the proxy setting through the runtime store on every
	// request, so a console edit applies to new connections without a restart.
	state.HTTPClient = newHTTPClient(settings.Upstream.DefaultTimeoutSeconds, state.Runtime.Get)

	go db.RunLogStatsRefreshLoop(jobsCtx, database, logStats, runtimeMetrics)
	go proxy.RunCleanupLoop(jobsCtx, database, state.Runtime.Get, runtimeMetrics, logStats)

	bindAddr := net.JoinHostPort(settings.Server.Host,
		fmt.Sprintf("%d", settings.Server.Port))
	listener, err := net.Listen("tcp", bindAddr)
	if err != nil {
		cancelJobs()
		logWriter.Close()
		database.Close()
		return nil, fmt.Errorf("bind %s: %w", bindAddr, err)
	}

	port := uint16(listener.Addr().(*net.TCPAddr).Port)
	return &Server{
		state: state,
		httpServer: &http.Server{
			Handler: NewRouter(state),
			// Only the header deadline is set. A connection that opens and then
			// dribbles its request line held a goroutine and a descriptor for
			// as long as it liked; bounding the headers alone costs a streaming
			// response nothing, whereas ReadTimeout would cap a long request
			// body and WriteTimeout would cut every SSE answer short.
			ReadHeaderTimeout: readHeaderTimeout,
		},
		listener:   listener,
		logWriter:  logWriter,
		cancelJobs: cancelJobs,
		Ready: ReadyInfo{
			Port:     port,
			AdminURL: AdminURLFromSettings(settings.Server.Host, port),
		},
	}, nil
}

// Serve accepts connections until ctx is cancelled, then shuts down gracefully.
func (s *Server) Serve(ctx context.Context) error {
	slog.Info("WildToken starting", "address", s.listener.Addr().String())

	serveErr := make(chan error, 1)
	go func() {
		err := s.httpServer.Serve(s.listener)
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		serveErr <- err
	}()

	select {
	case err := <-serveErr:
		s.shutdownResources()
		return err
	case <-ctx.Done():
	}

	slog.Info("shutdown signal received")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), requestDrainTimeout)
	defer cancel()
	err := s.httpServer.Shutdown(shutdownCtx)
	s.shutdownResources()
	slog.Info("WildToken stopped")
	return err
}

// shutdownResources drains the log queue and then stops the background jobs, so
// requests served just before shutdown still reach the database.
//
// The queue is drained first on purpose. Cancelling the jobs context ahead of it
// left every queued row to be written under a context that was already done, so
// the drain this function exists for failed in full — taking the quota
// increments those rows carry with it.
func (s *Server) shutdownResources() {
	drained := s.logWriter.CloseWithin(logDrainTimeout)
	s.cancelJobs()
	s.state.TokenRateLimiter.Close()
	s.state.UpstreamRateLimiter.Close()

	if !drained {
		// The writer is still working. Closing the database under it would turn
		// every batch it has left into an error about a closed database, which
		// says nothing about the actual cause: that shutdown stopped waiting.
		// The process is exiting either way, so the handle goes with it.
		return
	}
	s.state.DB.Close()
}

// openDatabase applies the SQLite pragmas the service depends on.
func openDatabase(ctx context.Context, settings config.DatabaseSettings) (*sql.DB, error) {
	dsn, err := sqliteDSN(settings)
	if err != nil {
		return nil, err
	}

	database, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}
	database.SetMaxOpenConns(max(settings.MaxConnections, 1))
	database.SetConnMaxIdleTime(time.Duration(settings.IdleTimeoutSeconds) * time.Second)

	if err := database.PingContext(ctx); err != nil {
		database.Close()
		return nil, fmt.Errorf("connect to database: %w", err)
	}
	return database, nil
}

// sqliteDSN converts the configured sqlx-style URL into the form the pure-Go
// driver expects, carrying the pragmas over as connection parameters.
func sqliteDSN(settings config.DatabaseSettings) (string, error) {
	raw := settings.URL
	raw = strings.TrimPrefix(raw, "sqlite://")
	raw = strings.TrimPrefix(raw, "sqlite:")
	if raw == "" {
		return "", errors.New("database url is empty")
	}

	path, query, _ := strings.Cut(raw, "?")
	values, err := url.ParseQuery(query)
	if err != nil {
		return "", fmt.Errorf("parse database url: %w", err)
	}
	// mode=rwc is sqlx's spelling of "create if missing", which is the driver's
	// default, and it rejects the parameter outside a file: URI.
	values.Del("mode")

	// synchronous(NORMAL) and a generous busy_timeout pair with WAL (set in
	// schema.go): FULL fsyncs on every commit, which lengthens the write-lock
	// hold that retention deletes already stretch, and 5s was shorter than the
	// worst hold, so concurrent BEGIN IMMEDIATE writers got SQLITE_BUSY instead
	// of waiting. NORMAL is crash-safe under WAL (it only risks the last commit
	// on power loss); 10s stays below the log writer's 15s write timeout.
	pragmas := []string{
		"foreign_keys(1)",
		"auto_vacuum(2)",
		fmt.Sprintf("cache_size(-%d)", max(settings.SQLiteCacheSizeKiB, 256)),
		fmt.Sprintf("mmap_size(%d)", max(settings.SQLiteMmapSizeBytes, 0)),
		"busy_timeout(10000)",
		"synchronous(1)",
	}
	for _, pragma := range pragmas {
		values.Add("_pragma", pragma)
	}
	// Stores that read and then write inside one transaction depend on this; see
	// db.SQLiteTxLock for what deferred locking does to them.
	values.Set("_txlock", db.SQLiteTxLock)

	return path + "?" + values.Encode(), nil
}

// maxUpstreamRedirects bounds how far a channel may forward the gateway.
//
// A provider that moved an endpoint is worth following; a chain longer than this
// is a channel leading the gateway somewhere it was not configured to go.
const maxUpstreamRedirects = 3

func newHTTPClient(defaultTimeoutSeconds float64, runtime func() models.RuntimeSettings) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConnsPerHost = 20
	// A streamed response must reach the client as it arrives, so the transport
	// is told not to buffer by requesting compression itself.
	transport.DisableCompression = true
	// The proxy decision is read per request rather than fixed at startup, so a
	// console edit takes effect without restarting. The transport keys pooled
	// connections by proxy, so toggling never reuses a connection dialed the
	// other way. When disabled, the environment variables keep their say, which
	// is what this transport inherited before the setting existed.
	transport.Proxy = func(req *http.Request) (*url.URL, error) {
		settings := runtime()
		if !settings.ProxyEnabled || strings.TrimSpace(settings.ProxyURL) == "" {
			return http.ProxyFromEnvironment(req)
		}
		return url.Parse(strings.TrimSpace(settings.ProxyURL))
	}
	// No ResponseHeaderTimeout: this transport is shared by every channel, so a
	// value here is a ceiling on all of them. Setting it to the default cut off
	// channels configured with a longer one, and did it in a way the caller
	// could not tell from a channel failing outright. Waiting for the response
	// headers is already bounded per attempt, against the channel's own timeout.
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxUpstreamRedirects {
				return fmt.Errorf("upstream redirected more than %d times", maxUpstreamRedirects)
			}
			return nil
		},
		// Per-request deadlines carry the real timeout, because a streaming
		// response legitimately outlives any client-wide limit.
		Timeout: 0,
	}
}

// loadOrBootstrapAdminCredential resolves the credential this process will
// authenticate against, refusing configurations that would expose a known
// default beyond the local machine.
func loadOrBootstrapAdminCredential(ctx context.Context, database *sql.DB,
	startupToken, bindHost string) (models.AdminCredential, error) {
	loopbackOnly := IsLoopbackBindHost(bindHost)

	credential, found, err := db.LoadAdminCredential(ctx, database)
	if err != nil {
		return models.AdminCredential{}, err
	}
	if found {
		usesKnownDefault := authstate.VerifyAdminToken(credential, "change-me")
		if usesKnownDefault && !loopbackOnly {
			return models.AdminCredential{}, errors.New(
				"refusing non-loopback startup with the legacy change-me admin credential; " +
					"start on localhost and rotate it first")
		}
		if usesKnownDefault {
			slog.Warn("the stored admin credential is still change-me; " +
				"rotate it before exposing WildToken beyond localhost")
		}
		return credential, nil
	}

	token := strings.TrimSpace(startupToken)
	if strings.EqualFold(token, "change-me") {
		if !loopbackOnly {
			return models.AdminCredential{}, errors.New(
				"a new database listening beyond localhost requires an explicit ADMIN_TOKEN")
		}
		slog.Warn("using the local-only bootstrap admin token change-me; " +
			"rotate it from the admin console")
	} else if _, err := models.ValidateAdminTokenValue(token); err != nil {
		return models.AdminCredential{}, err
	}

	hash, err := authstate.HashAdminToken(token)
	if err != nil {
		return models.AdminCredential{}, err
	}
	return db.BootstrapAdminCredential(ctx, database, hash)
}

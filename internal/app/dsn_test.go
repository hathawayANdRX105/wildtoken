package app

import (
	"net/url"
	"strings"
	"testing"

	"github.com/liguangsheng/wildtoken/internal/config"
	"github.com/liguangsheng/wildtoken/internal/db"
)

// The lock-discipline pragmas are invisible until a retention delete holds the
// write lock past a competing writer's patience — the SQLITE_BUSY reporters of
// 2026-09-28 and 2026-10-01. Nothing in the store fails visibly if the DSN
// silently loses one of them during a future edit, so the built DSN itself is
// pinned here the same way the db package pins the txlock.
func TestTheDSNCarriesTheLockDisciplinePragmas(t *testing.T) {
	settings := config.Default()
	dsn, err := sqliteDSN(settings.Database)
	if err != nil {
		t.Fatalf("dsn: %v", err)
	}
	_, rawQuery, _ := strings.Cut(dsn, "?")
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		t.Fatalf("parse query: %v", err)
	}

	pragma := func(want string) bool {
		for _, p := range values["_pragma"] {
			if p == want {
				return true
			}
		}
		return false
	}
	// 10s, not 5s: a bounded retention lock hold must be waited out, not
	// failed; 1 = synchronous NORMAL under WAL.
	if !pragma("busy_timeout(10000)") {
		t.Errorf("busy_timeout(10000) missing from %v", values["_pragma"])
	}
	if !pragma("synchronous(1)") {
		t.Errorf("synchronous(1) missing from %v", values["_pragma"])
	}
	if got := values.Get("_txlock"); got != db.SQLiteTxLock {
		t.Errorf("_txlock = %q, want %q", got, db.SQLiteTxLock)
	}
	// The pool must be wide enough that parked contended writers cannot
	// starve read-only admin queries (the 2026-09-28 recent-RPM deadline).
	if settings.Database.MaxConnections < 4 {
		t.Errorf("MaxConnections = %d, want a pool wide enough to park writers without starving reads", settings.Database.MaxConnections)
	}
}

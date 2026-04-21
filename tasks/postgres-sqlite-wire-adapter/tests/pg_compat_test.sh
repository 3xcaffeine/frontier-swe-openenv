#!/usr/bin/env bash
# PostgreSQL Wire Compatibility Test Suite
# Graded tests for postgres-sqlite-wire-adapter submissions
# Runs psql against a candidate server and checks results.
#
# Usage: PG_PORT=55432 ./pg_compat_test.sh
#        Expects candidate server already running on 127.0.0.1:$PG_PORT

set -uo pipefail

PORT="${PG_PORT:-55432}"
HOST="${PG_HOST:-127.0.0.1}"
PSQL="psql -h $HOST -p $PORT -d postgres -t -A -X --no-psqlrc"
PSQL_FULL="psql -h $HOST -p $PORT -d postgres -X --no-psqlrc"

PASS=0
FAIL=0
SKIP=0
TIER_SCORES=()
RESULTS=""

pass() { PASS=$((PASS+1)); RESULTS="${RESULTS}\n  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); RESULTS="${RESULTS}\n  FAIL: $1 (got: '${2:-}', expected: '${3:-}')"; }
skip() { SKIP=$((SKIP+1)); RESULTS="${RESULTS}\n  SKIP: $1"; }

# Run a query, capture output, compare to expected
check_query() {
    local name="$1"
    local query="$2"
    local expected="$3"

    local actual
    actual=$(echo "$query" | timeout 10 $PSQL 2>/dev/null | sed '/^$/d' | tr -d '\r') || true

    if [ "$actual" = "$expected" ]; then
        pass "$name"
    else
        fail "$name" "$actual" "$expected"
    fi
}

# Run a query, check it doesn't error (any output is ok)
check_no_error() {
    local name="$1"
    local query="$2"

    if echo "$query" | timeout 10 $PSQL 2>/dev/null >/dev/null; then
        pass "$name"
    else
        fail "$name" "ERROR" "no error"
    fi
}

# Run a query, check it returns at least N rows
check_has_rows() {
    local name="$1"
    local query="$2"
    local min_rows="$3"

    local count
    count=$(echo "$query" | timeout 10 $PSQL 2>/dev/null | wc -l | tr -d ' ') || count=0

    if [ "$count" -ge "$min_rows" ]; then
        pass "$name"
    else
        fail "$name" "${count} rows" ">=${min_rows} rows"
    fi
}

# Check that a query produces an error
check_error() {
    local name="$1"
    local query="$2"
    local error_fragment="$3"

    local stderr
    stderr=$(echo "$query" | timeout 10 $PSQL 2>&1 >/dev/null) || true

    if echo "$stderr" | grep -qi "$error_fragment"; then
        pass "$name"
    else
        fail "$name" "$stderr" "error containing '$error_fragment'"
    fi
}

# Check formatted output (with alignment, headers, row count)
check_formatted() {
    local name="$1"
    local query="$2"
    local expected="$3"

    local actual
    actual=$(echo "$query" | timeout 10 $PSQL_FULL 2>/dev/null | tr -d '\r') || true

    if [ "$actual" = "$expected" ]; then
        pass "$name"
    else
        # Show first differing line
        local diff_line
        diff_line=$(diff <(echo "$actual") <(echo "$expected") 2>/dev/null | head -5)
        fail "$name" "output differs" "exact match (diff: $diff_line)"
    fi
}

# ===================================================================
echo "=== PostgreSQL Wire Compatibility Test Suite ==="
echo "=== Target: $HOST:$PORT ==="
echo ""

# -------------------------------------------------------------------
echo "--- Tier 1: Connection & Basic Protocol ---"
TIER_START=$PASS

# Can we connect at all?
if timeout 5 $PSQL -c "SELECT 1" >/dev/null 2>&1; then
    pass "T1.1 psql connects"
else
    fail "T1.1 psql connects" "connection refused" "connection"
    echo ""
    echo "FATAL: Cannot connect to server. Aborting."
    echo "Total: $PASS passed, $FAIL failed"
    exit 1
fi

check_query "T1.2 SELECT 1" "SELECT 1;" "1"
check_query "T1.3 SELECT string literal" "SELECT 'hello';" "hello"
check_query "T1.4 SELECT arithmetic" "SELECT 1 + 1;" "2"
check_query "T1.5 SELECT multiple cols" "SELECT 1 AS a, 2 AS b;" "1|2"
check_query "T1.6 SELECT NULL" "SELECT NULL;" ""

TIER1=$((PASS - TIER_START))
TIER_SCORES+=("Tier1:$TIER1/6")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 2: Server Identity & Parameters ---"
TIER_START=$PASS

check_no_error "T2.1 SHOW server_version" "SHOW server_version;"
check_no_error "T2.2 SHOW server_encoding" "SHOW server_encoding;"
check_no_error "T2.3 SELECT version()" "SELECT version();"
check_no_error "T2.4 SELECT current_database()" "SELECT current_database();"
check_no_error "T2.5 SELECT current_user" "SELECT current_user;"
check_no_error "T2.6 SHOW search_path" "SHOW search_path;"

TIER2=$((PASS - TIER_START))
TIER_SCORES+=("Tier2:$TIER2/6")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 3: DDL & Basic DML ---"
TIER_START=$PASS

check_no_error "T3.1 CREATE TABLE" \
    "CREATE TABLE test_basic(id INTEGER PRIMARY KEY, name TEXT, value REAL);"
check_no_error "T3.2 INSERT single row" \
    "INSERT INTO test_basic VALUES (1, 'alice', 3.14);"
check_no_error "T3.3 INSERT multiple values" \
    "INSERT INTO test_basic VALUES (2, 'bob', 2.71), (3, 'carol', 1.41);"
check_query "T3.4 SELECT count" "SELECT count(*) FROM test_basic;" "3"
check_query "T3.5 SELECT with WHERE" \
    "SELECT name FROM test_basic WHERE id = 1;" "alice"
check_query "T3.6 SELECT with ORDER BY" \
    "SELECT name FROM test_basic ORDER BY id;" "alice
bob
carol"
check_no_error "T3.7 UPDATE" \
    "UPDATE test_basic SET value = 9.99 WHERE id = 2;"
check_query "T3.8 verify UPDATE" \
    "SELECT value FROM test_basic WHERE id = 2;" "9.99"
check_no_error "T3.9 DELETE" \
    "DELETE FROM test_basic WHERE id = 3;"
check_query "T3.10 verify DELETE" \
    "SELECT count(*) FROM test_basic;" "2"
check_no_error "T3.11 DROP TABLE" "DROP TABLE test_basic;"
check_error "T3.12 query dropped table" \
    "SELECT * FROM test_basic;" "does not exist\|no such table\|relation.*not"

TIER3=$((PASS - TIER_START))
TIER_SCORES+=("Tier3:$TIER3/12")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 4: Data Types & Formatting ---"
TIER_START=$PASS

# Integer type formatting (right-aligned in psql when OID is correct)
check_query "T4.1 integer returns integer" "SELECT 42::integer;" "42"
check_query "T4.2 boolean true → t" "SELECT true;" "t"
check_query "T4.3 boolean false → f" "SELECT false;" "f"
check_query "T4.4 bool cast 't'" "SELECT 't'::boolean;" "t"
check_query "T4.5 bool cast 'false'" "SELECT 'false'::boolean;" "f"
check_query "T4.6 NULL::integer" "SELECT NULL::integer;" ""
check_query "T4.7 text concat" "SELECT 'hello' || ' ' || 'world';" "hello world"
check_query "T4.8 integer division" "SELECT 7 / 2;" "3"
check_query "T4.9 float division" "SELECT 7.0 / 2;" "3.5000000000000000"

# Check that the column type OID is correct (psql right-aligns integers)
# This is the exact bug we found - OID_TEXT causes left-alignment
EXPECTED_INT=$(printf " one \n-----\n   1\n(1 row)\n")
check_formatted "T4.10 int4 column alignment" "SELECT 1 AS one;" "$EXPECTED_INT"

EXPECTED_BOOL=$(printf " bool \n------\n t\n(1 row)\n")
check_formatted "T4.11 bool column value" "SELECT true AS bool;" "$EXPECTED_BOOL"

TIER4=$((PASS - TIER_START))
TIER_SCORES+=("Tier4:$TIER4/11")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 5: Transactions ---"
TIER_START=$PASS

check_no_error "T5.1 BEGIN" "BEGIN;"
check_no_error "T5.2 CREATE in tx" \
    "BEGIN; CREATE TABLE tx_test(id INT); INSERT INTO tx_test VALUES(1); COMMIT;"
check_query "T5.3 data persisted after COMMIT" \
    "SELECT id FROM tx_test;" "1"
check_no_error "T5.4 ROLLBACK" \
    "BEGIN; INSERT INTO tx_test VALUES(2); ROLLBACK;"
check_query "T5.5 data not persisted after ROLLBACK" \
    "SELECT count(*) FROM tx_test;" "1"
check_no_error "T5.6 cleanup" "DROP TABLE tx_test;"

TIER5=$((PASS - TIER_START))
TIER_SCORES+=("Tier5:$TIER5/6")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 6: SQL Features ---"
TIER_START=$PASS

check_no_error "T6.0 setup" \
    "CREATE TABLE products(id SERIAL PRIMARY KEY, name TEXT NOT NULL, price NUMERIC, category TEXT);"
check_no_error "T6.0b insert data" \
    "INSERT INTO products(name, price, category) VALUES
     ('Widget', 9.99, 'A'), ('Gadget', 24.99, 'B'),
     ('Doohickey', 4.99, 'A'), ('Thingamajig', 49.99, 'B'),
     ('Whatsit', 14.99, 'A');"

check_query "T6.1 LIKE" \
    "SELECT name FROM products WHERE name LIKE 'W%' ORDER BY name;" "Whatsit
Widget"
check_query "T6.2 IN clause" \
    "SELECT count(*) FROM products WHERE category IN ('A');" "3"
check_query "T6.3 BETWEEN" \
    "SELECT count(*) FROM products WHERE price BETWEEN 10 AND 30;" "2"
check_query "T6.4 GROUP BY + aggregate" \
    "SELECT category, count(*) FROM products GROUP BY category ORDER BY category;" "A|3
B|2"
check_query "T6.5 HAVING" \
    "SELECT category FROM products GROUP BY category HAVING count(*) > 2;" "A"
check_query "T6.6 subquery" \
    "SELECT name FROM products WHERE price = (SELECT max(price) FROM products);" "Thingamajig"
check_query "T6.7 COALESCE" \
    "SELECT COALESCE(NULL, NULL, 'fallback');" "fallback"
check_query "T6.8 CASE expression" \
    "SELECT CASE WHEN 1=1 THEN 'yes' ELSE 'no' END;" "yes"
check_no_error "T6.9 CREATE INDEX" \
    "CREATE INDEX idx_products_cat ON products(category);"
check_query "T6.10 DISTINCT" \
    "SELECT DISTINCT category FROM products ORDER BY category;" "A
B"
check_no_error "T6.11 cleanup" "DROP TABLE products;"

TIER6=$((PASS - TIER_START))
TIER_SCORES+=("Tier6:$TIER6/13")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 7: PostgreSQL System Catalogs ---"
TIER_START=$PASS

check_has_rows "T7.1 pg_catalog.pg_type" \
    "SELECT typname FROM pg_catalog.pg_type LIMIT 5;" 1
check_has_rows "T7.2 pg_catalog.pg_class" \
    "SELECT relname FROM pg_catalog.pg_class LIMIT 5;" 1
check_has_rows "T7.3 pg_catalog.pg_namespace" \
    "SELECT nspname FROM pg_catalog.pg_namespace LIMIT 5;" 1
check_no_error "T7.4 information_schema.tables" \
    "SELECT table_name FROM information_schema.tables LIMIT 1;"
check_no_error "T7.5 pg_database" \
    "SELECT datname FROM pg_catalog.pg_database LIMIT 1;"
check_has_rows "T7.6 pg_settings" \
    "SELECT name FROM pg_catalog.pg_settings LIMIT 5;" 1

TIER7=$((PASS - TIER_START))
TIER_SCORES+=("Tier7:$TIER7/6")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 8: Error Handling ---"
TIER_START=$PASS

check_error "T8.1 syntax error" \
    "SELEC 1;" "syntax"
check_error "T8.2 table not found" \
    "SELECT * FROM nonexistent_table_xyz;" "does not exist\|no such\|not found"
check_error "T8.3 column not found" \
    "CREATE TABLE err_test(id INT); SELECT nonexistent_col FROM err_test;" \
    "does not exist\|no such\|not found\|no column"
check_error "T8.4 type mismatch" \
    "SELECT 'not_a_number'::integer;" "invalid\|cannot\|error"
check_error "T8.5 duplicate key" \
    "CREATE TABLE dup_test(id INT PRIMARY KEY); INSERT INTO dup_test VALUES(1); INSERT INTO dup_test VALUES(1);" \
    "duplicate\|unique\|constraint\|UNIQUE"
check_no_error "T8.6 cleanup" \
    "DROP TABLE IF EXISTS err_test; DROP TABLE IF EXISTS dup_test;"

TIER8=$((PASS - TIER_START))
TIER_SCORES+=("Tier8:$TIER8/6")
echo ""

# -------------------------------------------------------------------
echo "--- Tier 9: Multi-statement & Session ---"
TIER_START=$PASS

# Multiple statements in one query string (simple query mode)
check_no_error "T9.1 multi-statement" \
    "CREATE TABLE multi_test(x INT); INSERT INTO multi_test VALUES(1); SELECT * FROM multi_test; DROP TABLE multi_test;"
check_no_error "T9.2 SET command" "SET client_encoding TO 'UTF8';"
check_no_error "T9.3 RESET command" "RESET client_encoding;"
check_query "T9.4 pg_typeof" "SELECT pg_typeof(1);" "integer"
check_query "T9.5 pg_typeof text" "SELECT pg_typeof('hello'::text);" "text"
check_no_error "T9.6 empty query" ";"

TIER9=$((PASS - TIER_START))
TIER_SCORES+=("Tier9:$TIER9/6")
echo ""

# -------------------------------------------------------------------
echo "=== RESULTS ==="
echo -e "$RESULTS"
echo ""
echo "=== TIER SUMMARY ==="
TOTAL=$((PASS + FAIL))
for ts in "${TIER_SCORES[@]}"; do
    echo "  $ts"
done
echo ""
echo "Total: $PASS/$TOTAL passed ($FAIL failed, $SKIP skipped)"
echo ""

# Compute percentage
if [ "$TOTAL" -gt 0 ]; then
    PCT=$((PASS * 100 / TOTAL))
    echo "Score: ${PCT}%"
fi

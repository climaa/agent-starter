#!/usr/bin/env bash
# End-to-end smoke test: docker compose (API + n8n) with the offline fake LLM/embeddings.
#
#   n8n POST /webhook/ask ─▶ api POST /webhook/n8n ─▶ agent ─▶ create_ticket ─▶ n8n POST /webhook/ticket
#
# No cloud credentials and no model-provider calls. Needs docker (compose v2) and curl.
# Uses host ports 8000 and 5678, so stop any running `make docker-up` stack first.
#   scripts/smoke.sh               # run, then tear everything down (incl. volumes)
#   KEEP_UP=1 scripts/smoke.sh     # leave the stack running for poking around
set -euo pipefail

cd "$(dirname "$0")/.."

export N8N_VERSION="${N8N_VERSION:-2.47.4}"   # pinned: workflow publish CLI is n8n >= 2.x
export N8N_WEBHOOK_TOKEN="${N8N_WEBHOOK_TOKEN:-smoke-$(date +%s)}"  # exercises the auth path
PROJECT="${SMOKE_PROJECT:-agent-starter-smoke}"
WORKFLOW_ID="agentStarterAsk1"   # fixed id in n8n/agent-workflow.json
API="http://localhost:8000"
N8N="http://localhost:5678"
TIMEOUT_S="${SMOKE_TIMEOUT_S:-180}"

dc() { docker compose -p "$PROJECT" -f docker-compose.yml -f docker-compose.offline.yml "$@"; }
log() { printf '[smoke] %s\n' "$*"; }
fail() {
  log "FAIL: $*"
  dc ps || true
  dc logs --tail 60 || true
  exit 1
}
cleanup() {
  if [[ "${KEEP_UP:-0}" == "1" ]]; then
    log "KEEP_UP=1: stack left running (docker compose -p $PROJECT down -v to stop)"
  else
    dc down -v --remove-orphans >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# wait_for <description> <command...>: retry until the command succeeds or TIMEOUT_S passes.
wait_for() {
  local what="$1"; shift
  local deadline=$((SECONDS + TIMEOUT_S))
  until "$@" >/dev/null 2>&1; do
    ((SECONDS < deadline)) || fail "timed out after ${TIMEOUT_S}s waiting for $what"
    sleep 2
  done
  log "ready: $what"
}

# post <url> <json> [header]: prints "<body>\n<http code>".
post() {
  local args=(-sS -X POST "$1" -H 'content-type: application/json' -d "$2" -w '\n%{http_code}')
  [[ $# -ge 3 ]] && args+=(-H "$3")
  curl "${args[@]}"
}
webhook_registered() {
  # n8n answers 404 until the published workflow's webhooks are registered.
  [[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$N8N/webhook/ticket" \
    -H 'content-type: application/json' -d '{}')" == "200" ]]
}

log "starting api + n8n (n8n $N8N_VERSION, offline providers)"
dc down -v --remove-orphans >/dev/null 2>&1 || true
dc up -d --build
wait_for "api /health agent_ready" sh -c "curl -sf $API/health | grep -q '\"agent_ready\":true'"
# /healthz/readiness (not /healthz) waits for DB migrations; the CLI below races them otherwise.
wait_for "n8n /healthz/readiness" curl -sf "$N8N/healthz/readiness"

log "importing + publishing n8n workflow"
n8n_cli() {
  local out
  out="$(dc exec -T n8n n8n "$@" 2>&1)" || { printf '%s\n' "$out"; fail "n8n $1"; }
  grep -iE 'import|publish' <<<"$out" | grep -v license || true
}
n8n_cli import:workflow --input=/workflows/agent-workflow.json
n8n_cli publish:workflow --id="$WORKFLOW_ID"
wait_for "n8n webhooks registered" webhook_registered

log "1/3 RAG answer via n8n /webhook/ask"
out="$(post "$N8N/webhook/ask" '{"message":"How many days to request a refund?","thread_id":"smoke"}')"
code="${out##*$'\n'}"; body="${out%$'\n'*}"
log "HTTP $code $body"
[[ "$code" == "200" ]] || fail "ask returned HTTP $code"
grep -q '30 days' <<<"$body" || fail "answer missing '30 days'"
grep -q '\[S1\]' <<<"$body" || fail "answer missing citation"

log "2/3 tool call: agent create_ticket -> n8n /webhook/ticket"
out="$(post "$N8N/webhook/ask" '{"message":"Please open a ticket: my parcel arrived damaged"}')"
code="${out##*$'\n'}"; body="${out%$'\n'*}"
log "HTTP $code ${body:0:300}..."
[[ "$code" == "200" ]] || fail "ticket ask returned HTTP $code"
grep -q 'N8N-[0-9]' <<<"$body" || fail "no n8n ticket id in answer (ticket webhook not reached)"

log "3/3 api /webhook/n8n rejects a missing token (so 1/3 proves n8n sent it)"
out="$(post "$API/webhook/n8n" '{"message":"hi"}')"
code="${out##*$'\n'}"
[[ "$code" == "401" ]] || fail "expected 401 without token, got $code"
log "HTTP $code as expected"

log "api access log:"
dc logs api 2>/dev/null | grep 'POST /webhook/n8n' | tail -3
log "PASS"

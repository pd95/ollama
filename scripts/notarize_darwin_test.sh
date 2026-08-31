#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/notarize_darwin.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/notarize-darwin-test.XXXXXX")"
FAKE_BIN="$TEST_ROOT/bin"
APP_ID="92f6d9c5-ed60-490d-ae8d-e890a4bcafeb"
DMG_ID="2b168f6a-2c3d-4aa0-8533-a28f765e471f"
PASS_COUNT=0

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT HUP INT TERM

fail() {
  printf 'not ok - %s\n' "$*" >&2
  exit 1
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf 'ok %d - %s\n' "$PASS_COUNT" "$1"
}

assert_contains() {
  local file="$1" expected="$2"
  grep -F -- "$expected" "$file" >/dev/null || fail "expected '$expected' in $file"
}

assert_not_contains_tree() {
  local root="$1" unexpected="$2"
  if grep -R -F "$unexpected" "$root" >/dev/null 2>&1; then
    fail "found secret text below $root"
  fi
}

mkdir -p "$FAKE_BIN"

cat >"$FAKE_BIN/plutil" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  -create)
    : >"$3"
    ;;
  -replace|-insert)
    operation="$1"
    key="$2"
    value="$4"
    file="$5"
    if [[ "$operation" == -replace ]] && ! grep -q "^${key}=" "$file" 2>/dev/null; then
      exit 1
    fi
    temp="${file}.tmp"
    awk -F= -v key="$key" '$1 != key { print }' "$file" 2>/dev/null >"$temp" || true
    printf '%s=%s\n' "$key" "$value" >>"$temp"
    mv "$temp" "$file"
    ;;
  -extract)
    key="$2"
    file="${@: -1}"
    sed -n "s/^${key}=//p" "$file" | head -n 1
    ;;
  *)
    exit 2
    ;;
esac
EOF

cat >"$FAKE_BIN/shasum" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == -a && "$2" == 256 ]]
shift 2
sha256sum "$@"
EOF

cat >"$FAKE_BIN/xcrun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == -f ]]
printf 'xcrun:developer-dir:%s:%s\n' "${DEVELOPER_DIR:-active}" "$2" >>"$TRACE"
printf '%s/%s\n' "$FAKE_BIN" "$2"
EOF

cat >"$FAKE_BIN/codesign" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
target="${@: -1}"
if [[ " $* " == *" -d "* ]]; then
  printf '%s\n' 'Authority=Developer ID Application: Test' 'Identifier=com.electron.ollama' 'TeamIdentifier=TESTTEAM' 'Timestamp=2026-08-31 at 15:00:00' 'Runtime Version=26.4.0' 'CDHash=0123456789abcdef' >&2
  exit 0
fi
if [[ " $* " == *" --verify "* ]]; then
  printf 'codesign:verify:%s\n' "$target" >>"$TRACE"
  [[ "${FAKE_CODESIGN_FAIL:-0}" != 1 ]]
  [[ "$target" != "${FAKE_CODESIGN_FAIL_TARGET:-}" ]]
  exit
fi
printf 'codesign:sign:%s\n' "$target" >>"$TRACE"
EOF

cat >"$FAKE_BIN/ditto" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -x ]]; then
  destination="${@: -1}"
  mkdir -p "$destination"
  cp -R "$TEST_DIST/Ollama.app" "$destination/Ollama.app"
  if [[ -n "${FAKE_EXTRACT_METALLIB_SUBSTITUTE:-}" ]]; then
    printf 'substituted\n' >"$destination/Ollama.app/Contents/Resources/mlx_metal_${FAKE_EXTRACT_METALLIB_SUBSTITUTE}/mlx.metallib"
  fi
else
  destination="${@: -1}"
  printf 'zip:%s\n' "$TEST_CASE" >"$destination"
  printf 'ditto:create:%s\n' "$destination" >>"$TRACE"
fi
EOF

cat >"$FAKE_BIN/stapler" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'stapler:%s:%s\n' "$1" "$2" >>"$TRACE"
EOF

cat >"$FAKE_BIN/create-dmg" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'dmg:%s\n' "$TEST_CASE" >Ollama.dmg
printf 'dmg:create\n' >>"$TRACE"
EOF

cat >"$FAKE_BIN/notarytool" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
command="$1"
shift
app_id='92f6d9c5-ed60-490d-ae8d-e890a4bcafeb'
dmg_id='2b168f6a-2c3d-4aa0-8533-a28f765e471f'
stage_for_id() {
  if [[ "$1" == "$app_id" ]]; then printf app; else printf dmg; fi
}
next_status() {
  local stage="$1" file="$FAKE_STATUS_DIR/$stage.status" first temp
  [[ -s "$file" ]] || return 1
  first="$(head -n 1 "$file")"
  temp="$file.tmp"
  tail -n +2 "$file" >"$temp"
  mv "$temp" "$file"
  printf '%s\n' "$first"
}
case "$command" in
  --version)
    printf '1.1.2 (41)\n'
    ;;
  submit)
    artifact="$1"
    if [[ "$artifact" == *.dmg ]]; then stage=dmg; id="$dmg_id"; else stage=app; id="$app_id"; fi
    printf 'notarytool:submit:%s\n' "$stage" >>"$TRACE"
    if [[ " $* " == *" --no-s3-acceleration "* ]]; then
      printf 'notarytool:no-s3-acceleration:%s\n' "$stage" >>"$TRACE"
    fi
    if [[ "${FAKE_NO_ID_STAGE:-}" == "$stage" ]]; then
      printf 'transport failed before response\n' >&2
      exit 1
    fi
    if [[ "${FAKE_UPLOAD_UNCONFIRMED_STAGE:-}" != "$stage" ]]; then
      printf 'command: notarytool submit --apple-id %s --password %s --team-id %s\n' "$APPLE_ID" "$APPLE_PASSWORD" "$APPLE_TEAM_ID" >&2
      printf '[UPLOAD] Received new upload status: Succeeded\n' >&2
      printf '[UPLOAD] Multipart upload process has completed successfully.\n' >&2
    fi
    printf 'id=%s\n' "$id"
    if [[ "${FAKE_SUBMIT_NONZERO_STAGE:-}" == "$stage" ]]; then exit 1; fi
    ;;
  info)
    id="$1"
    stage="$(stage_for_id "$id")"
    printf 'notarytool:info:%s\n' "$stage" >>"$TRACE"
    if [[ "${FAKE_INFO_UNAVAILABLE_STAGE:-}" == "$stage" ]]; then exit 1; fi
    status="$(next_status "$stage")"
    if [[ "$stage" == app ]]; then name="${FAKE_APP_NAME:-Ollama-darwin.zip}"; else name="${FAKE_DMG_NAME:-Ollama.dmg}"; fi
    printf 'id=%s\nname=%s\ncreatedDate=2026-07-30T12:00:00Z\nstatus=%s\n' "$id" "$name" "$status"
    ;;
  wait)
    stage="$(stage_for_id "$1")"
    printf 'notarytool:wait:%s:%s\n' "$stage" "$3" >>"$TRACE"
    exit "${FAKE_WAIT_RC:-0}"
    ;;
  log)
    id="$1"
    path="$2"
    stage="$(stage_for_id "$id")"
    printf 'notarytool:log:%s\n' "$stage" >>"$TRACE"
    printf 'invalid artifact for %s; credentials %s %s %s\n' "$stage" "$APPLE_ID" "$APPLE_PASSWORD" "$APPLE_TEAM_ID" >"$path"
    ;;
  *) exit 2 ;;
esac
EOF

chmod +x "$FAKE_BIN"/*

init_case() {
  TEST_CASE="$1"
  CASE_ROOT="$TEST_ROOT/$TEST_CASE"
  TEST_DIST="$CASE_ROOT/dist"
  STATE_ROOT="$TEST_DIST/notarization"
  FAKE_STATUS_DIR="$CASE_ROOT/status"
  TRACE="$CASE_ROOT/trace.log"
  mkdir -p "$TEST_DIST/Ollama.app/Contents/Resources/mlx_metal_v3" "$TEST_DIST/Ollama.app/Contents/Resources/mlx_metal_v4" "$TEST_DIST/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A" "$TEST_DIST/Ollama.app/Contents/MacOS" "$FAKE_STATUS_DIR"
  printf 'CFBundleShortVersionString=0.31.2\nCFBundleVersion=7cb13ca-test\n' >"$TEST_DIST/Ollama.app/Contents/Info.plist"
  printf 'zip:original\n' >"$TEST_DIST/Ollama-darwin.zip"
  touch "$TEST_DIST/Ollama.app/Contents/Resources/ollama" "$TEST_DIST/Ollama.app/Contents/Resources/llama-server" "$TEST_DIST/Ollama.app/Contents/Resources/llama-quantize" "$TEST_DIST/Ollama.app/Contents/Resources/mlx_metal_v3/libmlx.dylib" "$TEST_DIST/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel" "$TEST_DIST/Ollama.app/Contents/MacOS/Ollama"
  printf 'v3 payload\n' >"$TEST_DIST/Ollama.app/Contents/Resources/mlx_metal_v3/mlx.metallib"
  printf 'v4 payload\n' >"$TEST_DIST/Ollama.app/Contents/Resources/mlx_metal_v4/mlx.metallib"
  PRE_SIGN_BUNDLE="$CASE_ROOT/pre-sign"; mkdir -p "$PRE_SIGN_BUNDLE"
  printf '%s\n' '{"status":"pre_sign_pass","acceptance":"pre_sign_only"}' >"$PRE_SIGN_BUNDLE/status.json"
  printf '%s\n' '{"candidate":"fixed"}' >"$PRE_SIGN_BUNDLE/gate-inputs.json"
  SIGNING_RECEIPT="$CASE_ROOT/signed-receipt.json"
  jq -n --arg gate_inputs_file_sha256 "$(sha256sum "$PRE_SIGN_BUNDLE/gate-inputs.json" | awk '{print $1}')" --arg gate_status_file_sha256 "$(sha256sum "$PRE_SIGN_BUNDLE/status.json" | awk '{print $1}')" '{equivalent:true,differences:[],developer_id_authority:"Developer ID Application: Test",gate_inputs_file_sha256:$gate_inputs_file_sha256,gate_status_file_sha256:$gate_status_file_sha256}' >"$SIGNING_RECEIPT"
  : >"$TRACE"
  export TEST_CASE TEST_DIST FAKE_STATUS_DIR TRACE FAKE_BIN PRE_SIGN_BUNDLE SIGNING_RECEIPT
  unset FAKE_NO_ID_STAGE FAKE_SUBMIT_NONZERO_STAGE FAKE_INFO_UNAVAILABLE_STAGE FAKE_WAIT_RC FAKE_APP_NAME FAKE_DMG_NAME FAKE_CODESIGN_FAIL FAKE_CODESIGN_FAIL_TARGET FAKE_EXTRACT_METALLIB_SUBSTITUTE FAKE_UPLOAD_UNCONFIRMED_STAGE MLX_REPLACE_NOTARIZATION_ID MLX_NOTARY_NO_S3_ACCELERATION
}

run_helper() {
  env \
    PATH="$FAKE_BIN:$PATH" \
    MLX_NOTARIZATION_TESTING=1 \
    MLX_NOTARIZATION_DIST_DIR="$TEST_DIST" \
    MLX_NOTARIZATION_STATE_ROOT="$STATE_ROOT" \
    MLX_CREATE_DMG_COMMAND="$FAKE_BIN/create-dmg" \
    APPLE_IDENTITY='Developer ID Application: Test' \
    APPLE_ID='secret-apple-id@example.invalid' \
    APPLE_TEAM_ID='SECRETTEAM' \
    APPLE_PASSWORD='secret-password' \
    "$HELPER" \
      --release-revision 7cb13cae \
      --release-version 0.31.2-7cb13ca-test \
      --app-version 0.31.2 \
      --app-build-version 7cb13ca-test \
      --timeout 1s --signing-receipt "$SIGNING_RECEIPT" --pre-sign-bundle "$PRE_SIGN_BUNDLE" "$@"
}

state_file() {
  printf '%s/0.31.2-7cb13ca-test/state.plist\n' "$STATE_ROOT"
}

init_case immediate_acceptance
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
run_helper >"$CASE_ROOT/output" 2>&1
assert_contains "$(state_file)" 'activeStage=complete'
assert_contains "$(state_file)" 'completedDmgSHA256='
assert_contains "$(state_file)" 'notarytoolVersion=1.1.2 (41)'
assert_contains "$TRACE" 'notarytool:submit:app'
assert_contains "$TRACE" 'notarytool:submit:dmg'
assert_contains "$(state_file)" 'appUploadStatus=Confirmed'
assert_contains "$(state_file)" 'dmgUploadStatus=Confirmed'
pass 'immediate app and DMG acceptance completes'

init_case fresh_archive_preflight
run_helper --preflight-archive --artifact-dir "$CASE_ROOT/preflight" >"$CASE_ROOT/output" 2>&1
[[ -f "$CASE_ROOT/preflight/Ollama-darwin.fresh.zip" ]] || fail 'fresh archive preflight did not create archive'
assert_not_contains_tree "$CASE_ROOT/preflight" 'zip:original'
if grep -F 'notarytool:submit:' "$TRACE" >/dev/null; then fail 'preflight contacted Apple submission service'; fi
pass 'fresh archive preflight never reuses stale ZIP or contacts Apple'

init_case unsigned_macho_rejected
export FAKE_CODESIGN_FAIL_TARGET="$TEST_DIST/Ollama.app/Contents/MacOS/Ollama"
set +e
run_helper --preflight-archive --artifact-dir "$CASE_ROOT/preflight" >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail 'unsigned Mach-O target was accepted'
[[ ! -e "$CASE_ROOT/preflight/Ollama-darwin.fresh.zip" ]] || fail 'unsigned Mach-O target created archive'
if grep -F 'notarytool:submit:' "$TRACE" >/dev/null; then fail 'unsigned Mach-O preflight contacted Apple'; fi
pass 'unsigned Mach-O target fails before archive creation'

init_case missing_metallib_rejected
rm "$TEST_DIST/Ollama.app/Contents/Resources/mlx_metal_v4/mlx.metallib"
set +e
run_helper --preflight-archive --artifact-dir "$CASE_ROOT/preflight" >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail 'missing Metal library was accepted'
assert_contains "$CASE_ROOT/output" 'missing required Metal library payload'
[[ ! -e "$CASE_ROOT/preflight/Ollama-darwin.fresh.zip" ]] || fail 'missing Metal library created archive'
pass 'missing Metal library fails before archive creation'

init_case substituted_metallib_rejected
export FAKE_EXTRACT_METALLIB_SUBSTITUTE=v3
set +e
run_helper --preflight-archive --artifact-dir "$CASE_ROOT/preflight" >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail 'substituted extracted Metal library was accepted'
assert_contains "$CASE_ROOT/output" 'Metal library payload hash mismatch after archive extraction'
[[ -f "$CASE_ROOT/preflight/Ollama-darwin.fresh.zip" ]] || fail 'substituted Metal library did not reach extraction binding'
if grep -F 'notarytool:submit:' "$TRACE" >/dev/null; then fail 'substituted Metal library preflight contacted Apple'; fi
pass 'substituted extracted Metal library fails archive binding'

init_case preflight_binding_substitution
sed -i 's/true/false/' "$SIGNING_RECEIPT"
set +e
run_helper --preflight-archive --artifact-dir "$CASE_ROOT/preflight" >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail 'invalid signing receipt was not rejected'
assert_contains "$CASE_ROOT/output" 'signed receipt is not equivalent'
[[ ! -e "$CASE_ROOT/preflight/Ollama-darwin.fresh.zip" ]] || fail 'invalid binding created preflight archive'
pass 'preflight rejects substituted signing receipt before archive creation'

init_case alternate_notarytool
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
export MLX_NOTARY_DEVELOPER_DIR="$TEST_ROOT/AlternateXcode/Contents/Developer"
mkdir -p "$MLX_NOTARY_DEVELOPER_DIR"
run_helper >"$CASE_ROOT/output" 2>&1
assert_contains "$TRACE" "xcrun:developer-dir:$MLX_NOTARY_DEVELOPER_DIR:notarytool"
assert_contains "$(state_file)" "notaryDeveloperDir=$MLX_NOTARY_DEVELOPER_DIR"
assert_contains "$CASE_ROOT/output" 'starting Apple app submission with notarytool 1.1.2 (41)'
pass 'alternate notarytool developer directory is scoped and recorded'

init_case transport_timeout
printf 'In Progress\nIn Progress\n' >"$FAKE_STATUS_DIR/app.status"
export FAKE_SUBMIT_NONZERO_STAGE=app
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 75 ]] || fail "transport timeout returned $rc instead of 75"
assert_contains "$(state_file)" "appSubmissionID=$APP_ID"
assert_contains "$CASE_ROOT/output" "--resume-notarization $APP_ID"
assert_contains "$CASE_ROOT/output" 'submit returned while it was still In Progress; waiting up to 1s'
assert_contains "$TRACE" 'notarytool:wait:app:1s'
pass 'transport timeout preserves a recoverable submission ID'

init_case upload_unconfirmed
printf 'In Progress\nIn Progress\n' >"$FAKE_STATUS_DIR/app.status"
export FAKE_UPLOAD_UNCONFIRMED_STAGE=app
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail "unconfirmed upload returned $rc instead of 1"
assert_contains "$(state_file)" 'appUploadStatus=Unconfirmed'
assert_contains "$CASE_ROOT/output" 'WARNING: S3 upload completion was not confirmed'
assert_contains "$CASE_ROOT/output" "--replace-notarization $APP_ID --notary-no-s3-acceleration"
assert_contains "$CASE_ROOT/output" 'status polling cannot repair missing S3 parts'
pass 'missing verbose S3 completion marker is retained as unconfirmed'

init_case pending_then_accepted
printf 'In Progress\nIn Progress\nIn Progress\nIn Progress\nAccepted\n' >"$FAKE_STATUS_DIR/app.status"
set +e
run_helper >"$CASE_ROOT/first" 2>&1
first_rc=$?
run_helper --resume "$APP_ID" >"$CASE_ROOT/second" 2>&1
second_rc=$?
set -e
[[ "$first_rc" == 75 && "$second_rc" == 75 ]] || fail 'pending retries did not return 75'
assert_contains "$CASE_ROOT/second" "found recorded app submission $APP_ID with Apple status: In Progress"
assert_contains "$CASE_ROOT/second" "waiting up to 1s for Apple to finish submission $APP_ID"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
run_helper --resume "$APP_ID" >"$CASE_ROOT/third" 2>&1
assert_contains "$(state_file)" 'activeStage=complete'
[[ "$(grep -c 'notarytool:submit:app' "$TRACE")" == 1 ]] || fail 'resume resubmitted the app'
pass 'pending submission can later resume without resubmission'

init_case app_invalid
printf 'Invalid\n' >"$FAKE_STATUS_DIR/app.status"
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail "invalid app returned $rc"
assert_contains "$TRACE" 'notarytool:log:app'
assert_not_contains_tree "$STATE_ROOT" 'secret-password'
pass 'invalid app fetches a redacted Apple log'

init_case dmg_invalid
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Invalid\n' >"$FAKE_STATUS_DIR/dmg.status"
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail "invalid DMG returned $rc"
assert_contains "$TRACE" 'notarytool:log:dmg'
pass 'invalid DMG fetches the Apple log and stops'

init_case no_submission_id
export FAKE_NO_ID_STAGE=app
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail "missing submission ID returned $rc"
assert_contains "$CASE_ROOT/output" 'failed before Apple returned a submission ID'
pass 'failure before an ID is fatal and not retried'

init_case unknown_service
export FAKE_INFO_UNAVAILABLE_STAGE=app
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 75 ]] || fail "unknown Apple status returned $rc"
assert_contains "$(state_file)" 'appLastStatus=Unknown'
pass 'unavailable Apple status retains state and returns 75'

init_case duplicate_prevention
printf 'In Progress\n' >"$FAKE_STATUS_DIR/app.status"
set +e
run_helper >"$CASE_ROOT/first" 2>&1
run_helper >"$CASE_ROOT/second" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail "duplicate attempt returned $rc"
[[ "$(grep -c 'notarytool:submit:app' "$TRACE")" == 1 ]] || fail 'duplicate app submission was created'
assert_contains "$CASE_ROOT/second" 'refusing to create a duplicate submission'
pass 'matching nonterminal state prevents duplicate submission'

init_case legacy_empty_preflight_rotation
mkdir -p "$STATE_ROOT/0.31.2-7cb13ca-test"
printf 'releaseRevision=7cb13cae\nreleaseVersion=0.31.2-7cb13ca-test\nappVersion=0.31.2\nappBuildVersion=7cb13ca-test\n' >"$(state_file)"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
run_helper >"$CASE_ROOT/output" 2>&1
compgen -G "$STATE_ROOT/0.31.2-7cb13ca-test.preflight.*" >/dev/null || fail 'legacy empty preflight state was not rotated'
assert_contains "$CASE_ROOT/output" 'rotated non-submission preflight state'
[[ "$(grep -c 'notarytool:submit:app' "$TRACE")" == 1 ]] || fail 'legacy empty preflight state did not begin exactly one fresh fake submission'
pass 'legacy empty preflight state rotates without resume or duplicate logic'

init_case malformed_empty_preflight_rejected
mkdir -p "$STATE_ROOT/0.31.2-7cb13ca-test"
printf 'releaseRevision=7cb13cae\nreleaseVersion=0.31.2-7cb13ca-test\nappVersion=0.31.2\nappBuildVersion=7cb13ca-test\nappArtifactPath=/bad\n' >"$(state_file)"
set +e
run_helper >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail 'malformed empty preflight state was accepted'
assert_contains "$CASE_ROOT/output" 'invalid local preflight state: empty stage has submission, artifact, or hash fields'
if grep -F 'notarytool:submit:' "$TRACE" >/dev/null; then fail 'malformed empty preflight state contacted Apple'; fi
pass 'malformed empty preflight state fails locally without resume guidance'

init_case explicit_replacement
printf 'In Progress\nIn Progress\n' >"$FAKE_STATUS_DIR/app.status"
set +e
run_helper >"$CASE_ROOT/first" 2>&1
first_rc=$?
set -e
[[ "$first_rc" == 75 ]] || fail 'replacement setup did not retain pending state'
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
export MLX_REPLACE_NOTARIZATION_ID="$APP_ID"
export MLX_NOTARY_NO_S3_ACCELERATION=1
run_helper >"$CASE_ROOT/second" 2>&1
assert_contains "$CASE_ROOT/second" "explicitly replaced submission $APP_ID"
[[ "$(grep -c 'notarytool:submit:app' "$TRACE")" == 2 ]] || fail 'explicit replacement did not create exactly one new app submission'
assert_contains "$TRACE" 'notarytool:no-s3-acceleration:app'
assert_contains "$(state_file)" 's3Acceleration=Disabled'
compgen -G "$STATE_ROOT/0.31.2-7cb13ca-test.replaced.*" >/dev/null || fail 'replaced state archive is missing'
pass 'matching explicit replacement archives state and creates one new submission'

init_case mismatches
printf 'In Progress\n' >"$FAKE_STATUS_DIR/app.status"
set +e
run_helper >"$CASE_ROOT/first" 2>&1
set -e
sed -i 's/releaseRevision=7cb13cae/releaseRevision=wrong/' "$(state_file)"
set +e
run_helper --resume "$APP_ID" >"$CASE_ROOT/revision" 2>&1
revision_rc=$?
set -e
[[ "$revision_rc" == 1 ]] || fail 'revision mismatch was not rejected'
sed -i 's/releaseRevision=wrong/releaseRevision=7cb13cae/' "$(state_file)"
printf 'changed\n' >>"$STATE_ROOT/0.31.2-7cb13ca-test/Ollama-darwin.submitted.zip"
set +e
run_helper --resume "$APP_ID" >"$CASE_ROOT/hash" 2>&1
hash_rc=$?
set -e
[[ "$hash_rc" == 1 ]] || fail 'hash mismatch was not rejected'
pass 'release revision and artifact hash mismatches are rejected'

init_case bootstrap_current
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
run_helper --resume "$APP_ID" >"$CASE_ROOT/output" 2>&1
assert_contains "$CASE_ROOT/output" "found Apple submission $APP_ID: name=Ollama-darwin.zip, stage=app, status=Accepted"
assert_contains "$CASE_ROOT/output" 'bootstrapped notarization state'
assert_contains "$(state_file)" 'activeStage=complete'
pass 'state-free current app submission bootstraps safely'

init_case reject_state_free_dmg
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
export FAKE_DMG_NAME=Ollama.dmg
set +e
run_helper --resume "$DMG_ID" >"$CASE_ROOT/output" 2>&1
rc=$?
set -e
[[ "$rc" == 1 ]] || fail 'state-free DMG recovery was not rejected'
assert_contains "$CASE_ROOT/output" 'state-free recovery only accepts an app submission'
pass 'state-free DMG recovery is rejected'

init_case operation_order
printf 'Accepted\n' >"$FAKE_STATUS_DIR/app.status"
printf 'Accepted\n' >"$FAKE_STATUS_DIR/dmg.status"
run_helper >"$CASE_ROOT/output" 2>&1
order="$(sed -nE '/stapler:staple:.*Ollama.app$|dmg:create|notarytool:submit:dmg|stapler:staple:.*Ollama.dmg$/p' "$TRACE" | sed -E 's#stapler:staple:.*Ollama.app#app-staple#; s#stapler:staple:.*Ollama.dmg#dmg-staple#' | paste -sd, -)"
[[ "$order" == 'app-staple,dmg:create,notarytool:submit:dmg,dmg-staple' ]] || fail "unexpected operation order: $order"
pass 'app staple precedes DMG creation and DMG staple'

assert_not_contains_tree "$TEST_ROOT" 'secret-password'
assert_not_contains_tree "$TEST_ROOT" 'secret-apple-id@example.invalid'
assert_not_contains_tree "$TEST_ROOT" 'SECRETTEAM'
pass 'credentials are absent from generated output and state'

printf '1..%d\n' "$PASS_COUNT"

#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="${MLX_NOTARIZATION_DIST_DIR:-$REPO_DIR/dist}"
STATE_ROOT="${MLX_NOTARIZATION_STATE_ROOT:-$DIST_DIR/notarization}"
TIMEOUT="${MLX_NOTARY_TIMEOUT:-20m}"
RELEASE_REVISION=""
RELEASE_VERSION=""
APP_VERSION=""
APP_BUILD_VERSION=""
RESUME_ID=""
PREFLIGHT_ARCHIVE=0
PREFLIGHT_DIR=""
SIGNING_RECEIPT=""
PRE_SIGN_BUNDLE=""
REPLACE_SUBMISSION_ID="${MLX_REPLACE_NOTARIZATION_ID:-}"
NO_S3_ACCELERATION="${MLX_NOTARY_NO_S3_ACCELERATION:-0}"
NOTARY_DEVELOPER_DIR="${MLX_NOTARY_DEVELOPER_DIR:-}"
STATE_DIR=""
STATE_FILE=""
NOTARYTOOL=""
NOTARYTOOL_VERSION=""
STAPLER=""
INFO_STATUS=""
INFO_NAME=""
INFO_CREATED_DATE=""
TEMP_PATHS=""
METALLIB_RELATIVE_PATHS=(
  "Contents/Resources/mlx_metal_v3/mlx.metallib"
  "Contents/Resources/mlx_metal_v4/mlx.metallib"
)

EX_TEMPFAIL=75

usage() {
  cat <<'EOF'
Usage: ./scripts/notarize_darwin.sh --release-revision REV --release-version VERSION \
    --app-version VERSION --app-build-version VERSION [--timeout DURATION] [--resume SUBMISSION_ID]
    --preflight-archive --artifact-dir DIR --signing-receipt FILE --pre-sign-bundle DIR

Without --resume, submit the signed app ZIP and continue through DMG notarization.
With --resume, continue an existing app-ZIP or DMG submission without rebuilding or resubmitting it.
EOF
}

require_release_signature() {
  local target="$1" details authority team
  codesign --verify --strict --verbose=4 "$target"
  details="$(codesign -d --verbose=4 "$target" 2>&1)"
  authority="$(printf '%s\n' "$details" | sed -nE 's/^Authority=(Developer ID Application:.*)$/\1/p' | head -n1)"
  team="$(printf '%s\n' "$details" | sed -nE 's/^TeamIdentifier=(.+)$/\1/p' | head -n1)"
  [[ -n "$authority" && -n "$team" ]] || die "missing Developer ID authority or TeamIdentifier: $target"
  printf '%s\n' "$details" | grep -q '^Timestamp=' || die "missing secure timestamp: $target"
  { printf '%s\n' "$details" | grep -q '^Runtime Version=' || printf '%s\n' "$details" | grep -q 'flags=.*runtime'; } || die "missing hardened runtime: $target"
}

verify_metallib_data() {
  local app="$1" actual expected relative
  for relative in "${METALLIB_RELATIVE_PATHS[@]}"; do
    [[ -f "$app/$relative" ]] || die "missing required Metal library payload: $app/$relative"
  done
  actual="$(cd "$app" && find Contents/Resources -type f -name '*.metallib' -print | LC_ALL=C sort)"
  expected="$(printf '%s\n' "${METALLIB_RELATIVE_PATHS[@]}")"
  [[ "$actual" == "$expected" ]] || die "unexpected Metal library payload path/count in $app"
}

verify_metallib_archive_binding() {
  local source_app="$1" extracted_app="$2" relative source_hash extracted_hash
  verify_metallib_data "$source_app"
  verify_metallib_data "$extracted_app"
  for relative in "${METALLIB_RELATIVE_PATHS[@]}"; do
    source_hash="$(sha256_file "$source_app/$relative")"
    extracted_hash="$(sha256_file "$extracted_app/$relative")"
    [[ "$source_hash" == "$extracted_hash" ]] || die "Metal library payload hash mismatch after archive extraction: $relative"
  done
}

die() {
  printf 'notarization: %s\n' "$*" >&2
  exit 1
}

status() {
  printf 'notarization: %s\n' "$*" >&2
}

cleanup() {
  local path
  for path in $TEMP_PATHS; do
    rm -rf "$path"
  done
}
trap cleanup EXIT HUP INT TERM

safe_component() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/-/g'
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

duration_seconds() {
  local duration="$1" value
  value="${duration%[smh]}"
  case "$duration" in
    *s) printf '%s\n' "$value" ;;
    *m) printf '%s\n' "$((10#$value * 60))" ;;
    *h) printf '%s\n' "$((10#$value * 3600))" ;;
    *) printf '%s\n' "$duration" ;;
  esac
}

plist_get_file() {
  local file="$1" key="$2"
  plutil -extract "$key" raw -o - "$file" 2>/dev/null || true
}

state_get() {
  plist_get_file "$STATE_FILE" "$1"
}

state_set() {
  local key="$1" value="$2"
  if ! plutil -replace "$key" -string "$value" "$STATE_FILE" 2>/dev/null; then
    plutil -insert "$key" -string "$value" "$STATE_FILE"
  fi
}

state_initialize() {
  mkdir -p "$STATE_DIR"
  plutil -create xml1 "$STATE_FILE"
  state_set releaseRevision "$RELEASE_REVISION"
  state_set releaseVersion "$RELEASE_VERSION"
  state_set appVersion "$APP_VERSION"
  state_set appBuildVersion "$APP_BUILD_VERSION"
  state_set notaryDeveloperDir "${NOTARY_DEVELOPER_DIR:-Active Xcode developer directory}"
  state_set notarytoolPath "$NOTARYTOOL"
  state_set notarytoolVersion "$NOTARYTOOL_VERSION"
  case "$(printf '%s' "$NO_S3_ACCELERATION" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) state_set s3Acceleration Disabled ;;
    *) state_set s3Acceleration Enabled ;;
  esac
}

state_validate_release() {
  local actual
  actual="$(state_get releaseRevision)"
  [[ "$actual" == "$RELEASE_REVISION" ]] || die "release revision mismatch: state has '$actual', current release is '$RELEASE_REVISION'"
  actual="$(state_get releaseVersion)"
  [[ "$actual" == "$RELEASE_VERSION" ]] || die "release version mismatch: state has '$actual', current release is '$RELEASE_VERSION'"
  actual="$(state_get appVersion)"
  [[ "$actual" == "$APP_VERSION" ]] || die "app version mismatch: state has '$actual', expected '$APP_VERSION'"
  actual="$(state_get appBuildVersion)"
  [[ "$actual" == "$APP_BUILD_VERSION" ]] || die "app build version mismatch: state has '$actual', expected '$APP_BUILD_VERSION'"
}

credential_args() {
  CREDENTIAL_ARGS=(
    --apple-id "$APPLE_ID"
    --password "$APPLE_PASSWORD"
    --team-id "$APPLE_TEAM_ID"
  )
}

extract_submission_id() {
  local file="$1" value
  value="$(plist_get_file "$file" id)"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
    return
  fi
  grep -Eo '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' "$file" | head -n 1 || true
}

parse_info_file() {
  local file="$1"
  INFO_STATUS="$(plist_get_file "$file" status)"
  INFO_NAME="$(plist_get_file "$file" name)"
  INFO_CREATED_DATE="$(plist_get_file "$file" createdDate)"

  if [[ -z "$INFO_STATUS" ]]; then
    INFO_STATUS="$(sed -nE 's/^[[:space:]]*status:[[:space:]]*([^[:space:]].*)$/\1/p' "$file" | head -n 1 | tr -d '"')"
  fi
  if [[ -z "$INFO_NAME" ]]; then
    INFO_NAME="$(sed -nE 's/^[[:space:]]*name:[[:space:]]*([^[:space:]].*)$/\1/p' "$file" | head -n 1 | tr -d '"')"
  fi
  if [[ -z "$INFO_CREATED_DATE" ]]; then
    INFO_CREATED_DATE="$(sed -nE 's/^[[:space:]]*createdDate:[[:space:]]*([^[:space:]].*)$/\1/p' "$file" | head -n 1 | tr -d '"')"
  fi
}

query_info() {
  local submission_id="$1" stage="$2" output rc=0
  output="$(mktemp "${TMPDIR:-/tmp}/ollama-notary-info.XXXXXX")"
  TEMP_PATHS="$TEMP_PATHS $output"
  "$NOTARYTOOL" info "$submission_id" --output-format plist "${CREDENTIAL_ARGS[@]}" >"$output" 2>&1 || rc=$?
  parse_info_file "$output"
  if [[ "$rc" -ne 0 || -z "$INFO_STATUS" ]]; then
    if [[ -f "$STATE_FILE" ]]; then
      state_set "${stage}LastStatus" "Unknown"
    fi
    return 1
  fi
  if [[ -f "$STATE_FILE" ]]; then
    state_set "${stage}SubmissionID" "$submission_id"
    state_set "${stage}SubmissionName" "$INFO_NAME"
    state_set "${stage}SubmissionCreatedDate" "$INFO_CREATED_DATE"
    state_set "${stage}LastStatus" "$INFO_STATUS"
  fi
  return 0
}

redact_file() {
  local file="$1"
  awk -v apple_id="$APPLE_ID" -v password="$APPLE_PASSWORD" -v team_id="$APPLE_TEAM_ID" '
    function replace_literal(text, needle, replacement, before, after) {
      if (needle == "") return text
      while ((before = index(text, needle)) != 0) {
        after = before + length(needle)
        text = substr(text, 1, before - 1) replacement substr(text, after)
      }
      return text
    }
    {
      line = replace_literal($0, apple_id, "<redacted>")
      line = replace_literal(line, password, "<redacted>")
      line = replace_literal(line, team_id, "<redacted>")
      gsub(/X-Amz-Credential=[^&[:space:]]+/, "X-Amz-Credential=<redacted>", line)
      gsub(/X-Amz-Security-Token=[^&[:space:]]+/, "X-Amz-Security-Token=<redacted>", line)
      gsub(/AWSAccessKeyId=[^&[:space:]]+/, "AWSAccessKeyId=<redacted>", line)
      if (line ~ /^[[:space:]]*Authorization:/) line = "Authorization: <redacted>"
      print line
    }
  ' "$file"
}

fetch_invalid_log() {
  local submission_id="$1" stage="$2" log_path rc=0
  log_path="$STATE_DIR/${stage}-notarytool.log"
  "$NOTARYTOOL" log "$submission_id" "$log_path" "${CREDENTIAL_ARGS[@]}" >/dev/null 2>&1 || rc=$?
  if [[ "$rc" -ne 0 || ! -s "$log_path" ]]; then
    status "Apple marked submission $submission_id Invalid, but its log could not be retrieved"
    return
  fi
  local redacted
  redacted="$(mktemp "${TMPDIR:-/tmp}/ollama-notary-log.XXXXXX")"
  TEMP_PATHS="$TEMP_PATHS $redacted"
  redact_file "$log_path" >"$redacted"
  mv "$redacted" "$log_path"
  status "Apple notarization log: $log_path"
  cat "$log_path" >&2
}

report_state() {
  local stage="$1" submission_id="$2" artifact hash status_value upload_status
  [[ "$stage" == app || "$stage" == dmg ]] || die "invalid local preflight state: non-submission stage '$stage' has no resumable Apple submission"
  [[ -n "$submission_id" ]] || die "invalid local preflight state: $stage stage is missing its Apple submission ID"
  artifact="$(state_get "${stage}ArtifactPath")"
  hash="$(state_get "${stage}ArtifactSHA256")"
  status_value="$(state_get "${stage}LastStatus")"
  upload_status="$(state_get "${stage}UploadStatus")"
  status "submission ID: $submission_id"
  status "stage: $stage"
  status "artifact: $artifact"
  status "SHA-256: $hash"
  status "Apple status: ${status_value:-Unknown}"
  status "S3 upload: ${upload_status:-Unknown (legacy submission without verbose evidence)}"
  if [[ "$upload_status" == Unconfirmed ]]; then
    status "replacement: ./scripts/mlx_build.sh --signed-package --replace-notarization $submission_id --notary-no-s3-acceleration --notary-timeout $TIMEOUT"
  else
    status "resume: ./scripts/mlx_build.sh --resume-notarization $submission_id --notary-timeout $TIMEOUT"
  fi
}

handle_status() {
  local stage="$1" submission_id="$2"
  case "$INFO_STATUS" in
    Accepted)
      return 0
      ;;
    "In Progress")
      report_state "$stage" "$submission_id"
      return "$EX_TEMPFAIL"
      ;;
    Invalid|Rejected)
      report_state "$stage" "$submission_id"
      fetch_invalid_log "$submission_id" "$stage"
      return 1
      ;;
    *)
      state_set "${stage}LastStatus" "${INFO_STATUS:-Unknown}"
      report_state "$stage" "$submission_id"
      return "$EX_TEMPFAIL"
      ;;
  esac
}

verify_artifact_hash() {
  local stage="$1" artifact expected actual
  artifact="$(state_get "${stage}ArtifactPath")"
  expected="$(state_get "${stage}ArtifactSHA256")"
  [[ -f "$artifact" ]] || die "recorded $stage artifact is missing: $artifact"
  actual="$(sha256_file "$artifact")"
  [[ "$actual" == "$expected" ]] || die "$stage artifact hash mismatch: expected $expected, got $actual"
}

verify_app_bundle() {
  local app="$1" actual
  [[ -d "$app" ]] || die "Ollama.app is missing: $app"
  codesign --verify --deep --strict --verbose=2 "$app"
  codesign -d --verbose=4 "$app" 2>&1 | grep -F 'Authority=Developer ID Application:' >/dev/null || die "Ollama.app is not signed with a Developer ID Application identity: $app"
  local target
  while IFS= read -r target; do require_release_signature "$target"; done < <(
    printf '%s\n' "$app" "$app/Contents/Resources/ollama" "$app/Contents/Resources/llama-server" "$app/Contents/Resources/llama-quantize" "$app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel" "$app/Contents/Frameworks/Squirrel.framework" "$app/Contents/MacOS/Ollama"
    find "$app/Contents/Resources" -type f \( -name '*.so' -o -name '*.dylib' \) -print | sort
  )
  verify_metallib_data "$app"
  actual="$(plist_get_file "$app/Contents/Info.plist" CFBundleShortVersionString)"
  [[ "$actual" == "$APP_VERSION" ]] || die "Ollama.app version mismatch: expected '$APP_VERSION', got '$actual'"
  actual="$(plist_get_file "$app/Contents/Info.plist" CFBundleVersion)"
  [[ "$actual" == "$APP_BUILD_VERSION" ]] || die "Ollama.app build version mismatch: expected '$APP_BUILD_VERSION', got '$actual'"
}

verify_signing_bindings() {
  [[ -f "$SIGNING_RECEIPT" && -f "$PRE_SIGN_BUNDLE/status.json" && -f "$PRE_SIGN_BUNDLE/gate-inputs.json" ]] || die "missing signed receipt or pre-sign bundle"
  jq -e '(.equivalent == true) and ((.differences|length) == 0) and (.developer_id_authority|startswith("Developer ID Application:"))' "$SIGNING_RECEIPT" >/dev/null || die "signed receipt is not equivalent Developer-ID evidence"
  jq -e '.status == "pre_sign_pass" and .acceptance == "pre_sign_only"' "$PRE_SIGN_BUNDLE/status.json" >/dev/null || die "pre-sign status is not pass"
  [[ "$(shasum -a 256 "$PRE_SIGN_BUNDLE/gate-inputs.json" | awk '{print $1}')" == "$(jq -r .gate_inputs_file_sha256 "$SIGNING_RECEIPT")" ]] || die "pre-sign inputs hash mismatch"
  [[ "$(shasum -a 256 "$PRE_SIGN_BUNDLE/status.json" | awk '{print $1}')" == "$(jq -r .gate_status_file_sha256 "$SIGNING_RECEIPT")" ]] || die "pre-sign status hash mismatch"
}

fresh_archive_preflight() {
  local output="$1" app="$DIST_DIR/Ollama.app" extract extracted
  [[ ! -e "$output" ]] || die "refusing existing fresh archive path: $output"
  verify_signing_bindings
  verify_app_bundle "$app"
  ditto -c -k --norsrc --keepParent "$app" "$output"
  [[ -s "$output" && "$output" -nt "$app" ]] || die "fresh archive was not created after current app"
  extract="$(mktemp -d "${TMPDIR:-/tmp}/ollama-notary-fresh.XXXXXX")"; TEMP_PATHS="$TEMP_PATHS $extract"
  ditto -x -k "$output" "$extract"; extracted="$extract/Ollama.app"
  verify_app_bundle "$extracted"
  verify_metallib_archive_binding "$app" "$extracted"
  [[ "$(signature_identity "$app")" == "$(signature_identity "$extracted")" ]] || die "fresh archive signature identity mismatch"
  printf '%s\n' "fresh archive SHA-256: $(sha256_file "$output")"
}

signature_identity() {
  codesign -d --verbose=4 "$1" 2>&1 | sed -nE '/^(Identifier|TeamIdentifier|CDHash|Designated Requirement)=/p'
}

verify_bootstrap_zip() {
  local zip="$1" app="$2" extract_dir extracted_app local_identity extracted_identity
  verify_app_bundle "$app"
  extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/ollama-notary-bootstrap.XXXXXX")"
  TEMP_PATHS="$TEMP_PATHS $extract_dir"
  ditto -x -k "$zip" "$extract_dir"
  extracted_app="$extract_dir/Ollama.app"
  verify_app_bundle "$extracted_app"
  local_identity="$(signature_identity "$app")"
  extracted_identity="$(signature_identity "$extracted_app")"
  [[ -n "$local_identity" && "$local_identity" == "$extracted_identity" ]] || die "preserved ZIP and Ollama.app do not have matching sealed bundle identities"
}

preserve_artifact() {
  local source="$1" destination="$2" stage="$3"
  cp -f "$source" "$destination"
  state_set activeStage "$stage"
  state_set "${stage}ArtifactPath" "$destination"
  state_set "${stage}ArtifactSHA256" "$(sha256_file "$destination")"
}

wait_for_submission() {
  local stage="$1" submission_id="$2" wait_timeout="${3:-$TIMEOUT}" output rc=0
  output="$(mktemp "${TMPDIR:-/tmp}/ollama-notary-wait.XXXXXX")"
  TEMP_PATHS="$TEMP_PATHS $output"
  "$NOTARYTOOL" wait "$submission_id" --timeout "$wait_timeout" --output-format plist "${CREDENTIAL_ARGS[@]}" >"$output" 2>&1 || rc=$?
  if ! query_info "$submission_id" "$stage"; then
    INFO_STATUS="Unknown"
  fi
  if [[ "$rc" -ne 0 && "$INFO_STATUS" == Accepted ]]; then
    report_state "$stage" "$submission_id"
  fi
  handle_status "$stage" "$submission_id"
}

submit_artifact() {
  local stage="$1" artifact="$2" output submit_log upload_status rc=0 submission_id started_at finished_at elapsed timeout_seconds remaining
  output="$(mktemp "${TMPDIR:-/tmp}/ollama-notary-submit.XXXXXX")"
  TEMP_PATHS="$TEMP_PATHS $output"
  submit_log="$STATE_DIR/${stage}-submit-verbose.log"
  state_set "${stage}NotarytoolPath" "$NOTARYTOOL"
  state_set "${stage}NotarytoolVersion" "$NOTARYTOOL_VERSION"
  status "starting Apple $stage submission with notarytool $NOTARYTOOL_VERSION at $NOTARYTOOL (timeout $TIMEOUT)"
  started_at="$(date +%s)"
  case "$(printf '%s' "$NO_S3_ACCELERATION" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      "$NOTARYTOOL" submit "$artifact" --verbose --no-s3-acceleration --wait --timeout "$TIMEOUT" --output-format plist "${CREDENTIAL_ARGS[@]}" >"$output" 2>&1 || rc=$?
      ;;
    *)
      "$NOTARYTOOL" submit "$artifact" --verbose --wait --timeout "$TIMEOUT" --output-format plist "${CREDENTIAL_ARGS[@]}" >"$output" 2>&1 || rc=$?
      ;;
  esac
  finished_at="$(date +%s)"
  redact_file "$output" >"$submit_log"
  chmod 600 "$submit_log"
  if grep -F 'Received new upload status: Succeeded' "$submit_log" >/dev/null || \
      grep -F 'Multipart upload process has completed successfully.' "$submit_log" >/dev/null; then
    upload_status=Confirmed
    status "S3 upload confirmed by notarytool verbose log: $submit_log"
  else
    upload_status=Unconfirmed
    status "WARNING: S3 upload completion was not confirmed by the notarytool verbose log: $submit_log"
  fi
  state_set "${stage}UploadStatus" "$upload_status"
  state_set "${stage}SubmitVerboseLogPath" "$submit_log"
  submission_id="$(extract_submission_id "$output")"
  if [[ -z "$submission_id" ]]; then
    state_set "${stage}LastStatus" "Submission Failed Before ID"
    die "$stage submission failed before Apple returned a submission ID (notarytool exit $rc); no automatic retry will be attempted"
  fi
  state_set "${stage}SubmissionID" "$submission_id"
  if ! query_info "$submission_id" "$stage"; then
    INFO_STATUS="Unknown"
  fi
  if [[ "$upload_status" == Unconfirmed ]]; then
    report_state "$stage" "$submission_id"
    status "upload is incomplete; status polling cannot repair missing S3 parts"
    return 1
  fi
  if [[ "$INFO_STATUS" == "In Progress" ]]; then
    elapsed=$((finished_at - started_at))
    timeout_seconds="$(duration_seconds "$TIMEOUT")"
    remaining=$((timeout_seconds - elapsed))
    if [[ "$remaining" -gt 0 ]]; then
      status "Apple accepted the upload but submit returned while it was still In Progress; waiting up to ${remaining}s (the remainder of $TIMEOUT) for submission $submission_id"
      wait_for_submission "$stage" "$submission_id" "${remaining}s"
      return
    fi
  fi
  if [[ "$rc" -ne 0 && "$INFO_STATUS" == Accepted ]]; then
    report_state "$stage" "$submission_id"
  fi
  handle_status "$stage" "$submission_id"
}

staple_and_validate_app() {
  local app="$DIST_DIR/Ollama.app"
  verify_app_bundle "$app"
  "$STAPLER" staple "$app"
  "$STAPLER" validate "$app"
  rm -f "$DIST_DIR/Ollama-darwin.zip"
  ditto -c -k --norsrc --keepParent "$app" "$DIST_DIR/Ollama-darwin.zip"
}

create_and_sign_dmg() {
  local create_dmg_command="${MLX_CREATE_DMG_COMMAND:-$SCRIPT_DIR/create-dmg.sh}"
  rm -f "$DIST_DIR/Ollama.dmg"
  (
    cd "$DIST_DIR"
    "$create_dmg_command" \
      --volname "${VOL_NAME:-Ollama}" \
      --volicon Ollama.app/Contents/Resources/icon.icns \
      --background "$REPO_DIR/app/assets/background.png" \
      --window-pos 200 120 \
      --window-size 800 400 \
      --icon-size 128 \
      --icon Ollama.app 200 190 \
      --hide-extension Ollama.app \
      --app-drop-link 600 190 \
      --text-size 12 \
      Ollama.dmg Ollama.app
  )
  rm -f "$DIST_DIR"/rw*.dmg
  codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$DIST_DIR/Ollama.dmg"
}

continue_after_app_acceptance() {
  verify_artifact_hash app
  verify_bootstrap_zip "$(state_get appArtifactPath)" "$DIST_DIR/Ollama.app"
  staple_and_validate_app
  create_and_sign_dmg
  preserve_artifact "$DIST_DIR/Ollama.dmg" "$STATE_DIR/Ollama.submitted.dmg" dmg
  submit_artifact dmg "$(state_get dmgArtifactPath)"
  continue_after_dmg_acceptance
}

continue_after_dmg_acceptance() {
  verify_artifact_hash dmg
  cp -f "$(state_get dmgArtifactPath)" "$DIST_DIR/Ollama.dmg"
  codesign --verify --verbose=2 "$DIST_DIR/Ollama.dmg"
  codesign -d --verbose=4 "$DIST_DIR/Ollama.dmg" 2>&1 | grep -F 'Authority=Developer ID Application:' >/dev/null || die "Ollama.dmg is not signed with a Developer ID Application identity"
  "$STAPLER" staple "$DIST_DIR/Ollama.dmg"
  "$STAPLER" validate "$DIST_DIR/Ollama.dmg"
  state_set completedDmgPath "$DIST_DIR/Ollama.dmg"
  state_set completedDmgSHA256 "$(sha256_file "$DIST_DIR/Ollama.dmg")"
  state_set activeStage complete
  state_set completedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status "notarization complete: $DIST_DIR/Ollama.dmg"
}

bootstrap_app_state() {
  local submission_id="$1" source_zip="$DIST_DIR/Ollama-darwin.zip"
  [[ -f "$source_zip" ]] || die "state-free app recovery requires the preserved ZIP: $source_zip"
  [[ -d "$DIST_DIR/Ollama.app" ]] || die "state-free app recovery requires the preserved app: $DIST_DIR/Ollama.app"
  if ! query_info "$submission_id" app; then
    die "cannot bootstrap state because Apple submission $submission_id is unavailable"
  fi
  status "found Apple submission $submission_id: name=$INFO_NAME, stage=app, status=$INFO_STATUS"
  [[ "$INFO_NAME" == "Ollama-darwin.zip" ]] || die "state-free recovery only accepts an app submission named Ollama-darwin.zip; Apple reports '$INFO_NAME'"
  verify_bootstrap_zip "$source_zip" "$DIST_DIR/Ollama.app"
  state_initialize
  preserve_artifact "$source_zip" "$STATE_DIR/Ollama-darwin.submitted.zip" app
  state_set appSubmissionID "$submission_id"
  state_set appSubmissionName "$INFO_NAME"
  state_set appSubmissionCreatedDate "$INFO_CREATED_DATE"
  state_set appLastStatus "$INFO_STATUS"
  status "bootstrapped notarization state for app submission $submission_id"
}

resume_submission() {
  local submission_id="$1" stage recorded_id bootstrapped=0
  if [[ ! -f "$STATE_FILE" ]]; then
    bootstrap_app_state "$submission_id"
    bootstrapped=1
  else
    state_validate_release
  fi

  if [[ "$(state_get s3Acceleration)" == Disabled ]]; then
    NO_S3_ACCELERATION=1
  fi

  stage="$(state_get activeStage)"
  case "$stage" in
    app|dmg) ;;
    complete)
      die "notarization state is already complete"
      ;;
    *)
      die "state-free DMG recovery is not supported and the recorded active stage is invalid: '${stage:-missing}'"
      ;;
  esac
  recorded_id="$(state_get "${stage}SubmissionID")"
  [[ "$recorded_id" == "$submission_id" ]] || die "submission ID mismatch: active $stage state records '$recorded_id', not '$submission_id'"
  verify_artifact_hash "$stage"

  if [[ "$bootstrapped" == 0 ]]; then
    if ! query_info "$submission_id" "$stage"; then
      INFO_STATUS="Unknown"
      handle_status "$stage" "$submission_id"
    fi
    status "found recorded $stage submission $submission_id with Apple status: $INFO_STATUS"
  fi
  case "$INFO_STATUS" in
    "In Progress")
      status "waiting up to $TIMEOUT for Apple to finish submission $submission_id"
      wait_for_submission "$stage" "$submission_id"
      ;;
    *) handle_status "$stage" "$submission_id" ;;
  esac

  if [[ "$stage" == app ]]; then
    continue_after_app_acceptance
  else
    continue_after_dmg_acceptance
  fi
}

rotate_existing_state_for_new_build() {
  local stage last_status submission_id artifact hash upload_status rotated
  [[ -f "$STATE_FILE" ]] || return 0
  state_validate_release
  stage="$(state_get activeStage)"
  if [[ -z "$stage" ]]; then
    submission_id="$(state_get appSubmissionID)$(state_get dmgSubmissionID)"
    artifact="$(state_get appArtifactPath)$(state_get dmgArtifactPath)"
    hash="$(state_get appArtifactSHA256)$(state_get dmgArtifactSHA256)"
    [[ -z "$submission_id$artifact$hash" ]] || die "invalid local preflight state: empty stage has submission, artifact, or hash fields"
    state_set activeStage preflight
    state_set preflightStatus local_preflight_incomplete_legacy
    stage=preflight
  fi
  if [[ "$stage" == preflight ]]; then
    submission_id="$(state_get appSubmissionID)$(state_get dmgSubmissionID)"
    artifact="$(state_get appArtifactPath)$(state_get dmgArtifactPath)"
    hash="$(state_get appArtifactSHA256)$(state_get dmgArtifactSHA256)"
    upload_status="$(state_get appUploadStatus)$(state_get dmgUploadStatus)"
    [[ -z "$submission_id$artifact$hash$upload_status" ]] || die "invalid local preflight state: preflight state contains Apple submission, artifact, hash, or upload fields"
    rotated="$STATE_DIR.preflight.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv "$STATE_DIR" "$rotated"
    status "rotated non-submission preflight state to $rotated"
    return 0
  fi
  [[ "$stage" == app || "$stage" == dmg || "$stage" == complete ]] || die "invalid local preflight state: unknown active stage '$stage'"
  if [[ "$stage" == complete ]]; then
    last_status=Accepted
  else
    last_status="$(state_get "${stage}LastStatus")"
  fi
  submission_id="$(state_get "${stage}SubmissionID")"
  if [[ -n "$REPLACE_SUBMISSION_ID" ]]; then
    [[ -n "$submission_id" ]] || die "cannot replace notarization state because its active $stage stage has no submission ID"
    [[ "$submission_id" == "$REPLACE_SUBMISSION_ID" ]] || die "replacement submission ID mismatch: state records '$submission_id', not '$REPLACE_SUBMISSION_ID'"
    rotated="$STATE_DIR.replaced.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv "$STATE_DIR" "$rotated"
    status "explicitly replaced submission $submission_id; archived its state at $rotated"
    return 0
  fi
  case "$last_status" in
    Invalid|Rejected|Accepted|"Submission Failed Before ID")
      rotated="$STATE_DIR.previous.$(date -u +%Y%m%dT%H%M%SZ).$$"
      mv "$STATE_DIR" "$rotated"
      status "rotated terminal notarization state to $rotated"
      ;;
    *)
      report_state "$stage" "$submission_id"
      die "refusing to create a duplicate submission while matching state is nonterminal; resume the recorded submission"
      ;;
  esac
}

start_new_submission() {
  rotate_existing_state_for_new_build
  state_initialize
  state_set activeStage preflight
  state_set preflightStatus local_preflight_incomplete
  local source_zip="$STATE_DIR/Ollama-darwin.fresh.zip"
  fresh_archive_preflight "$source_zip"
  preserve_artifact "$source_zip" "$STATE_DIR/Ollama-darwin.submitted.zip" app
  submit_artifact app "$(state_get appArtifactPath)"
  continue_after_app_acceptance
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release-revision) RELEASE_REVISION="${2:-}"; shift 2 ;;
    --release-version) RELEASE_VERSION="${2:-}"; shift 2 ;;
    --app-version) APP_VERSION="${2:-}"; shift 2 ;;
    --app-build-version) APP_BUILD_VERSION="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --resume) RESUME_ID="${2:-}"; shift 2 ;;
    --preflight-archive) PREFLIGHT_ARCHIVE=1; shift ;;
    --artifact-dir) PREFLIGHT_DIR="${2:-}"; shift 2 ;;
    --signing-receipt) SIGNING_RECEIPT="${2:-}"; shift 2 ;;
    --pre-sign-bundle) PRE_SIGN_BUNDLE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == Darwin || "${MLX_NOTARIZATION_TESTING:-0}" == 1 ]] || die "Darwin notarization must run on macOS"
[[ -n "$RELEASE_REVISION" && -n "$RELEASE_VERSION" && -n "$APP_VERSION" && -n "$APP_BUILD_VERSION" ]] || { usage >&2; die "release and app identity arguments are required"; }
[[ -n "$TIMEOUT" ]] || die "notary timeout must not be empty"
if [[ "$PREFLIGHT_ARCHIVE" == 1 ]]; then
  [[ -n "$PREFLIGHT_DIR" && -n "$SIGNING_RECEIPT" && -n "$PRE_SIGN_BUNDLE" ]] || die "preflight requires artifact dir, signed receipt, and pre-sign bundle"
  case "$PREFLIGHT_DIR" in /*) ;; *) PREFLIGHT_DIR="$REPO_DIR/../$PREFLIGHT_DIR";; esac
  case "$SIGNING_RECEIPT" in /*) ;; *) SIGNING_RECEIPT="$REPO_DIR/../$SIGNING_RECEIPT";; esac
  case "$PRE_SIGN_BUNDLE" in /*) ;; *) PRE_SIGN_BUNDLE="$REPO_DIR/../$PRE_SIGN_BUNDLE";; esac
  if [[ "${MLX_NOTARIZATION_TESTING:-0}" == 1 ]]; then
    [[ ! -e "$PREFLIGHT_DIR" ]] || die "preflight artifact dir must be new"
  else
    [[ "$PREFLIGHT_DIR" == "$REPO_DIR/../tmp/"* && ! -e "$PREFLIGHT_DIR" ]] || die "preflight artifact dir must be new below tmp"
  fi
  mkdir -p "$PREFLIGHT_DIR"; fresh_archive_preflight "$PREFLIGHT_DIR/Ollama-darwin.fresh.zip"
  exit 0
fi
for name in APPLE_IDENTITY APPLE_ID APPLE_TEAM_ID APPLE_PASSWORD; do
  [[ -n "${!name:-}" ]] || die "missing macOS signing environment variable: $name"
done
for tool in plutil shasum codesign ditto xcrun; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done

STATE_DIR="$STATE_ROOT/$(safe_component "$RELEASE_VERSION")"
STATE_FILE="$STATE_DIR/state.plist"
if [[ -z "$NOTARY_DEVELOPER_DIR" && -f "$STATE_FILE" ]]; then
  stored_developer_dir="$(plist_get_file "$STATE_FILE" notaryDeveloperDir)"
  if [[ -n "$stored_developer_dir" && "$stored_developer_dir" != "Active Xcode developer directory" ]]; then
    NOTARY_DEVELOPER_DIR="$stored_developer_dir"
  fi
fi
if [[ -n "$NOTARY_DEVELOPER_DIR" ]]; then
  [[ -d "$NOTARY_DEVELOPER_DIR" ]] || die "notary developer directory does not exist: $NOTARY_DEVELOPER_DIR"
  NOTARYTOOL="$(DEVELOPER_DIR="$NOTARY_DEVELOPER_DIR" xcrun -f notarytool)"
else
  NOTARYTOOL="$(xcrun -f notarytool)"
fi
[[ -x "$NOTARYTOOL" ]] || die "notarytool is not executable: $NOTARYTOOL"
NOTARYTOOL_VERSION="$($NOTARYTOOL --version)"
STAPLER="$(xcrun -f stapler)"
credential_args

if [[ -n "$RESUME_ID" ]]; then
  resume_submission "$RESUME_ID"
else
  start_new_submission
fi

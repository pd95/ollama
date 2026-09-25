# Desktop update sources

The desktop updater keeps discovery, staging, and verification provenance in
each update candidate. The normal Ollama service is the `official` source. MLX
distribution builds use `mlx-preview` as their primary source while retaining
the official discovery implementation for a separate, manual replacement
action in Settings. Official candidates never inherit the MLX signing policy.

## MLX preview manifest v1

The fixed discovery URL is:

```text
https://pd95.github.io/Ollama/updates/v1/preview/darwin-arm64.json
```

It serves one strict JSON object. Unknown fields, redirects outside the Pages
host, bodies larger than 64 KiB, and values that do not match the build policy
are rejected.

```json
{
  "schema_version": 1,
  "source": "mlx-preview",
  "channel": "preview",
  "repository": "pd95/ollama",
  "marketing_version": "0.34.2",
  "build_version": "34.2.1",
  "release_tag": "preview/mlx-0.34.2-20260919",
  "release_page_url": "https://github.com/pd95/ollama/releases/tag/preview%2Fmlx-0.34.2-20260919",
  "platform": "darwin",
  "architecture": "arm64",
  "asset": {
    "name": "Ollama-darwin.zip",
    "url": "https://github.com/pd95/ollama/releases/download/preview%2Fmlx-0.34.2-20260919/Ollama-darwin.zip",
    "size_bytes": 123456789,
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
}
```

Marketing versions contain exactly three numeric components. Build versions
use `<major*1000+minor>.<patch>.<distribution-revision>`; the distribution
revision starts at `1` and increases for another build of the same Ollama
version. A candidate must be newer by marketing version or, when marketing
versions match, by build version. Same-version updates and downgrades are
ignored.

The asset URL and release page must encode the exact declared release tag. The
download may redirect only from `github.com` to
`release-assets.githubusercontent.com`. The declared byte size and SHA-256 are
mandatory.

## Installation policy

The MLX automatic-update preference permits background discovery and download.
Installation always requires the user to choose **Restart to update**. Hidden
startup does not install a staged MLX update.

Before installation, the updater repeats checksum and archive validation, then
requires bundle identifier `com.electron.ollama`, matching marketing/build
versions, an arm64 executable, Apple Developer ID Team `P8CC95REUG`, and a
successful Gatekeeper assessment. A valid signature from any other signer is
not accepted.

## Official Ollama replacement policy

MLX builds expose official Ollama only as a manual replacement action. It is
never checked, downloaded, or installed by background update work. The user is
warned that replacement removes MLX-only behavior and that returning to MLX
requires a manual MLX installation.

Official discovery uses the Ollama update service, but a candidate is accepted
only when it is a strictly newer marketing version and its initial URL is the
exact arm64 macOS archive under `github.com/ollama/ollama/releases/download`.
Downloads may redirect only to `release-assets.githubusercontent.com`.

The official signing identity was independently sampled from the notarized
arm64 DMGs for v0.34.1 and v0.34.2. Both packages use bundle identifier
`com.electron.ollama`, Apple Developer ID Team `3MU9H2V9Y9`, and authority
`Developer ID Application: Infra Technologies, Inc (3MU9H2V9Y9)`. The sampled
DMG SHA-256 values were respectively
`a040adcda7fc8a278cb6a6360aa9d5e17def35097a83cbffa67e63cefd1814f6`
and `ca3c5c1587fe05eebedacd33ec9448e7999fcce9c19dafcb4ca7eca2f89d5d7b`.

Installation requires matching bundle marketing/build versions, arm64 code,
the pinned official bundle and Team identity, strict nested-signature
validation, and Gatekeeper acceptance. The MLX Team ID is not an allowed
substitute for the official Team ID.

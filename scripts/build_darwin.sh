#!/bin/sh

# Note:
#  While testing, if you double-click on the Ollama.app
#  some state is left on MacOS and subsequent attempts
#  to build again will fail with:
#
#    hdiutil: create failed - Operation not permitted
#
#  To work around, specify another volume name with:
#
#    VOL_NAME="$(date)" ./scripts/build_darwin.sh
#
VOL_NAME=${VOL_NAME:-"Ollama"}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"

if [ -z "${VERSION:-}" ]; then
    VERSION=$(git describe --tags --first-parent --abbrev=7 --long --dirty --always 2>/dev/null | sed -e "s/^v//g" || true)
    VERSION=${VERSION:-0.0.0-local}
fi
export VERSION
export OLLAMA_DISABLE_UPDATES=${OLLAMA_DISABLE_UPDATES:-1}
export CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_CXXFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_LDFLAGS="-mmacosx-version-min=14.0"

set -e

status() { echo >&2 ">>> $@"; }
truthy() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|t|true|y|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}
usage() {
    echo "usage: $(basename $0) [build package app sign]"
    exit 1
}

mkdir -p dist

ARCHS="arm64 amd64"
while getopts "a:h" OPTION; do
    case $OPTION in
        a) ARCHS=$OPTARG ;;
        h) usage ;;
    esac
done

shift $(( $OPTIND - 1 ))

_build_darwin() {
    BUILD_CPUS=$(getconf _NPROCESSORS_ONLN)
    BUILD_JOBS=${OLLAMA_BUILD_PARALLEL:-$BUILD_CPUS}
    BUILD_LOAD=${OLLAMA_BUILD_LOAD:-$BUILD_CPUS}
    status "Build version: $VERSION"
    status "Build parallelism: $BUILD_JOBS, load limit: $BUILD_LOAD"

    SOURCE_BUILD=build/darwin-sources
    status "Preparing shared native sources"
    cmake -S . -B "$SOURCE_BUILD" -DOLLAMA_MLX_BACKENDS=metal_v3 -DOLLAMA_LLAMA_BACKENDS=
    cmake --build "$SOURCE_BUILD" --target ollama-llama-cpp-source --target ollama-mlx-sources
    LLAMA_CPP_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/llama_cpp-src"
    MLX_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/mlx-src"
    MLX_C_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/mlx-c-src"

    for ARCH in $ARCHS; do
        status "Building darwin $ARCH"
        INSTALL_PREFIX=dist/darwin-$ARCH/
        BUILD_DIR=build/darwin-$ARCH

        if [ "$ARCH" = "amd64" ]; then
            CMAKE_ARCH=x86_64
            MLX_BACKENDS=metal_v3
            MLX_EXTRA_ARGS="-DMLX_ENABLE_X64_MAC=ON"
            MLX_CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
            MLX_CGO_LDFLAGS="-ldl -lc++ -framework Accelerate -mmacosx-version-min=14.0"
        else
            CMAKE_ARCH=arm64
            MLX_BACKENDS="metal_v3;metal_v4"
            MLX_EXTRA_ARGS=
            MLX_CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
            MLX_CGO_LDFLAGS="-lc++ -framework Metal -framework Foundation -framework Accelerate -mmacosx-version-min=14.0"
        fi

        cmake -S . -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_OSX_ARCHITECTURES=$CMAKE_ARCH \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
            -DOLLAMA_PAYLOAD_INSTALL_PREFIX=$INSTALL_PREFIX \
            -DOLLAMA_GO_OUTPUT=$INSTALL_PREFIX/ollama \
            -DOLLAMA_VERSION="$VERSION" \
            -DOLLAMA_MLX_BACKENDS="$MLX_BACKENDS" \
            -DOLLAMA_LLAMA_BACKENDS= \
            -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$LLAMA_CPP_SHARED_SRC \
            -DFETCHCONTENT_SOURCE_DIR_MLX=$MLX_SHARED_SRC \
            -DFETCHCONTENT_SOURCE_DIR_MLX-C=$MLX_C_SHARED_SRC \
            $MLX_EXTRA_ARGS

        GOOS=darwin GOARCH=$ARCH CGO_ENABLED=1 CGO_CFLAGS="$MLX_CGO_CFLAGS" CGO_LDFLAGS="$MLX_CGO_LDFLAGS" \
            cmake --build "$BUILD_DIR" --target ollama-local --target ollama-mlx-backends --parallel "$BUILD_JOBS" -- -l "$BUILD_LOAD"
    done
}

_merge_darwin_payload() {
    status "Preparing Darwin runtime payload"
    rm -rf dist/darwin/lib
    mkdir -p dist/darwin/lib/ollama

    for ROOT in dist/darwin-amd64/lib/ollama dist/darwin-arm64/lib/ollama; do
        [ -d "$ROOT" ] || continue
        for F in "$ROOT"/*; do
            [ -e "$F" ] || continue
            BASE=$(basename "$F")
            case "$BASE" in
                llama-server|llama-quantize|mlx_*) continue ;;
            esac
            [ -e "dist/darwin/lib/ollama/$BASE" ] || cp -P "$F" dist/darwin/lib/ollama/
        done
    done

    for VARIANT in dist/darwin-arm64/lib/ollama/mlx_metal_v*/; do
        [ -d "$VARIANT" ] || continue
        VNAME=$(basename "$VARIANT")
        DEST=dist/darwin/lib/ollama/$VNAME
        AMD_VARIANT=dist/darwin-amd64/lib/ollama/$VNAME
        [ -d "$AMD_VARIANT" ] || AMD_VARIANT=dist/darwin-amd64/lib/ollama
        mkdir -p "$DEST"

        for LIB in libmlx.dylib libmlxc.dylib libollama_xgrammar.dylib; do
            if [ -f "$AMD_VARIANT/$LIB" ] && [ -f "$VARIANT$LIB" ]; then
                lipo -create -output "$DEST/$LIB" "$AMD_VARIANT/$LIB" "$VARIANT$LIB"
            elif [ -f "$VARIANT$LIB" ]; then
                cp "$VARIANT$LIB" "$DEST/"
            elif [ -f "$AMD_VARIANT/$LIB" ]; then
                cp "$AMD_VARIANT/$LIB" "$DEST/"
            fi
        done

        for F in "$VARIANT"*; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            case "$(basename "$F")" in
                libmlx.dylib|libmlxc.dylib|libollama_xgrammar.dylib) continue ;;
            esac
            cp "$F" "$DEST/"
        done
    done
}

_prepare_darwin_runtime() {
    status "Preparing Darwin runtime binaries"
    mkdir -p dist/darwin
    _create_available_darwin_binary dist/darwin/ollama dist/darwin-amd64/ollama dist/darwin-arm64/ollama
    _create_available_darwin_binary dist/darwin/llama-server dist/darwin-amd64/lib/ollama/llama-server dist/darwin-arm64/lib/ollama/llama-server
    _create_available_darwin_binary dist/darwin/llama-quantize dist/darwin-amd64/lib/ollama/llama-quantize dist/darwin-arm64/lib/ollama/llama-quantize
    _merge_darwin_payload
}

_create_available_darwin_binary() {
    OUT=$1
    shift

    INPUTS=
    for F in "$@"; do
        [ -f "$F" ] || continue
        INPUTS="$INPUTS $F"
    done

    if [ -z "$INPUTS" ]; then
        echo "missing Darwin runtime input for $OUT" >&2
        exit 1
    fi

    # shellcheck disable=SC2086
    set -- $INPUTS
    if [ "$#" -gt 1 ]; then
        lipo -create -output "$OUT" "$@"
    else
        cp "$1" "$OUT"
    fi
    chmod +x "$OUT"
}

_prepare_darwin_app_runtime() {
    status "Preparing Darwin runtime payload for app bundle"
    mkdir -p dist/darwin
    _create_available_darwin_binary dist/darwin/ollama dist/darwin-amd64/ollama dist/darwin-arm64/ollama
    _create_available_darwin_binary dist/darwin/llama-server dist/darwin-amd64/lib/ollama/llama-server dist/darwin-arm64/lib/ollama/llama-server
    _create_available_darwin_binary dist/darwin/llama-quantize dist/darwin-amd64/lib/ollama/llama-quantize dist/darwin-arm64/lib/ollama/llama-quantize
    _merge_darwin_payload
}

_build_darwin_app_launcher() {
    APP_INPUTS=

    for ARCH in $ARCHS; do
        case "$ARCH" in
            arm64|amd64) ;;
            *)
                echo "unsupported Darwin app architecture: $ARCH" >&2
                exit 1
                ;;
        esac

        OUT="dist/darwin-app-$ARCH"
        GOARCH=$ARCH CGO_ENABLED=1 GOOS=darwin go build -o "$OUT" -ldflags="$APP_LDFLAGS" ./app/cmd/app
        APP_INPUTS="$APP_INPUTS $OUT"
    done

    mkdir -p dist/Ollama.app/Contents/MacOS
    # shellcheck disable=SC2086
    set -- $APP_INPUTS
    if [ "$#" -gt 1 ]; then
        lipo -create -output dist/Ollama.app/Contents/MacOS/Ollama "$@"
    else
        cp "$1" dist/Ollama.app/Contents/MacOS/Ollama
    fi
    chmod +x dist/Ollama.app/Contents/MacOS/Ollama
    rm -f dist/darwin-app-amd64 dist/darwin-app-arm64
}

_codesign_one() {
    IDENTITY=$1
    IDENTIFIER=$2
    TARGET=$3

    if [ "$IDENTITY" = "-" ]; then
        codesign -f -s - --identifier "$IDENTIFIER" "$TARGET"
    else
        codesign -f --timestamp -s "$IDENTITY" --identifier "$IDENTIFIER" --options=runtime "$TARGET"
    fi
}

_sign_app_bundle() {
    IDENTITY=$1

    _codesign_one "$IDENTITY" ai.ollama.ollama dist/Ollama.app/Contents/Resources/ollama
    _codesign_one "$IDENTITY" ai.ollama.ollama dist/Ollama.app/Contents/Resources/llama-server
    _codesign_one "$IDENTITY" ai.ollama.ollama dist/Ollama.app/Contents/Resources/llama-quantize

    for lib in dist/Ollama.app/Contents/Resources/*.so dist/Ollama.app/Contents/Resources/*.dylib dist/Ollama.app/Contents/Resources/*.metallib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.dylib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.metallib dist/Ollama.app/Contents/Resources/mlx_metal_v*/*.so; do
        [ -f "$lib" ] || continue
        _codesign_one "$IDENTITY" ai.ollama.ollama "$lib"
    done

    _codesign_one "$IDENTITY" com.electron.ollama dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
    _codesign_one "$IDENTITY" com.github.Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework
    _codesign_one "$IDENTITY" com.electron.ollama dist/Ollama.app/Contents/MacOS/Ollama
    if [ "$IDENTITY" = "-" ]; then
        codesign -f -s - --deep dist/Ollama.app
    else
        codesign -f --timestamp -s "$IDENTITY" --identifier com.electron.ollama --deep --options=runtime dist/Ollama.app
    fi
}

_prepare_local_app_for_launch() {
    xattr -cr dist/Ollama.app

    if xattr -lr dist/Ollama.app 2>/dev/null | grep -q com.apple.quarantine; then
        echo "failed to remove quarantine metadata from dist/Ollama.app" >&2
        exit 1
    fi

    codesign --verify --deep --strict --verbose=2 dist/Ollama.app
}

_create_darwin_runtime_tarball() {
    status "Creating universal tarball..."
    rm -f dist/ollama-darwin.tar dist/ollama-darwin.tgz
    tar -cf dist/ollama-darwin.tar --strip-components 2 dist/darwin/ollama dist/darwin/llama-server dist/darwin/llama-quantize
    tar -rf dist/ollama-darwin.tar --strip-components 4 dist/darwin/lib/ollama
    gzip -9vc <dist/ollama-darwin.tar >dist/ollama-darwin.tgz
}

_package_darwin_runtime() {
    _prepare_darwin_runtime
    _create_darwin_runtime_tarball
}

_sign_darwin() {
    _prepare_darwin_runtime
    if [ -n "$APPLE_IDENTITY" ]; then
        for F in dist/darwin/ollama dist/darwin/llama-server dist/darwin/llama-quantize dist/darwin/lib/ollama/* dist/darwin/lib/ollama/mlx_metal_v*/*; do
            [ -f "$F" ] && [ ! -L "$F" ] || continue
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$F"
        done

        # create a temporary zip for notarization
        TEMP=$(mktemp -u).zip
        ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
        xcrun notarytool submit "$TEMP" --wait --timeout 20m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
        rm -f "$TEMP"
    fi

    _create_darwin_runtime_tarball
}

_build_custom_app_icon() {
    if [ -n "${OLLAMA_APP_ICON_PNG:-}" ]; then
        APP_ICON_PNG=$OLLAMA_APP_ICON_PNG
    elif [ -f ../release-test-site/AppIcon-1024.png ]; then
        APP_ICON_PNG=../release-test-site/AppIcon-1024.png
    elif [ -f ../release-test-site/AppIcon.png ]; then
        APP_ICON_PNG=../release-test-site/AppIcon.png
    elif [ -f ../inspiration/Ollama/AppIcon-1024.png ]; then
        APP_ICON_PNG=../inspiration/Ollama/AppIcon-1024.png
    else
        APP_ICON_PNG=../inspiration/Ollama/AppIcon.png
    fi

    if [ ! -f "$APP_ICON_PNG" ]; then
        return
    fi

    if ! command -v sips >/dev/null 2>&1 || ! command -v iconutil >/dev/null 2>&1; then
        echo "custom app icon requires macOS sips and iconutil: $APP_ICON_PNG" >&2
        exit 1
    fi

    status "Building custom app icon from $APP_ICON_PNG"
    ICONSET=dist/OllamaAppIcon.iconset
    rm -rf "$ICONSET"
    mkdir -p "$ICONSET"

    sips -z 16 16 "$APP_ICON_PNG" --out "$ICONSET/icon_16x16.png" >/dev/null
    sips -z 32 32 "$APP_ICON_PNG" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
    sips -z 32 32 "$APP_ICON_PNG" --out "$ICONSET/icon_32x32.png" >/dev/null
    sips -z 64 64 "$APP_ICON_PNG" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
    sips -z 128 128 "$APP_ICON_PNG" --out "$ICONSET/icon_128x128.png" >/dev/null
    sips -z 256 256 "$APP_ICON_PNG" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
    sips -z 256 256 "$APP_ICON_PNG" --out "$ICONSET/icon_256x256.png" >/dev/null
    sips -z 512 512 "$APP_ICON_PNG" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
    sips -z 512 512 "$APP_ICON_PNG" --out "$ICONSET/icon_512x512.png" >/dev/null
    sips -z 1024 1024 "$APP_ICON_PNG" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

    iconutil -c icns "$ICONSET" -o dist/Ollama.app/Contents/Resources/icon.icns
    rm -rf "$ICONSET"
}

_derive_app_versions() {
    if [ -z "${OLLAMA_APP_VERSION:-}" ]; then
        OLLAMA_APP_VERSION=$(printf '%s' "$VERSION" | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+(-rc[0-9]+)?).*/\1/')
    fi

    if [ -z "${OLLAMA_APP_BUILD_VERSION:-}" ]; then
        case "$VERSION" in
            "$OLLAMA_APP_VERSION"-*)
                OLLAMA_APP_BUILD_VERSION=${VERSION#"$OLLAMA_APP_VERSION"-}
                ;;
            *)
                OLLAMA_APP_BUILD_VERSION=$VERSION
                ;;
        esac
    fi

    export OLLAMA_APP_VERSION
    export OLLAMA_APP_BUILD_VERSION
}

_build_macapp() {
    if ! command -v npm &> /dev/null; then
        echo "npm is not installed. Please install Node.js and npm first:"
        echo "   Visit: https://nodejs.org/"
        exit 1
    fi

    if ! command -v tsc &> /dev/null; then
        echo "Installing TypeScript compiler..."
        npm install -g typescript
    fi

    echo "Installing required Go tools..."

    cd app/ui/app
    npm install
    npm run build
    cd ../../..

    _derive_app_versions

    # Build the Ollama.app bundle
    rm -rf dist/Ollama.app
    cp -a ./app/darwin/Ollama.app dist/Ollama.app
    _build_custom_app_icon

    # update the modified date of the app bundle to now
    touch dist/Ollama.app

    go clean -cache
    APP_LDFLAGS="-s -w -X=github.com/ollama/ollama/app/version.Version=${VERSION}"
    if truthy "${OLLAMA_DISABLE_UPDATES:-}"; then
        status "Building app with automatic updates disabled"
        APP_LDFLAGS="$APP_LDFLAGS -X=github.com/ollama/ollama/app/updater.DisableUpdates=true"
    fi
    _build_darwin_app_launcher

    # Create a mock Squirrel.framework bundle
    mkdir -p dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/
    cp -a dist/Ollama.app/Contents/MacOS/Ollama dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
    ln -s ../Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/ShipIt
    cp -a ./app/cmd/squirrel/Info.plist dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/Info.plist
    ln -s A dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/Current
    ln -s Versions/Current/Resources dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Resources
    ln -s Versions/Current/Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Squirrel

    # Update the version in the Info.plist
    plutil -replace CFBundleShortVersionString -string "$OLLAMA_APP_VERSION" dist/Ollama.app/Contents/Info.plist
    plutil -replace CFBundleVersion -string "$OLLAMA_APP_BUILD_VERSION" dist/Ollama.app/Contents/Info.plist

    # Setup the ollama binaries
    mkdir -p dist/Ollama.app/Contents/Resources
    _prepare_darwin_app_runtime
    cp -a dist/darwin/ollama dist/Ollama.app/Contents/Resources/ollama
    cp dist/darwin/llama-server dist/Ollama.app/Contents/Resources/
    cp dist/darwin/llama-quantize dist/Ollama.app/Contents/Resources/
    if [ -d dist/darwin/lib/ollama ]; then
        cp -a dist/darwin/lib/ollama/. dist/Ollama.app/Contents/Resources/
    fi
    chmod a+x dist/Ollama.app/Contents/Resources/ollama

    LOCAL_SIGN=${OLLAMA_LOCAL_SIGN:-1}
    if [ -n "$APPLE_IDENTITY" ]; then
        _sign_app_bundle "$APPLE_IDENTITY"
    elif truthy "$LOCAL_SIGN"; then
        status "Ad-hoc signing local app bundle"
        _sign_app_bundle -
        _prepare_local_app_for_launch
        status "Local app is ad-hoc signed and not notarized; do not redistribute it"
    fi

    rm -f dist/Ollama-darwin.zip
    ditto -c -k --norsrc --keepParent dist/Ollama.app dist/Ollama-darwin.zip
    (cd dist/Ollama.app/Contents/Resources/; tar -cf - ollama llama-server llama-quantize *.so *.dylib *.metallib mlx_metal_v*/ 2>/dev/null) | gzip -9vc > dist/ollama-darwin.tgz

    # Notarize, staple, and create the signed DMG through the resumable helper.
    if [ -n "$APPLE_IDENTITY" ]; then
        # Do not leave a previous DMG available for the caller to mistake for
        # the output of this notarization attempt.
        rm -f dist/Ollama.dmg
        ./scripts/notarize_darwin.sh \
            --release-revision "${MLX_RELEASE_REVISION:-$(git rev-parse HEAD)}" \
            --release-version "$VERSION" \
            --app-version "$OLLAMA_APP_VERSION" \
            --app-build-version "$OLLAMA_APP_BUILD_VERSION" \
            --timeout "${MLX_NOTARY_TIMEOUT:-20m}" || return $?
    else
        if truthy "$LOCAL_SIGN"; then
            echo "WARNING: Developer ID signing and notarization disabled; local app is ad-hoc signed only"
        else
            echo "WARNING: Code signing disabled, this bundle will not work for upgrade testing"
        fi
    fi
}

if [ "$#" -eq 0 ]; then
    _build_darwin
    _sign_darwin
    _build_macapp
    exit 0
fi

for CMD in "$@"; do
    case $CMD in
        build) _build_darwin ;;
        package) _package_darwin_runtime ;;
        sign) _sign_darwin ;;
        app) _build_macapp ;;
        *) usage ;;
    esac
done

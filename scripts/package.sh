#!/usr/bin/env bash
# Build a relocatable native package without publishing or installing it.
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
cargo build --release --locked --target-dir "$PACKAGE_ROOT/target"
PACKAGE_NAME="nemocode-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
PACKAGE_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/nemocode-package.XXXXXX")"
mkdir -p "$PACKAGE_STAGE/$PACKAGE_NAME/bin" "$PACKAGE_STAGE/$PACKAGE_NAME/libexec/nemocode" "$PACKAGE_STAGE/$PACKAGE_NAME/share/nemocode"
install -m 755 target/release/nemocode "$PACKAGE_STAGE/$PACKAGE_NAME/bin/nemocode"
install -m 755 start-nemo.sh "$PACKAGE_STAGE/$PACKAGE_NAME/libexec/nemocode/start-nemo.sh"
cp -R templates "$PACKAGE_STAGE/$PACKAGE_NAME/libexec/nemocode/"
cp README.md LICENSE .env.example "$PACKAGE_STAGE/$PACKAGE_NAME/share/nemocode/"
cp -R profiles docs "$PACKAGE_STAGE/$PACKAGE_NAME/share/nemocode/"
mkdir -p dist
tar -czf "dist/$PACKAGE_NAME.tar.gz" -C "$PACKAGE_STAGE" "$PACKAGE_NAME"
if command -v sha256sum >/dev/null 2>&1; then
  (cd dist && sha256sum "$PACKAGE_NAME.tar.gz" > "$PACKAGE_NAME.tar.gz.sha256")
else
  (cd dist && shasum -a 256 "$PACKAGE_NAME.tar.gz" > "$PACKAGE_NAME.tar.gz.sha256")
fi
echo "Built $PACKAGE_ROOT/dist/$PACKAGE_NAME.tar.gz"
echo "Staging files retained at $PACKAGE_STAGE"

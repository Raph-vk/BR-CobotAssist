#!/usr/bin/env bash
# encrypt_tos_app.sh – build → Nuitka → one-file → AppImage
# final artefacts:   ~/tos_app/utils/app_encryption/dist/

set -euo pipefail
IFS=$'\n\t'

# ────────── fixed paths ──────────
APP_ROOT="$HOME/tos_app"
# where users will find the finished AppImage
CRYPT_ROOT="$APP_ROOT/utils/app_encryption"
DIST_DIR="$CRYPT_ROOT/dist"

# throw-away build workspace (sibling to app_encryption)
WORK_ROOT="$APP_ROOT/utils/app_encryption/build_utils"
BUILD_DIR="$WORK_ROOT/build"
APPDIR="$BUILD_DIR/TOS_UI.AppDir"

ENV_NAME="TOS"
OUTPUT_NAME="tos_ui.bin"
APP_NAME="TOS_UI"
BIN_DIR="$HOME/bin"

# fresh workspace
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT" "$BUILD_DIR" "$APPDIR/usr/bin" "$DIST_DIR"

# ────────── system packages ──────────
sudo dpkg --configure -a
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
     build-essential gcc g++ zlib1g-dev python3-dev python3-venv \
     wget git patchelf imagemagick

# ────────── appimagetool ──────────
mkdir -p "$BIN_DIR"
if [[ ! -x "$BIN_DIR/appimagetool" ]]; then
  wget -qO "$BIN_DIR/appimagetool" \
    https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$BIN_DIR/appimagetool"
fi
export PATH="$BIN_DIR:$PATH"

# ────────── conda & python deps ──────────
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
conda install -y libpython-static >/dev/null
set -u
python -m pip install --upgrade --quiet nuitka wheel ordered-set zstandard

# ────────── Nuitka project file inside WORK_ROOT ──────────
cat > "$WORK_ROOT/.nuitka-project" <<'EOF'
--standalone
--onefile
--assume-yes-for-downloads
--follow-imports
--enable-plugin=torch
--enable-plugin=numpy
--enable-plugin=multiprocessing
--include-data-dir=applications/tos_ui/templates=templates
--include-data-dir=applications/tos_ui/static=static
--include-data-dir=config=config
--disable-asserts --remove-docstrings
EOF

# ────────── compile ──────────
cd "$WORK_ROOT"
python -m nuitka \
       --output-filename="$OUTPUT_NAME" \
       "$APP_ROOT/applications/tos_ui/main.py"

cp "$OUTPUT_NAME" "$APPDIR/usr/bin/tos_ui"

# ────────── icon (256×256 PNG) ──────────
convert "$APP_ROOT/applications/tos_ui/static/tos_logo.jpg" -resize 256x256 \
        "$APPDIR/tos_logo.png"

# ────────── desktop entry & AppRun ──────────
cat > "$APPDIR/${APP_NAME,,}.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Exec=tos_ui
Icon=tos_logo
Categories=Science;
EOF

cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
APPDIR="$(dirname "$(readlink -f "$0")")"
export TOS_ROOT="$APPDIR"
export TOS_CONFIG_PATH="$APPDIR/config"
export TOS_LOG_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/tos_ui/logs"
mkdir -p "$TOS_LOG_PATH"
exec "$APPDIR/usr/bin/tos_ui" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# ────────── package ──────────
IMG_NAME="${APP_NAME}-$(date +%Y%m%d).AppImage"
ARCH=x86_64 appimagetool "$APPDIR" "$DIST_DIR/$IMG_NAME"

# ────────── cleanup scratch workspace ──────────
rm -rf "$WORK_ROOT"

echo -e "\n✅  Finished: $DIST_DIR/$IMG_NAME"
echo "   Only dist/ remains inside utils/app_encryption/"

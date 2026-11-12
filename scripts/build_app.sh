#!/usr/bin/env bash
# Build app/build/TacitVoice.app from the Swift package. Needs Xcode.
#
# macOS ties the Camera, Microphone and Accessibility grants to the app's
# signature. With no identity the app is signed ad hoc and every rebuild
# counts as a new app, so the grants must be given again. To keep them, sign
# with your Apple Development certificate (listed by
# `security find-identity -v -p codesigning`):
#   CODESIGN_IDENTITY="Apple Development: you@example.com (TEAMID)" ./scripts/build_app.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../app"

swift build -c release
mkdir -p build/TacitVoice.app/Contents/MacOS
cp .build/release/TacitVoice build/TacitVoice.app/Contents/MacOS/TacitVoice
cp Info.plist build/TacitVoice.app/Contents/Info.plist
codesign --force --sign "${CODESIGN_IDENTITY:--}" build/TacitVoice.app

echo "Built app/build/TacitVoice.app"
echo "Run it with:  open app/build/TacitVoice.app"

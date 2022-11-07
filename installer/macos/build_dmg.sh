#!/bin/sh
mkdir -p dmg
rm -r dmg/*
cp -r "../bundle/app/sscanss.app" dmg
cp -r "../bundle/editor.app" dmg
rm -f -- "sscanss.dmg"
create-dmg \
  --volname "sscanss" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --icon "../icons/logo.icns" 175 120 \
  --hide-extension "sscanss.app" \
  --app-drop-link 425 120 \
  "sscanss.dmg" \
  "dmg/"
rm -r "dmg"
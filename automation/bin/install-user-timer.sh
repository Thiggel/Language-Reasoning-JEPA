#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$HOME/.config/systemd/user"
test -f "$HOME/.config/textjepa-research/controller.toml" || {
  echo "missing external controller policy: $HOME/.config/textjepa-research/controller.toml" >&2
  exit 2
}
sed "s#/vol/home-vol2/ml/laitenbf/TextJEPA#$ROOT#g" \
  "$ROOT/automation/systemd/textjepa-research-watch.service" \
  > "$HOME/.config/systemd/user/textjepa-research-watch.service"
cp "$ROOT/automation/systemd/textjepa-research-watch.timer" \
  "$HOME/.config/systemd/user/textjepa-research-watch.timer"
systemctl --user daemon-reload
systemctl --user enable --now textjepa-research-watch.timer
systemctl --user status textjepa-research-watch.timer --no-pager

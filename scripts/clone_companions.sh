#!/usr/bin/env bash
# Clone the companion repositories used as labs and reference implementations.
#
# Each is cloned into ./companions/<name> and checked out at the commit pinned in
# docs/companion-repos.md. Re-running is safe: existing clones are fetched and
# re-pinned, never overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${RL_LADDER_COMPANIONS:-$ROOT/companions}"
FORK_OWNER="${RL_LADDER_FORK_OWNER:-davidmiheev}"

# name|owner|repo|pinned commit|branch to track
# Owner "-" means the fork owner above; hftbacktest is tracked upstream (no fork).
REPOS=(
  "Gymnasium|-|Gymnasium|c956926f9caefd507597fb2bf18b789a96c45626|main"
  "Practical_RL|-|Practical_RL|6f7fa8bc521af3d6bb47833c4670f44740f26aa6|master"
  "nlp_course|-|nlp_course|5cbabc4d5cbf4d411748ea2cc0701db37a32eefd|2025"
  "open-r1|-|open-r1|1416fa0cf21595d2083b399a2a0bbddd7f6e9563|main"
  "stable-baselines3|-|stable-baselines3|3246f5060eacd8bff4de575eaafbbc6e0baac376|master"
  "trl|-|trl|ac5184017e890d64fde8d2ecf47462d193eef14f|main"
  "hftbacktest|nkaz001|hftbacktest|5f3ec40b2afb764e0fea112f941ed85523ef4e88|master"
)

mkdir -p "$DEST"

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name owner repo sha branch <<<"$entry"
  [ "$owner" = "-" ] && owner="$FORK_OWNER"
  target="$DEST/$name"
  url="https://github.com/$owner/$repo"

  if [ -d "$target/.git" ]; then
    echo "==> $name: fetching"
    git -C "$target" fetch --quiet origin "$branch" || git -C "$target" fetch --quiet origin
  else
    echo "==> $name: cloning $url"
    git clone --quiet "$url" "$target"
  fi

  if git -C "$target" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$target" checkout --quiet --detach "$sha"
    echo "    pinned at $sha"
  else
    echo "    WARNING: pinned commit $sha not found; staying on $(git -C "$target" rev-parse --abbrev-ref HEAD)"
  fi
done

echo
echo "Companions in $DEST. See docs/companion-repos.md for which phase uses which."

#!/usr/bin/env bash
# Fails when the given GitHub login has not signed the Plumb CLA and is
# not exempt. Usage: check.sh <github-login> [signatures-file]
set -euo pipefail

author_raw="${1:?usage: check.sh <github-login> [signatures-file]}"
signatures="${2:-$(dirname "$0")/signatures.json}"
author="$(printf '%s' "$author_raw" | tr '[:upper:]' '[:lower:]')"

# The steward already holds his own copyright; bots hold none.
exempt=("fabiansalge" "dependabot[bot]" "github-actions[bot]")
for login in "${exempt[@]}"; do
  if [[ "$author" == "$login" ]]; then
    echo "$author_raw is exempt from the CLA."
    exit 0
  fi
done

if jq -e --arg author "$author" '
    .signatures[]
    | select(
        ((.github // "") | ascii_downcase) == $author
        or ([.authorized[]? | ascii_downcase] | index($author)) != null
      )
  ' "$signatures" >/dev/null; then
  echo "$author_raw has signed the CLA."
  exit 0
fi

echo "::error::$author_raw has not signed the Plumb Contributor License Agreement."
echo "To sign, read .github/cla/individual-cla.md (or entity-cla.md for corporate"
echo "contributions) and add an entry for yourself to .github/cla/signatures.json"
echo "in this pull request. CONTRIBUTING.md documents the entry format."
exit 1

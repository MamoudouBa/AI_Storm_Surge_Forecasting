#!/bin/bash
# Master copyright/acknowledgment and Keras 2 legacy flag injection script

for f in *.py; do
    # 1. Inject Gemini acknowledgment header if missing
    if ! grep -q "Gemini" "$f"; then
        echo -e "# Developed with the assistance of the Gemini AI code assistant.\n$(cat "$f")" > "$f.tmp" && mv "$f.tmp" "$f"
        echo "✅ Added acknowledgment to $f"
    else
        echo "⏭️ Skipped acknowledgment for $f (already present)"
    fi

    # 2. Inject TF_USE_LEGACY_KERAS flag if missing
    if ! grep -q "TF_USE_LEGACY_KERAS" "$f"; then
        python3 -c "
import sys

with open('$f', 'r') as file:
    lines = file.readlines()

inserted = False
new_lines = []
flag_block = [
    '# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)\n',
    'import os\n',
    'os.environ[\"TF_USE_LEGACY_KERAS\"] = \"1\"\n'
]

for line in lines:
    new_lines.append(line)
    if not inserted and ('import os' in line or 'import sys' in line):
        if 'import os' not in line:
            new_lines.append('import os\n')
        new_lines.append('# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)\n')
        new_lines.append('os.environ[\"TF_USE_LEGACY_KERAS\"] = \"1\"\n')
        inserted = True

if not inserted:
    # Insert near top if no os/sys imports found
    idx = 1 if lines and lines[0].startswith('#!') else 0
    new_lines = lines[:idx] + flag_block + lines[idx:]

with open('$f', 'w') as file:
    file.writelines(new_lines)
"
        echo "🛠️ Injected TF_USE_LEGACY_KERAS flag into $f"
    else
        echo "⏭️ Skipped legacy Keras flag for $f (already present)"
    fi
done

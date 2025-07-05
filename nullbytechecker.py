# nullbytechecker.py
#
# Végigjárja a projekt „src” mappáját, és kilistázza azokat
# a .py fájlokat, amelyekben 0x00 (null byte) található.
# Futás:  python nullbytechecker.py

import os

root_dir = "src"          # gyökérmappa, ahol keresünk
bad_files = []

for root, _, files in os.walk(root_dir):
    for fn in files:
        if fn.endswith(".py"):
            path = os.path.join(root, fn)
            try:
                with open(path, "rb") as fh:
                    if b"\x00" in fh.read():
                        bad_files.append(path)
            except Exception as e:
                print(f"Hiba a fájl olvasásakor: {path} -> {e}")

print(f"\nTalált fájlok null byte-tal: {len(bad_files)}")
for p in bad_files:
    print("  ", p)

if not bad_files:
    print("✔️  Minden .py fájl tiszta (UTF-8).")
else:
    print("\n❗ Ezeket a fájlokat mentsd át UTF-8 kódolásba.")
import os
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [
    os.path.join(_ROOT, "logs"),
    os.path.join(_ROOT, "trained_models"),
    os.path.join(_ROOT, "results"),
]

for target in TARGETS:
    if not os.path.isdir(target):
        print(f"Skipping {target} (not found).")
        continue
    for entry in os.listdir(target):
        entry_path = os.path.join(target, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)
    print(f"Cleared {target}.")

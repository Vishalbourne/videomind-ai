from pathlib import Path
import os

a = Path("database")

b = os.listdir(a)

if '7eed1f52-fa21-44fe-bbbf-381eb125efc4' in b:
    print(True)
else:
    print(False)


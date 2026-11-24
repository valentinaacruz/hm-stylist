"""Ejecuta un notebook del proyecto dentro de Kaggle y guarda la versión EJECUTADA (con salidas y gráficas)
en /kaggle/working, para poder bajarla con `kaggle kernels output` y dejarla en el repo.

El notebook llega como parte del dataset hm-stylist-src. `kaggle kernels pull` solo devuelve el código,
por eso se usa este ejecutor.
"""
import json
import os
import subprocess
import sys

NOTEBOOK = "05-ranking.ipynb"   # notebook a ejecutar (debe estar en el dataset hm-stylist-src)


def find(name):
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x != "images"]    # no recorrer las 105k fotos
        if name in files:
            return os.path.join(d, name)
    sys.exit(f"No encontré {name}: agrega el dataset hm-stylist-src como Input")


src = find(NOTEBOOK)
os.chdir("/kaggle/working")
print("Ejecutando", src, flush=True)
subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", src,
                "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3", "--allow-errors",
                "--output-dir", "/kaggle/working", "--output", NOTEBOOK], check=True)

nb = json.load(open(NOTEBOOK, encoding="utf-8"))
errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb["cells"])
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
for i, name, value in errors:
    print(f"ERROR en la celda {i}: {name}: {value}")
sys.exit(1 if errors else 0)

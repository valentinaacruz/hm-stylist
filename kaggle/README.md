# Ejecutar los notebooks en Kaggle desde el repo

Los datos pesados (dataset H&M y artefactos intermedios) nunca salen de Kaggle. El código se edita en el repo,
se ejecuta en Kaggle con el CLI y el notebook ejecutado (con salidas y gráficas) regresa al repo para la entrega.

```
repo: src/*.py + notebook ──kaggle datasets version──▶ dataset privado hm-stylist-src
                                                              │
kaggle kernels push (run_notebook.py) ──▶ Kaggle ejecuta el notebook con nbconvert (GPU + Internet)
                                                              │
repo ◀── kaggle kernels output: notebook ejecutado + archivos de salida ◀──┘
```

`kaggle kernels pull` solo devuelve código sin salidas; por eso el kernel es un script (`run_notebook.py`) que
ejecuta el notebook y lo guarda ya ejecutado en `/kaggle/working`.

## Requisitos (una vez)
1. `pip install kaggle`
2. Token en <https://www.kaggle.com/settings/api> → guardarlo en `~/.kaggle/access_token` (nunca en el repo).
3. Aceptar las reglas de la competencia H&M con tu cuenta.

## Contenido
| Archivo | Qué es |
|---|---|
| `hm-stylist-src/dataset-metadata.json` | Dataset privado con `src/*.py` y el notebook a ejecutar |
| `06-rag-y-chat-estilista/kernel-metadata.json` | Kernel del 06: inputs (competencia, 01, 03, 04, src), GPU e Internet |
| `run_notebook.py` | Ejecutor: corre el notebook con nbconvert y falla si alguna celda da error |
| `_staging/`, `_runs/` | Temporales (ignorados por git) |

## Comandos (desde la raíz del repo; el CLI necesita rutas absolutas)
```bash
# 1. Publicar código + notebook
mkdir -p kaggle/_staging/hm-stylist-src
cp src/*.py notebooks/06-rag-y-chat-estilista.ipynb kaggle/hm-stylist-src/dataset-metadata.json kaggle/_staging/hm-stylist-src/
kaggle datasets version -p "$PWD/kaggle/_staging/hm-stylist-src" -m "descripción del cambio"
kaggle datasets status angelnoelprezmartnez/hm-stylist-src        # esperar "ready"

# 2. Ejecutar en Kaggle (~12 min con GPU T4)
kaggle kernels push -p "$PWD/kaggle/06-rag-y-chat-estilista"
kaggle kernels status angelnoelprezmartnez/06-rag-y-chat-estilista

# 3. Bajar el notebook ejecutado y las salidas, y copiarlo a notebooks/
kaggle kernels output angelnoelprezmartnez/06-rag-y-chat-estilista -p "$PWD/kaggle/_runs/06" -o
cp kaggle/_runs/06/06-rag-y-chat-estilista.ipynb notebooks/
```
Después se revisa el diff y se hace commit desde VS Code.

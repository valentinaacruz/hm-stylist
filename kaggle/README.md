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
| `07-exportar-demo/` | Kernel del 07 (CPU): arma `hm-demo-data.zip` para la app; su propio `run_notebook.py` apunta al 07 |
| `05-ranking/` | Kernel del 05 (CPU), para re-ejecutarlo con las imágenes de la competencia |
| `_staging/`, `_runs/` | Temporales (ignorados por git) |

## Comandos (desde la raíz del repo)
En Windows el CLI solo acepta rutas absolutas con diagonal invertida (`D:\...`). Con rutas relativas o con `/`
la subida del dataset falla y Kaggle sigue usando la versión anterior sin avisar.
```bash
R='D:\Proyecto_Datos_Masivos_HM\hm-stylist'

# 1. Publicar código + notebooks (incluye todos los que se ejecutan con el CLI)
mkdir -p kaggle/_staging/hm-stylist-src
cp src/*.py notebooks/0[567]-*.ipynb kaggle/hm-stylist-src/dataset-metadata.json kaggle/_staging/hm-stylist-src/
kaggle datasets version -p "$R\kaggle\_staging\hm-stylist-src" -m "descripción del cambio"
kaggle datasets files angelnoelprezmartnez/hm-stylist-src         # confirmar que aparecen los archivos nuevos

# 2. Ejecutar en Kaggle (05: ~15 min sin GPU · 06: ~12 min con GPU T4 · 07: ~10 min sin GPU)
kaggle kernels push -p "$R\kaggle\06-rag-y-chat-estilista"
kaggle kernels status angelnoelprezmartnez/06-rag-y-chat-estilista

# 3. Bajar el notebook ejecutado y las salidas, y copiarlo a notebooks/
kaggle kernels output angelnoelprezmartnez/06-rag-y-chat-estilista -p "$R\kaggle\_runs\06" -o
cp kaggle/_runs/06/06-rag-y-chat-estilista.ipynb notebooks/
```
Después se revisa el diff y se hace commit desde VS Code.

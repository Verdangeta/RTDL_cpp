# RTDL_cpp

Библиотека **RTD-Lite** — топологические баркоды между двумя матрицами расстояний (C++ ядро + Python-обёртка).

Ветка `LCT_RTDL` добавляет быстрый путь восстановления баркодов через Link-Cut Tree: `O(n log n)` вместо наивного `O(n²)`. По умолчанию поведение прежнее (naive); fast включается через `RTDL_FAST_BARCODE=1`.

## Сборка

```bash
conda activate TDA_L2C   # или своё окружение с numpy, torch, matplotlib
python setup.py build_ext --inplace
```

## Быстрый путь

```bash
RTDL_FAST_BARCODE=1 python your_script.py
```

Дифф-тест naive vs fast (43 кейса, bit-identical):

```bash
python tests/test_fast_barcode_diff.py
```

## Как получены картинки в `docs/figures/`

Основная метрика — **время после того, как MST построены** (`after_mst_median_sec`), без копирования dense-матриц, Prim и TSP-хвоста. Wall-time графики — только контекст (`docs/figures/context/`).

**Полный перезамер + PNG** (лог-log и linear-x для TSP и 4 плотностей графа):

```bash
python scripts/plot_time_curves.py \
  --tsp-sizes 100 200 400 800 1600 2400 3200 4000 \
  --graph-sizes 100 200 400 800 1600 2400 3200 4000 \
  --outer-repeats 3 --inner-repeats 5 --warmup 1 --with-wall
```

Скрипт вызывает публичный API `rtd_lite_run_matrix`, читает тайминг из stderr (`RTDL_TIMING=1`, `RTDL_BENCH_REPEAT=5`) и пишет PNG в `docs/figures/`.

**Только перерисовка** из закэшированных чисел (без повторного C++ benchmark):

```bash
python scripts/plot_linear_x_from_latest.py --with-wall
```

Числа в `plot_linear_x_from_latest.py` обновляются вручную после свежего прогона `plot_time_curves.py`.

Табличный вывод по одному кейсу:

```bash
RTDL_TIMING=1 RTDL_BENCH_REPEAT=5 \
  python scripts/benchmark_fast_barcode.py --case tsp \
  --sizes 100 200 400 800 1600 2400 3200 4000
```

## Структура

```
RTDLite/          C++ ядро, link_cut_tree.h, rtdlite.cpp
scripts/          benchmark и plot_time_curves.py
tests/            test_fast_barcode_diff.py
docs/figures/     закоммиченные PNG (основные + context/)
```

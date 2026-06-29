# RTDLite

Библиотека для вычисления **Representation Topological Distance (RTD-Lite)** — топологических баркодов между двумя матрицами расстояний.

Ключевое свойство: при **r₁ = полный граф** и **r₂ = частичный тур** (inf для отсутствующих рёбер) алгоритм даёт соответствие между рёбрами MST(r₁) и рёбрами тура r₂, что используется в задачах TSP и обучения нейросетей.

## Использование

### Python (PyTorch)

```python
from RTDLite import RTD_Lite
import torch

# r1 — полный граф (координаты точек или матрица расстояний)
# r2 — частичный тур (матрица с inf для рёбер вне тура)
rtd = RTD_Lite(r1, r2, distance='euclidean')  # или distance='precomputed'
barcodes = rtd()

# barcodes['1->2'], barcodes['2->1'] — баркоды как тензоры [birth, death]
# Веса рёбер для обучения:
edge_indices, edge_weights = rtd.get_edge_weights(tour_edges)
```

### C/C++

```c
#include "rtdlite.h"

rtd_lite_result res = rtd_lite_run_matrix(r1_data, r2_data, n, true);
// ... работа с res.left_to_right, res.right_to_left ...
rtd_lite_result_free(&res);
```

Подробнее: [RTDLite/CPP_USAGE.md](RTDLite/CPP_USAGE.md)

## Сборка

### Зависимости

- Python: numpy, torch
- C++: C++17, CMake 3.5+

### Установка (Python)

```bash
pip install .
# или
python setup.py install
```

### Сборка C++ библиотеки отдельно

```bash
mkdir build && cd build
cmake .. -DRTDLITE_BUILD_CLI=ON
make
```

## Структура

```
RTDLite/
├── CMakeLists.txt      # Сборка C++ библиотеки
├── setup.py            # Python-пакет (PyTorch + ctypes)
├── recompile.sh        # Пересборка после изменений в C++
├── RTDLite/
│   ├── rtdlite.cpp     # Ядро алгоритма (C++)
│   ├── rtdlite.h       # C API
│   ├── __init__.py     # Python API
│   ├── converter.py    # ctypes-мост
│   └── CPP_USAGE.md    # Документация C API
└── README.md
```

## Алгоритм

1. Вычисляется rmin = min(r₁, r₂) (для inf в r₂: min(x, inf) = x).
2. MST для rmin, r₁, r₂.
3. Для каждого ребра MST(rmin) по возрастанию веса:
   - birth = вес ребра в rmin;
   - death = минимальный порог, при котором соответствующие компоненты соединяются в r₁ (направление 1→2) или r₂ (2→1).
4. Баркод (birth, death) задаётся индексами вершин рёбер.

## Лицензия

MIT (см. [LICENSE](LICENSE))

## Авторы

Eduard Tulchinskii, Daria Voronkova, Ilya Trofimov, Serguei Barannikov  
Исходный репозиторий: [ArGintum/RTD-Lite](https://github.com/ArGintum/RTD-Lite)

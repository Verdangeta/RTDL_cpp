# RTDLite C++ Library Usage

Библиотека RTDLite может использоваться как самостоятельная C/C++ библиотека.

## API (rtdlite.h)

```c
#include "rtdlite.h"

// Вычислить баркоды из двух матриц (row-major, n*n)
rtd_lite_result result = rtd_lite_run_matrix(data_1, data_2, n, true);

// С предпосчитанным r1 MST (рекомендуется, когда r1 фиксирован)
rtd_lite_result result = rtd_lite_run_matrix_with_mst(
    data_1, data_2, n,
    r1_mst_edge_idx,  // [u0,v0, u1,v1, ...] — n-1 рёбер
    r1_mst_edge_w,    // n-1 весов, отсортированы по возрастанию
    true
);

// Из файла (сначала n*n значений r1, затем n*n r2)
rtd_lite_result result = rtd_lite_run_from_file("matrices.txt", n, true);

// Освободить память результата
rtd_lite_result_free(&result);
```

## Структура результата

```c
typedef struct {
    rtd_index_t left_bars;    // число баркодов 1->2
    rtd_index_t right_bars;  // число баркодов 2->1
    rtd_birth_death_edges *left_to_right;  // баркоды 1->2
    rtd_birth_death_edges *right_to_left;  // баркоды 2->1
} rtd_lite_result;
```

Каждый `rtd_birth_death_edges` содержит: `birth_i, birth_j` (ребро рождения), `death_i, death_j` (ребро смерти).

## Сборка

```bash
mkdir build && cd build
cmake .. -DRTDLITE_BUILD_CLI=ON  # ON — собрать CLI
make
```

Получаем:
- `librtd_lite.so` (или `rtd_lite.so`) — библиотека
- `rtdlite_cli` — CLI (если `RTDLITE_BUILD_CLI=ON`)

## Установка

```bash
cd build
cmake --install . --prefix /usr/local  # или другой prefix
```

Устанавливаются:
- `include/rtdlite.h`
- `lib/librtd_lite.so` (или аналог)

## Ссылка из своего проекта

```cmake
find_library(RTDLITE_LIB rtd_lite PATHS /path/to/install/lib)
target_link_libraries(your_target ${RTDLITE_LIB})
target_include_directories(your_target PRIVATE /path/to/install/include)
```

## Важно

- Результат выделяется через `malloc`; вызывающий обязан вызвать `rtd_lite_result_free(&result)`.
- Для r1 = полный граф, r2 = частичный тур (inf для отсутствующих рёбер): rmin = r1, соответствие рёбер r1 MST ↔ r2 MST даётся в `right_to_left`.

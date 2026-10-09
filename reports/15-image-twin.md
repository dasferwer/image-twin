# P2-15 — внешняя проверка ImageTwin

Добавлена воспроизводимая проверка реального MobileNet v2 + pHash + ORB/RANSAC на новом семействе изображений. Алгоритм и его пороги не менялись. Метрики рассчитаны через текущий маршрут `/images/{id}/duplicates?limit=80`: реальная PostgreSQL/pgvector, реальные сохранённые признаки и объекты MinIO. Обёртка `compare` записывает фактически проверенные кандидаты, вызывая исходную функцию; результаты не подменяются. Нормализация и inference выполняются до сохранения, без очереди в измеряемой части. Отдельный HTTP smoke проверяет настоящий API, RabbitMQ и worker.

## Источник и замороженный протокол

Источник — [Fashion-MNIST, Zalando SE, 2017](https://github.com/zalandoresearch/fashion-mnist), официальный test split, ревизия `b2617bb6d3ffa2e429640350f613e3291e10b141`. Изображения находятся непосредственно в `data/fashion` официального репозитория набора; раздел License README этого набора объявляет MIT, копия сохранена в `docs/external/LICENSE`. Проверена страница издателя, а не лицензия стороннего загрузчика. Формулировка самой лицензии использует слово Software; отдельные права исходных фотографий в ней не перечислены. Поэтому архивы и изображения остаются во внешнем локальном кеше, их перераспространение в проекте не производится.

`docs/external/source.json` фиксирует ревизию, SHA-256 двух архивов, индексы, преобразования и правила меток до итогового запуска. `corpus.jsonl` содержит индексы, официальные классы и SHA-256 исходных пикселей 509 различных изображений. Никакого обучения/подбора порогов по этим меткам нет. Набор относится к новому семейству Zalando и не содержит восьми scikit-image calibration/evaluation fixtures; отсутствие совпадений с историческим ImageNet pretraining нельзя проверить по доступной модели.

Размеры 50/200/500 включают одну запись запроса: каталог содержит первые N−1 изображений, временный запрос удаляется после каждого поиска. 500 — текущий лимит коллекции проекта. Положительные запросы: индексы 0–9 × JPEG/resize/crop/watermark, всего 40 на размер. Это **сгенерированные преобразования** исходных внешних изображений. Отрицательные запросы: ранее отсутствующие индексы 500–509, всего 10 на размер. В проверенных кандидатах отдельно считаются изображения того же класса (тематически похожие негативные пары) и другого класса. Метка duplicate означает один исходный test index, а не одинаковый класс товара.

## Итоговые метрики

| Записей в коллекции | Candidate recall | Recall | Precision | Geometry recall / precision | FP | p95 поиска, мс | Пик RSS процесса, МиБ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1,00 | 0,675 | 1,00 | 0,375 / 1,00 | 0 | 30,83 | 223,75 |
| 200 | 1,00 | 0,675 | 1,00 | 0,375 / 1,00 | 0 | 44,82 | 225,50 |
| 500 | 1,00 | 0,675 | 1,00 | 0,375 / 1,00 | 0 | 54,48 | 226,25 |

27/40 копий обнаружены; 13/40 пропущены на каждом размере. Геометрия подтверждает 15/40, без ложных подтверждений. На 500 записях проверены 1415 негативных пар одного класса и 1885 другого класса; FP=0 в обеих группах. Candidate recall измеряет попадание известного оригинала в фактические кандидаты, а geometry precision учитывает только геометрическое подтверждение, независимо от pHash ветки.

| Преобразование, 500 записей | Recall | Geometry recall | p95, мс |
|---|---:|---:|---:|
| JPEG | 1,00 | 0,60 | 45,18 |
| Resize | 1,00 | 0,20 | 26,63 |
| Crop | 0,30 | 0,30 | 49,98 |
| Watermark | 0,40 | 0,40 | 82,06 |
| Unseen original | нет положительных меток | нет положительных меток | 29,26 |

Полные метрики, знаменатели FP, все запросы и значения для каждого размера/преобразования: `docs/external/evaluation.json`.

## Ограничения

Нативные снимки — grayscale 28×28; перед тестом они увеличены bilinear до 224×224 RGB, поскольку API требует минимум 32 пикселя. Увеличение не добавляет деталей. Это ограниченный тест одежды/низкой текстуры, не доказательство качества на высоком разрешении или реальном маркетплейсе. Тематически похожий класс не гарантирует визуально близкую пару; реальные разные ракурсы одного товара не размечены. Возможные естественные дубликаты разных индексов не прошли ручную экспертизу. Нулевая FP на малой выборке не означает нулевую FP в эксплуатации.

Время — ASGI HTTP маршрут с реальными PostgreSQL/S3, без сетевого TCP API, загрузки, нормализации, inference, очереди и сохранения объекта. Работа БД включена в маршрут; S3 используется при подготовке. RSS — весь процесс Linux, включая модель и заранее вычисленные признаки 509 снимков; это high-water mark, не независимое измерение памяти каждой коллекции и не память всех контейнеров. Замеры выполнены в Docker на этом Mac, 50 запросов на размер, без параллельной пользовательской нагрузки; RANSAC/UUID tie-break могут влиять на повторный запуск.

## Команды и проверки

Команды выполнялись из worktree `/Users/mac/Documents/Codex/portfolio-p2-2026-10-09/worktrees/15-image-twin`. Внешний override `/Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml`:

```yaml
networks:
  default:
    ipam:
      config:
        - subnet: 10.241.15.0/24
services:
  api:
    ports: !override ["127.0.0.1:8159:8000"]
```

```sh
uv run --extra dev pytest evaluation_tests -o testpaths=evaluation_tests
uv run ruff check .
uv run ruff format --check .
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml --profile test build test test-storage api migrate
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml --profile test up -d --wait test-db test-storage
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml --profile test run --rm -v /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/worktrees/15-image-twin/scripts:/app/scripts:ro -v /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/worktrees/15-image-twin/docs:/app/docs -v /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/worktrees/15-image-twin/evaluation_tests:/app/evaluation_tests:ro -v /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/cache/imagetwin:/external-cache:ro test sh -c 'pytest tests evaluation_tests && PYTHONPATH=/app python -m scripts.evaluate_external --cache /external-cache'
```

Загрузчик сам скачивает архивы по зафиксированной ревизии, если кеш пуст; SHA-256 обязателен. Для Docker пустой кеш следует смонтировать с записью вместо `:ro`. Для текущего прогона архивы предварительно скачаны во внешний кеш; в проект включены только метаданные и агрегированные/числовые результаты.

TDD: 2 исходных теста упали из-за отсутствующего evaluator, затем прошли; добавлена проверка отказа на повреждённом кеше. 3 evaluation tests прошли локально. Полный suite в Docker: **30 passed** (27 существующих + 3 новых). Новые тесты включены в default pytest testpaths и Docker test image. Ruff check и format check прошли. Отдельной существующей настройки type checker в проекте нет; проверка типов не заявляется.

```sh
docker tag codex-p2-15-1009-api codex-p2-15-1009-worker
docker tag codex-p2-15-1009-api codex-p2-15-1009-dispatcher
docker tag codex-p2-15-1009-api codex-p2-15-1009-cleanup-worker
docker tag codex-p2-15-1009-test-storage codex-p2-15-1009-storage
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml up -d --no-build --wait --wait-timeout 180
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml exec -T api python scripts/smoke.py
```

Smoke: `ok=true`, четыре преобразования найдены, exact upload deduplicated, unrelated rejected, удаление исключило объект и поиск. Все собственные сервисы healthy. Исходные пользовательские контейнеры не запускались и не удалялись.

Самопроверка: алгоритм и основная конфигурация Compose не менялись; новые файлы не содержат изображений/секретов; числа согласованы с JSON; отрицательные и положительные знаменатели разделены; окружение защищено проверками TESTING, *_test БД и отдельного S3 bucket. Вызовы текущего encoder/compare, БД и S3 не заменены заглушками. Основной README, исходный checkout, .env и общая инфраструктура не менялись. Публикация/CI в этой подзадаче не выполнялись.

После включения новых тестов в Dockerfile/testpaths повторно собран test image и выполнена штатная команда (без bind-mount тестов):

```sh
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml --profile test build test
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml --profile test run --rm test
```

Результат: **30 passed in 6.14s**, Ruff check и format check прошли (34 Python-файла), `git diff --check` чистый.

Очистка выполнена только для собственного проекта:

```sh
docker compose -p codex-p2-15-1009 -f docker-compose.yml -f /Users/mac/Documents/Codex/portfolio-p2-2026-10-09/imagetwin-compose.yml --profile test down -v --remove-orphans
docker ps -a --filter name=codex-p2-15-1009 --format '{{.Names}}'
docker network ls --filter name=codex-p2-15-1009 --format '{{.Name}}'
docker volume ls --filter name=codex-p2-15-1009 --format '{{.Name}}'
```

Собственные контейнеры, сеть и volumes отсутствуют. Собственные build images оставлены как воспроизводимый build cache; исходные image tags не менялись. Внешний кеш изображений сохранён за пределами Git. Временные строки/объекты evaluator удаляет в `finally`; окончательная очистка удаляет весь собственный тестовый/демонстрационный storage.

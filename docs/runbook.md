# Запуск и проверка

```bash
docker compose up --build -d --wait --wait-timeout 180
docker compose ps
curl -fsS http://localhost:8150/health
docker compose logs --tail=100 api dispatcher worker
```

Только API публикует порт на 127.0.0.1. PostgreSQL, RabbitMQ и MinIO доступны
в сети Compose. Миграции создают расширение vector, seed — пользователя,
коллекцию и закрытый bucket. Seed можно запускать повторно.

```bash
uv sync --extra dev --frozen
uv run ruff check .
uv run ruff format --check .
docker compose --profile test build test
docker compose --profile test run --rm test
docker compose exec -T api python scripts/smoke.py
uv run python scripts/evaluate.py
uv run python scripts/recovery_smoke.py
```

Тесты используют отдельные PostgreSQL и MinIO с tmpfs. Перед очисткой проверяются
TESTING, суффикс базы `_test` и точное имя bucket `imagetwin-test`. После проверки:

```bash
docker compose --profile test stop test-db test-storage
```

## Если файл не готов

Uploading после ошибки S3: повторить исходную загрузку с тем же ключом.
После оборванного соединения может потребоваться дождаться 60-секундного lease.
Pending/queued: проверить dispatcher и RabbitMQ. Processing после падения:
worker восстановится по истечении lease. Failed: исправить причину ошибки,
проверить объект в S3 и вызвать `/images/{id}/reindex`.

Удаление с `object_removed=false` можно повторить тем же DELETE. Запись уже
не видна в поиске. Команда `docker compose down` сохраняет тома; вариант
с `--volumes` удаляет данные и не нужен для обычной остановки.

Перед внешним развёртыванием заменить демонстрационные секреты, настроить
TLS, квоты на дисковое пространство, ограничения запросов, резервные копии
и очистку старых записей. Отдельно проверить качество на изображениях
конкретного каталога и подобрать пороги по независимой разметке.

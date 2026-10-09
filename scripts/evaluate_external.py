"""Внешняя проверка текущего API; признаки и объекты сохраняются в изолированной БД/S3."""

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import logging
import resource
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    positives = [r for r in rows if r["positive"]]
    negatives = [r for r in rows if not r["positive"]]
    tp = sum(r["true_match"] for r in rows)
    predictions = sum(r["matches"] for r in rows)
    fp = predictions - tp
    negative_candidates = sum(r["candidates"] for r in rows) - sum(
        r["candidate_hit"] for r in positives
    )
    return {
        "same_class_negative_pairs": sum(r.get("same_class_negative_pairs", 0) for r in rows),
        "same_class_false_positives": sum(r.get("same_class_false_positives", 0) for r in rows),
        "different_class_negative_pairs": sum(
            r.get("different_class_negative_pairs", 0) for r in rows
        ),
        "different_class_false_positives": sum(
            r.get("different_class_false_positives", 0) for r in rows
        ),
        "geometry_true_positives": sum(r.get("geometry_true_match", False) for r in rows),
        "geometry_false_positives": sum(
            r.get("geometry_matches", 0) - r.get("geometry_true_match", False) for r in rows
        ),
        "geometry_precision": (
            sum(r.get("geometry_true_match", False) for r in rows)
            / sum(r.get("geometry_matches", 0) for r in rows)
        )
        if sum(r.get("geometry_matches", 0) for r in rows)
        else None,
        "geometry_recall": sum(r.get("geometry_true_match", False) for r in positives)
        / len(positives)
        if positives
        else None,
        "queries": len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": len(positives) - tp,
        "candidate_recall": sum(r["candidate_hit"] for r in positives) / len(positives)
        if positives
        else None,
        "recall": tp / len(positives) if positives else None,
        "precision": tp / predictions if predictions else None,
        "negative_query_false_positive_rate": sum(r["matches"] > 0 for r in negatives)
        / len(negatives)
        if negatives
        else None,
        "verified_candidate_false_positive_rate": fp / negative_candidates
        if negative_candidates
        else None,
        "p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)) if rows else None,
        "p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)) if rows else None,
    }


def corpus(cache):
    manifest = json.loads((ROOT / "docs/external/source.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    for name, checksum in manifest["files"].items():
        path = cache / name
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
                + manifest["revision"]
                + "/data/fashion/"
                + name
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read(10_000_000))
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Dataset checksum mismatch: " + name)
    pixels = np.frombuffer(
        gzip.decompress((cache / "t10k-images-idx3-ubyte.gz").read_bytes()),
        dtype=np.uint8,
        offset=16,
    ).reshape(10000, 28, 28)
    labels = np.frombuffer(
        gzip.decompress((cache / "t10k-labels-idx1-ubyte.gz").read_bytes()),
        dtype=np.uint8,
        offset=8,
    )
    return manifest, pixels, labels


async def evaluate(cache, output):
    import httpx
    from sqlalchemy import text

    import imagetwin.main as api
    from imagetwin import storage
    from imagetwin.config import settings
    from imagetwin.db import engine
    from imagetwin.vision import encoder, normalize
    from scripts.fixtures import variants

    if not settings.testing or not settings.database_url.endswith("_test"):
        raise ValueError("External evaluation requires isolated *_test database and TESTING=true")
    if settings.s3_bucket != "imagetwin-test":
        raise ValueError("External evaluation requires isolated S3 bucket")
    logging.getLogger("api").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    manifest, pixels, labels = corpus(cache)
    features = {}
    data = {}
    for index in list(range(499)) + manifest["negative_source_indices"]:
        image = Image.fromarray(pixels[index]).resize((224, 224), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        data[index] = normalize(buffer.getvalue())[0]
        features[index] = encoder().extract(data[index])
    signatures = {(f["phash"], f["descriptors"]): i for i, f in features.items()}
    originals = api.compare
    captured = []

    def observed(first, second):
        result = originals(first, second)
        captured.append(
            (
                second["vector"],
                {**result, "source_index": signatures[(second["phash"], second["descriptors"])]},
            )
        )
        return result

    storage.ensure_bucket()
    report = {
        "protocol": manifest,
        "encoder_version": encoder().version,
        "thresholds_changed": False,
        "sizes": {},
    }
    api.compare = observed
    owned_objects = []
    collection_ids = []
    uid = uuid4()
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO users(id,email,password_hash) VALUES (:id,:email,:hash)"),
                {"id": uid, "email": str(uid) + "@evaluation.local", "hash": "unused"},
            )
        from datetime import UTC, datetime, timedelta

        import jwt

        now = datetime.now(UTC)
        token = jwt.encode(
            {
                "sub": str(uid),
                "iat": now,
                "exp": now + timedelta(hours=1),
                "iss": "imagetwin",
                "aud": "imagetwin",
            },
            settings.jwt_secret,
            algorithm="HS256",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://evaluation"
        ) as client:
            for size in manifest["catalog_sizes_including_query"]:
                cid = uuid4()
                collection_ids.append(cid)
                ids = {}
                async with engine.begin() as conn:
                    await conn.execute(
                        text("INSERT INTO collections(id,user_id,name) VALUES (:id,:uid,:name)"),
                        {"id": cid, "uid": uid, "name": "external-" + str(size)},
                    )

                async def persist(index, content, feature, image_id, cid=cid):
                    key = "external-evaluation/" + str(cid) + "/" + str(image_id)
                    owned_objects.append(key)
                    normalized, pixel_sha, width, height = normalize(content)
                    storage.client().put_object(Bucket=settings.s3_bucket, Key=key, Body=normalized)
                    async with engine.begin() as conn:
                        await conn.execute(
                            text("""INSERT INTO images(id,collection_id,pixel_sha256,source_sha256,object_key,width,height,status,phash,embedding,points,descriptors,contrast,encoder_version)
                        VALUES (:id,:cid,:pixel_sha,:sha,:key,:width,:height,'ready',CAST(CAST(:phash AS text) AS bit(64)),CAST(:vector AS vector),CAST(:points AS jsonb),:descriptors,:contrast,:version)"""),
                            {
                                "id": image_id,
                                "cid": cid,
                                "sha": hashlib.sha256(content).hexdigest(),
                                "pixel_sha": pixel_sha,
                                "key": key,
                                "width": width,
                                "height": height,
                                "phash": feature["phash"],
                                "vector": json.dumps(feature["vector"]),
                                "points": json.dumps(feature["points"]),
                                "descriptors": feature["descriptors"],
                                "contrast": feature["contrast"],
                                "version": feature["encoder_version"],
                            },
                        )

                for index in range(size - 1):
                    ids[index] = uuid4()
                    await persist(index, data[index], features[index], ids[index])
                rows = []
                queries = [
                    (i, name, value, True)
                    for i in manifest["positive_source_indices"]
                    for name, value in variants(data[i]).items()
                ]
                queries += [
                    (i, "unseen_original", data[i], False)
                    for i in manifest["negative_source_indices"]
                ]
                for index, transform, content, positive in queries:
                    query_id = uuid4()
                    feature = encoder().extract(normalize(content)[0])
                    await persist(index, content, feature, query_id)
                    captured.clear()
                    started = time.perf_counter()
                    response = await client.get(
                        f"/images/{query_id}/duplicates?limit=80",
                        headers={"Authorization": "Bearer " + token},
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    result = response.json()
                    target = str(ids[index]) if positive else None
                    # PostgreSQL vector text roundtrip is float32; compare with tolerance.
                    hit = positive and any(
                        np.allclose(v, features[index]["vector"], atol=1e-7) for v, _ in captured
                    )
                    rows.append(
                        {
                            "same_class_negative_pairs": sum(
                                int(labels[e["source_index"]]) == int(labels[index])
                                and not (positive and e["source_index"] == index)
                                for _, e in captured
                            ),
                            "same_class_false_positives": sum(
                                e["duplicate"]
                                and int(labels[e["source_index"]]) == int(labels[index])
                                and not (positive and e["source_index"] == index)
                                for _, e in captured
                            ),
                            "different_class_negative_pairs": sum(
                                int(labels[e["source_index"]]) != int(labels[index])
                                for _, e in captured
                            ),
                            "different_class_false_positives": sum(
                                e["duplicate"]
                                and int(labels[e["source_index"]]) != int(labels[index])
                                for _, e in captured
                            ),
                            "source_index": index,
                            "class": int(labels[index]),
                            "transform": transform,
                            "positive": positive,
                            "candidate_hit": bool(hit),
                            "candidates": len(captured),
                            "matches": len(result["matches"]),
                            "true_match": any(m["image_id"] == target for m in result["matches"]),
                            "geometry_true_match": positive
                            and any(
                                np.allclose(v, features[index]["vector"], atol=1e-7)
                                and e["geometric_inliers"] >= 12
                                and e["inlier_ratio"] >= 0.5
                                and e["matched_area"] >= 0.06
                                for v, e in captured
                            ),
                            "geometry_matches": sum(
                                e["geometric_inliers"] >= 12
                                and e["inlier_ratio"] >= 0.5
                                and e["matched_area"] >= 0.06
                                for _, e in captured
                            ),
                            "latency_ms": elapsed,
                        }
                    )
                    async with engine.begin() as conn:
                        await conn.execute(
                            text("DELETE FROM images WHERE id=:id"), {"id": query_id}
                        )
                report["sizes"][str(size)] = {
                    "overall": summarize(rows),
                    "per_transform": {
                        name: summarize([r for r in rows if r["transform"] == name])
                        for name in sorted({r["transform"] for r in rows})
                    },
                    "query_rows": rows,
                    "process_peak_rss_kib_linux": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                }
                print(json.dumps({size: report["sizes"][str(size)]["overall"]}), flush=True)
        report["process_peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        api.compare = originals
        async with engine.begin() as conn:
            for cid in collection_ids:
                await conn.execute(text("DELETE FROM images WHERE collection_id=:id"), {"id": cid})
                await conn.execute(text("DELETE FROM collections WHERE id=:id"), {"id": cid})
            await conn.execute(text("DELETE FROM users WHERE id=:id"), {"id": uid})
        for key in owned_objects:
            storage.client().delete_object(Bucket=settings.s3_bucket, Key=key)
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/external/evaluation.json")
    args = parser.parse_args()
    asyncio.run(evaluate(args.cache, args.output))

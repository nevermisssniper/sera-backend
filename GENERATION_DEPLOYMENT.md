# Generation service contract

Deploy with the matching `sera-frontend` fix and its additive database SQL.

- Frontend and API: identical `GENERATION_SERVICE_TOKEN` (random, at least 32 bytes).
- Frontend, API and workers: identical, separate `GENERATION_CALLBACK_SECRET`.
- API and workers: `GENERATION_CALLBACK_URL` pointing to the frontend's HTTPS `/api/generate-callback` endpoint. No client-selected callback URLs are accepted.
- Keep the existing Redis and Uploadcare configuration.

`POST /generate` requires `Authorization: Bearer <service token>`, a UUID `jobId`, a `userId`, and a nonempty list of source image UUIDs. The job ID is the Celery task ID. Drain old jobs before upgrading; their callbacks will not authenticate against the new frontend.

Callbacks carry `{userId, jobId, generatedUuids}` and two headers: `X-Generation-Timestamp` (Unix seconds) and `X-Generation-Signature` (hex HMAC-SHA256 of `timestamp + "." + exact UTF-8 JSON body`). Keep clocks synchronized. The frontend accepts a five-minute clock window and only the matching pending job; completed-job retries do not write again.

Training uses a private image directory and class-image directory per job. The wrapper returns the final checkpoint's sample directory; missing or empty outputs fail without replacing a gallery. Callback delivery retries transient failures without retraining; redirects are rejected. If delivery exhausts retries, the task fails and the uploaded images remain in Uploadcare. An operator can recover them with a fresh signed callback for the same pending job. For failed training, confirm the job has stopped before clearing its pending ID in the frontend database and retrying.

Run focused tests with `python -m pytest -q tests` (pytest, requests, fastapi, httpx and pydantic required). Tests mock Celery and GPU training; no production service or paid generation is invoked.

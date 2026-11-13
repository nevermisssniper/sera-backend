# Sera Engine — runnable workbench

Walk through a generation job and see where authentication, training and delivery fit.

## Open the interface

Python 3.10+; no npm install, API keys, model download or GPU required for the demo.

```sh
python serve_demo.py
```

Open <http://127.0.0.1:8000>. To use another port: `python serve_demo.py --port 8001`.
Serve `demo/` over HTTP rather than opening `index.html` as a file. The server binds only to localhost.

## What is implemented

The browser demonstrates job states and failure cases. It does not start Celery or train a model. Follow `GENERATION_DEPLOYMENT.md` for the real authenticated service, callback secrets, Redis and GPU worker. The earlier callback and per-job output fixes are inherited by this branch.

Interface capabilities: Failure scenarios, Job timeline, Request manifest.

## Core workflow

From the repository root (install the project's existing dependencies first for model-backed commands):

```sh
python -m pytest tests/test_generation.py -q
```

## Verify

The browser's calculation tests use Node 18+ and no third-party packages:

```sh
node --test demo/core.test.cjs
python -m pytest tests/test_generation.py -q
```

The GitHub demo workflow checks browser calculations and the dependency-free core tests where present. Full original model/service suites have separate dependencies; passing demo checks does not certify GPU training or external services.

## Before live use

Real GPU training, model dependencies and callback delivery need deployment smoke tests.

Local browser data belongs to this browser profile. Export records before clearing it or changing devices. Photo previews stay in memory; exported briefs contain filenames, not uploaded image bytes. There are no payment or outbound model calls in this demo.

## Provenance

Workbench and starter additions: AI-assisted implementation, 2026-11-13, for Ray's review. Existing repository history and upstream attribution are preserved. New starter directions are proposals where the GitHub repository had no code.

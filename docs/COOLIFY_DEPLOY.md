# Deploy the OCR Benchmark on Coolify (CPU)

Click-by-click guide to run the VLM/OCR benchmark on your Coolify server, on CPU.

> **Why this works on Coolify but stalled on the Mac:** vLLM's CPU backend needs
> oneDNN, which works on **Linux x86_64** but fails on Apple Silicon (arm64),
> where it falls back to a path too slow to even finish model loading. A normal
> x86_64 Coolify host with adequate RAM runs these models on CPU fine.

## Two containers, two jobs (read this)

This deployment has **two** containers:

| Container | Image | Job | Do you run commands here? |
| :-- | :-- | :-- | :-- |
| `vllm` | `vllm/vllm-openai-cpu` | Serves ONE model over HTTP. That's all it does. | **No.** `document-ocr-bench` is not installed here — running it gives "not found". |
| `bench` | `innovantics/document-ocr-bench` | Runs the benchmark CLI, calls `vllm` over HTTP, scores, reports. | **Yes — always here.** |

So: the `vllm` service just answers model requests; you drive everything from the
**`bench`** container's terminal.

---

## 0. Check your Coolify host first (30 seconds)

SSH to the server (or use Coolify's server terminal) and run:

```bash
uname -m         # must print: x86_64   (if aarch64 → CPU path is slow; use a GPU host)
free -g          # total RAM. Need ~6 GB for the 0.9B models (GLM-OCR/PaddleOCR-VL),
                 # ~24-32 GB for the 7-8B ones (Qwen3-VL-8B / olmOCR-7B)
nproc            # more cores = faster CPU inference
```

Run **one model at a time**. Plan RAM around the largest model you'll serve.

---

## 1. Push this repo to Git (Coolify deploys from Git)

From your Mac, in the project folder:

```bash
cd /Users/danielkomolafe/Documents/document-ocr
git add -A
git commit -m "Document OCR benchmark harness + Coolify CPU deploy"
git branch -M main
```

Create an empty repo on GitHub (or GitLab/Gitea), then:

```bash
git remote add origin https://github.com/<you>/document-ocr.git
git push -u origin main
```

A **private** repo is fine — Coolify connects via its GitHub App or a deploy key.

---

## 2. Create the project in Coolify

1. Open your Coolify dashboard.
2. **Projects → + Add** → name it `document-ocr` → **Continue**. Coolify creates a
   `production` environment.
3. (First time only) Connect your Git: **Sources → + Add → GitHub App** and install
   it on the repo. Public repos can instead be added by URL in the next step.

## 3. Add the resource (Docker Compose from your repo)

1. Inside the project's `production` environment → **+ New Resource**.
2. Choose your source:
   - **Private Repository (GitHub App)** → pick `document-ocr`, or
   - **Public Repository** → paste the repo URL.
3. **Branch:** `main`.
4. **Build Pack:** select **Docker Compose**.
5. **Docker Compose Location:** `/docker-compose.coolify.yml`
6. Click **Continue / Save**. Coolify parses the compose and shows two services:
   `vllm` and `bench`.

## 4. Set environment variables

Open the resource → **Environment Variables** → add these (Coolify feeds them into
the compose). Start with the smallest model:

```
VLLM_IMAGE=vllm/vllm-openai-cpu:latest-x86_64
VLLM_MODEL=zai-org/GLM-OCR
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_NUM_SEQS=1
VLLM_CPU_KVCACHE_SPACE=8
OCR_SYNC_TIMEOUT_MS=600000
HUGGING_FACE_HUB_TOKEN=          # optional; only if a model download is rate-limited
```

> **Persistent storage:** Coolify auto-creates the named volumes `hf-cache`
> (model weights — keep this so models don't re-download) and `bench-data`
> (generated samples + results). Confirm them under the resource's **Storages** tab.

> **Domains/ports:** none required — this is internal tooling. Leave domains empty.

## 5. Deploy

1. Click **Deploy**.
2. Watch the deploy + container logs:
   - `bench` builds quickly and goes **Up** (it idles on `sleep infinity`).
   - `vllm` pulls the CPU image, then **downloads the model weights and loads on
     CPU** — first run can take 10–30 min. It is **healthy** once
     `http://localhost:8000/health` returns 200 (Coolify shows the service green).
3. The deployment may show "degraded" while `vllm` is still downloading — that's
   expected; watch the `vllm` logs until you see the API server start listening.

## 6. Run the benchmark (in the `bench` container)

Open the **`bench`** service → **Terminal** (Execute Command).

**First, confirm the model is actually reachable** — this is a real health check
and prevents the "VLM runs instantly / Tesseract always wins" trap (a VLM that
returns instantly is *failing fast*, not inferring):

```bash
document-ocr-bench providers
```

You want the served model's provider to show **Avail = yes**. If instead you see:
- `glm-ocr: cannot reach http://vllm:8000/v1 (ConnectError)` → `vllm` isn't ready
  yet (still downloading/loading) or crashed. Wait, or check the `vllm` logs.
- `glm-ocr: endpoint serves ['X'], not 'Y'` → you're running the wrong provider
  for the served model. Use the mapping in §7 (run the provider matching `VLLM_MODEL`).

Only once the matching provider shows **yes**, run the benchmark:

```bash
document-ocr-bench gen-samples
document-ocr-bench run --providers glm-ocr,tesseract
```

(`glm-ocr` because `VLLM_MODEL=zai-org/GLM-OCR`. Always run the provider that
matches the served model — see the mapping below.) If a provider errors on samples
mid-run, the CLI now prints a loud `⚠ <provider>: N/N samples errored` warning so
you never mistake a broken endpoint for a real low score.

View the report:

```bash
document-ocr-bench report "$(ls -dt benchmarks/document-ocr/results/*/ | head -1)"
cat "$(ls -dt benchmarks/document-ocr/results/*/ | head -1)/report.md"
```

**No Terminal button in your Coolify version?** SSH to the server instead:

```bash
docker ps --filter name=bench --format '{{.Names}}'       # find the container name
docker exec -it <bench-container-name> document-ocr-bench gen-samples
docker exec -it <bench-container-name> document-ocr-bench run --providers glm-ocr,tesseract
```

## 7. Benchmark the other models (one at a time)

For each model: change `VLLM_MODEL` in **Environment Variables**, **Redeploy**, wait
for `vllm` healthy, then run the **matching provider**.

| `VLLM_MODEL` | Run provider | Approx RAM (CPU) |
| :-- | :-- | :-- |
| `zai-org/GLM-OCR` | `glm-ocr` | ~6 GB |
| `PaddlePaddle/PaddleOCR-VL` | `paddleocr-vl` | ~6 GB |
| `Qwen/Qwen3-VL-8B-Instruct` | `qwen-vl` | ~24–32 GB |
| `allenai/olmOCR-2-7B-1025` | `olmocr` | ~20–28 GB |

```bash
# always include tesseract as the baseline:
document-ocr-bench run --providers <provider>,tesseract
```

Each run writes a new `results/<timestamp>/` (kept in the `bench-data` volume), so
you accumulate a per-model evidence set. To compare everything in one report later,
copy the per-run `results.json` files together and re-run `document-ocr-bench report`.

## 8. Retrieve results off the server (optional)

```bash
# on the Coolify host:
docker cp <bench-container-name>:/app/benchmarks/document-ocr/results ./ocr-results
```

---

## Troubleshooting

| Symptom | Cause / Fix |
| :-- | :-- |
| `vllm` never goes healthy, logs show download progress | Still pulling weights — wait; first load is slow on CPU. |
| `vllm` exits **137** | OOM. Give the host/container more RAM, lower `VLLM_MAX_MODEL_LEN`, or serve a smaller model. Don't run two model servers at once. |
| `vllm` logs `oneDNN ... could not create primitive` then hangs | Host is **arm64**. Use an x86_64 host or a GPU host. |
| Provider run errors with model-name mismatch | The provider you ran doesn't match `VLLM_MODEL`. Use the mapping table in §7. |
| `document-ocr-bench: command not found` in exec | You exec'd the wrong container — target the **bench** container, not `vllm`. |
| Download rate-limited | Set `HUGGING_FACE_HUB_TOKEN` and redeploy. |

## Going faster / to production

CPU is fine for the benchmark spike. For production throughput, serve on a GPU host
with `docker-compose.gpu.yml` (override the image to `vllm/vllm-openai:latest` and
attach a GPU). The harness and provider contracts are identical — only the model
server changes.

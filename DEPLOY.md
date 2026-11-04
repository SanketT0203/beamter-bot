# Deploying Beamter-Bot to Google Cloud Run

```
Browser ──HTTPS──▶ Cloud Run service  ◀── Cloud Scheduler (keep-alive, every 3 days)
                     ├─ chat page + API
                     ├─ both models baked into the image (CPU)
                     ├──▶ Qdrant Cloud   read-only key
                     └──▶ Anthropic API

Your PC (offline) ──▶ python -m beamter.ingest_job ──▶ Qdrant Cloud   (the only writer)
```

The service never sees your PDFs and cannot change the index. Ingestion stays on your PC.

The commands below are for **Google Cloud Shell** (the `>_` button in the Cloud Console): it is
bash, already logged in, and has `gcloud` and `git`. No local install, no PowerShell quoting.

---

## 0. Before you start

- [ ] A Google Cloud project with billing enabled (the free-trial credit counts).
- [ ] The code in its own public GitHub repo (part A).
- [ ] A **read-only** API key for your Qdrant cluster (Qdrant Cloud → your cluster → *API Keys* → create, access *read-only*). Keep the existing read/write key for ingestion on your PC only.
- [ ] A **monthly spend limit** in the Anthropic console (Settings → Limits). The site is public, so this is the real cost ceiling. Google budgets only send emails; they do not stop spending.

## A. Put the code in its own repo (on your PC, PowerShell)

```powershell
cd D:\ai-engineer-journey\beamter-bot-cloud
git init -b main
git add .
git status
```
**Read the `git status` list before committing.** It must not contain `.env`, `corpus/`, `.cache/` or
`eval_results/` (all are in `.gitignore`). It must contain `uv.lock`, `Dockerfile`, `src/`, `data/`.
```powershell
git commit -m "Beamter-Bot: multilingual RAG over German official documents"
gh repo create beamter-bot --public --source . --push
```
(No GitHub CLI? Create an empty public repo on github.com, then
`git remote add origin https://github.com/<you>/beamter-bot.git` and `git push -u origin main`.)

## B. Set up the project (Cloud Shell)

```bash
git clone https://github.com/<you>/beamter-bot && cd beamter-bot

export PROJECT_ID="<your-project-id>"
export REGION="europe-west3"            # Frankfurt: close to your EU Qdrant cluster
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/beamter/beamter-bot"
gcloud config set project "$PROJECT_ID"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com cloudscheduler.googleapis.com

gcloud artifacts repositories create beamter --repository-format=docker \
  --location="$REGION" --description="Beamter-Bot images"
```

## C. Secrets (typed into a hidden prompt: never in a file, a command line or a chat)

```bash
read -rsp "Anthropic API key: " V; echo
printf '%s' "$V" | gcloud secrets create anthropic-api-key --replication-policy=user-managed --locations="$REGION" --data-file=-
unset V

read -rsp "Qdrant READ-ONLY key: " V; echo
printf '%s' "$V" | gcloud secrets create qdrant-read-key --replication-policy=user-managed --locations="$REGION" --data-file=-
unset V
```

## D. Build the image (about 10 to 20 minutes the first time)

```bash
gcloud builds submit --region="$REGION" --tag "$IMAGE:v1" --machine-type=e2-highcpu-8 --timeout=45m
```
This installs CPU-only PyTorch, downloads both models into the image, then re-loads them with the
network switched off. If that second step fails, the build fails, not your first user.

## E. A runtime identity that can only read two secrets, then deploy

```bash
gcloud iam service-accounts create beamter-run --display-name="Beamter-Bot runtime"
export SA="beamter-run@$PROJECT_ID.iam.gserviceaccount.com"
for s in anthropic-api-key qdrant-read-key; do
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done

gcloud run deploy beamter-bot \
  --image "$IMAGE:v1" --region "$REGION" --service-account "$SA" \
  --allow-unauthenticated \
  --cpu 2 --memory 4Gi --cpu-boost \
  --min-instances 0 --max-instances 2 \
  --concurrency 8 --timeout 120 --port 8080 \
  --set-env-vars "QDRANT_URL=https://<your-cluster-host>.cloud.qdrant.io,BEAMTER_RATE_LIMIT_PER_MIN=20,BEAMTER_MAX_CONCURRENCY=4" \
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,QDRANT_API_KEY=qdrant-read-key:latest" \
  --startup-probe "httpGet.path=/readyz,httpGet.port=8080,initialDelaySeconds=5,periodSeconds=5,timeoutSeconds=3,failureThreshold=60"
```

| Flag | Why |
|---|---|
| `--cpu 2 --memory 4Gi` | Both models need about 2 GB; reranking runs on the CPU. `OMP_NUM_THREADS=2` in the Dockerfile matches `--cpu 2`: change them together. |
| `--cpu-boost` | Extra CPU while the instance starts: shorter cold starts. |
| `--startup-probe … /readyz` | No traffic until models and index are loaded (up to 5 minutes). |
| `--min-instances 0` | Pay only while requests run. Cost: the first visit after idle waits ~1 minute (the page shows "Starting …"). |
| `--max-instances 2`, `--concurrency 8` | Hard cap on spending even if someone hammers the public URL. |
| `BEAMTER_RATE_LIMIT_PER_MIN=20` | Per client address, per instance: a cost brake, not an exact quota. |
| no `SERVICE_API_KEY` | Public demo. To lock it, add the secret `service-api-key` and `SERVICE_API_KEY=service-api-key:latest` to `--set-secrets`. |

## F. Check it

```bash
URL=$(gcloud run services describe beamter-bot --region "$REGION" --format='value(status.url)')
echo "$URL"
curl -s "$URL/healthz"; echo
curl -s "$URL/readyz"; echo
curl -s -X POST "$URL/ask" -H 'Content-Type: application/json' \
  -d '{"question": "Wie melde ich meinen Wohnsitz an?"}'
```
Then open `$URL` in a browser. The first request after idle is slow (cold start): that is expected.

## G. Budget alert (emails you; it does not stop spending)

Console → Billing → *Budgets & alerts* → *Create budget*: amount e.g. 25 EUR/month, alerts at 50%, 90%, 100%.
(CLI: `gcloud billing budgets create --billing-account=<ID> --display-name=beamter --budget-amount=25EUR --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0`)

## H. Keep the free Qdrant cluster from being suspended

The free tier suspends a cluster after about a week without activity. A scheduled request to the
service starts it (if idle), makes it read the index from Qdrant, and checks the whole chain:

```bash
gcloud scheduler jobs create http beamter-keepalive --location="$REGION" \
  --schedule="0 6 */3 * *" --time-zone="Europe/Berlin" \
  --uri="$URL/readyz" --http-method=GET --attempt-deadline=300s
```
No key is stored in the job.

---

## Day to day

| I want to … | Do this |
|---|---|
| Ship a code change | Build a new tag (`:v2`) as in D, then `gcloud run deploy beamter-bot --image "$IMAGE:v2" --region "$REGION"` (other settings are kept) |
| Roll back | `gcloud run services update-traffic beamter-bot --region "$REGION" --to-revisions=<REVISION>=100` (list: `gcloud run revisions list --service beamter-bot --region "$REGION"`) |
| Update the documents | On your PC: `uv run python -m beamter.ingest_job`. Warm instances switch to the new index within `BEAMTER_RELOAD_CHECK_S` (60 s) when they get traffic; new instances load it at start. |
| Keep it warm for an interview | `gcloud run services update beamter-bot --region "$REGION" --min-instances 1` (billed continuously; set back to `0` afterwards) |
| Read the logs | `gcloud run services logs read beamter-bot --region "$REGION" --limit 50` |
| See memory use | Console → Cloud Run → beamter-bot → *Metrics* → *Memory utilization*. Above ~80%: redeploy with `--memory 8Gi`. |

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| Build fails with a permission error | The build service account needs Artifact Registry write access. Paste the exact message; it names the missing role. |
| Deploy says the revision "failed the startup probe" | Open the logs: the line `Startup load failed: …` says why (wrong `QDRANT_URL`, wrong key, no index yet: run the ingest first). |
| `/readyz` shows `failed` with a 403 from Qdrant | The read-only key lacks a permission the service needs; tell me the message. |
| Secret access denied | The service account is missing `secretAccessor` on that secret (step E). |
| "Memory limit exceeded" in the logs | Raise `--memory` (see above). |
| Everyone gets 429 | `BEAMTER_RATE_LIMIT_PER_MIN` is too low for your traffic, or many people share one office IP. |

@echo off
rem Runs the retrieval evaluation on the tuned set and the held-out set, saves full output.
rem Usage (from the beamter-bot-cloud folder):  scripts\run_eval.cmd
cd /d "%~dp0.."
if not exist eval_results mkdir eval_results
echo 1/2 tuned set (data\golden.json) ...
uv run python -m beamter.evaluation --golden data\golden.json --modes dense dense_rerank hybrid_rerank --show-misses > eval_results\tuned.txt 2>&1
echo 2/2 held-out set (data\heldout.json) - this is the number to quote ...
uv run python -m beamter.evaluation --golden data\heldout.json --modes dense dense_rerank hybrid_rerank --show-misses > eval_results\heldout.txt 2>&1
echo Done. Send me eval_results\tuned.txt and eval_results\heldout.txt

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from openai import OpenAI

METRIC_KEYS = ["SC", "TT", "TC", "TP", "GC", "BN", "MB"]
ALLOWED_ORDERS = {
    "A>B>C",
    "A>C>B",
    "B>A>C",
    "B>C>A",
    "C>A>B",
    "C>B>A",
}


class RankingParseError(ValueError):
    pass


@dataclass
class EvalRecord:
    sample_id: str
    sample_path: str
    valid: bool
    parsed: dict[str, str]
    raw_response: str
    attempts: int
    error: str | None


def _resolve_base_url(explicit_base_url: str | None) -> str | None:
    if explicit_base_url:
        return explicit_base_url
    env_base_url = os.getenv("OPENAI_BASE_URL")
    if env_base_url:
        return env_base_url
    return None


def _resolve_api_token(token_env: str) -> str:
    token = os.getenv(token_env)
    if token:
        return token
    raise click.BadParameter(
        f"API token missing. Set environment variable {token_env}.",
        param_hint="token_env",
    )


def _build_prompt(doc_text: str) -> tuple[str, str]:
    system_prompt = (
        "You are an evaluator of segmentation quality. "
        "Follow output format exactly. No explanations."
    )

    user_prompt = f"""
You will compare three anonymous annotators A, B, C for one lecture segmentation sample.

Output must contain exactly 7 lines, in this exact order and with this exact key format:
SC:X>Y>Z
TT:X>Y>Z
TC:X>Y>Z
TP:X>Y>Z
GC:X>Y>Z
BN:X>Y>Z
MB:X>Y>Z

Rules:
- X,Y,Z must be a strict permutation of A,B,C with no ties.
- Rank from best to worst for the metric: X is the best annotator, Y is second, and Z is third.
- A higher rank always means better segmentation quality, not more errors.
- Do not output any extra text.
- Do not use markdown.
- If uncertain, still output a forced ranking.

Metric definitions (avoid confusion):
- SC (Semantic coherence): how well the sentences within each segment stay on one coherent theme; higher is better.
- TT (Topic transition): how well boundaries align with genuine topic shifts; higher is better.
- TC (Topic completeness): how well each topic is kept together instead of being unnecessarily split; higher is better.
- TP (Topic purity): how well each segment contains one topic without mixing unrelated topics; higher is better.
- GC (Granularity consistency): how consistently the lecture uses an appropriate segmentation granularity; higher is better.
- BN (Boundary necessity): how well the annotator places boundaries only where a split is justified by a meaningful topic change; higher is better.
- MB (Missing boundary): does the annotator miss important boundaries that are needed to separate distinct topics? Rank the annotator with fewer missed important boundaries higher.

Input sample:
{doc_text}
""".strip()
    return system_prompt, user_prompt


def _chat_complete(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    if model.startswith("gpt-5") or "codex" in model.lower():
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            timeout=timeout_seconds,
        )
        content = response.output_text
        if not content:
            raise RuntimeError("Model returned empty content")
        return str(content)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=timeout_seconds,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned empty content")
    return str(content)


def _parse_rank_response(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != len(METRIC_KEYS):
        raise RankingParseError(f"Expected {len(METRIC_KEYS)} non-empty lines, got {len(lines)}")

    parsed: dict[str, str] = {}
    expected_keys = METRIC_KEYS

    for idx, line in enumerate(lines):
        if ":" not in line:
            raise RankingParseError(f"Missing ':' in line {idx + 1}: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().replace(" ", "")

        if key != expected_keys[idx]:
            raise RankingParseError(
                f"Expected key '{expected_keys[idx]}' at line {idx + 1}, got '{key}'"
            )
        if value not in ALLOWED_ORDERS:
            raise RankingParseError(f"Invalid ranking '{value}' at line {idx + 1}")
        parsed[key] = value

    return parsed


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_valid_csv(path: Path, records: list[EvalRecord]) -> None:
    fieldnames = ["sample_id", "sample_path", *METRIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if not rec.valid:
                continue
            row = {
                "sample_id": rec.sample_id,
                "sample_path": rec.sample_path,
            }
            for key in METRIC_KEYS:
                row[key] = rec.parsed[key]
            writer.writerow(row)


@click.command()
@click.option(
    "--docs_dir",
    type=str,
    default="artifacts/mitfld_llm_judging_inputs/docs",
    show_default=True,
    help="Directory of per-sample .txt documents.",
)
@click.option(
    "--mapping_config",
    type=str,
    default="artifacts/mitfld_llm_judging_inputs/run_config.json",
    show_default=True,
    help="Config file containing annotator mapping from the preprocessing step.",
)
@click.option(
    "--output_dir",
    type=str,
    default="artifacts/mitfld_llm_eval",
    show_default=True,
    help="Output directory for responses and parsed rankings.",
)
@click.option("--model", type=str, default="gpt-5.3-codex", show_default=True)
@click.option("--base_url", type=str, default=None, help="Optional OpenAI-compatible base URL.")
@click.option("--token_env", type=str, default="OPENAI_API_KEY", show_default=True)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--timeout_seconds", type=int, default=120, show_default=True)
@click.option("--max_retries", type=int, default=1, show_default=True, help="Retries after parse failure.")
@click.option("--limit", type=int, default=0, show_default=True, help="If > 0, only evaluate first N samples.")
@click.option("--sleep_seconds", type=float, default=0.0, show_default=True)
def main(
    docs_dir: str,
    mapping_config: str,
    output_dir: str,
    model: str,
    base_url: str | None,
    token_env: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    limit: int,
    sleep_seconds: float,
) -> None:
    docs_path = Path(docs_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        raise click.BadParameter(f"docs_dir not found: {docs_dir}", param_hint="docs_dir")
    if timeout_seconds <= 0:
        raise click.BadParameter("timeout_seconds must be > 0", param_hint="timeout_seconds")
    if max_retries < 0:
        raise click.BadParameter("max_retries must be >= 0", param_hint="max_retries")
    if limit < 0:
        raise click.BadParameter("limit must be >= 0", param_hint="limit")
    if not 0.0 <= temperature <= 2.0:
        raise click.BadParameter("temperature must be in [0, 2]", param_hint="temperature")

    resolved_base_url = _resolve_base_url(base_url)
    token = _resolve_api_token(token_env)
    client_kwargs: dict[str, Any] = {"api_key": token}
    if resolved_base_url is not None:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    annotator_mapping: dict[str, Any] = {}
    mapping_path = Path(mapping_config)
    if mapping_path.exists():
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        annotator_mapping = mapping_payload.get("annotator_mapping", {})

    doc_files = sorted(docs_path.glob("*.txt"))
    if not doc_files:
        raise click.BadParameter(f"No .txt files under {docs_dir}", param_hint="docs_dir")
    if limit > 0:
        doc_files = doc_files[:limit]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    responses_jsonl = out_root / f"responses_{run_stamp}.jsonl"
    valid_csv = out_root / f"parsed_valid_{run_stamp}.csv"
    run_config = out_root / f"eval_run_config_{run_stamp}.json"

    records: list[EvalRecord] = []
    interrupted = False
    try:
        for idx, doc_file in enumerate(doc_files, start=1):
            sample_id = doc_file.stem
            doc_text = doc_file.read_text(encoding="utf-8")
            system_prompt, user_prompt = _build_prompt(doc_text)

            valid = False
            parsed: dict[str, str] = {}
            raw_response = ""
            error_text: str | None = None
            attempts = 0

            for attempt in range(max_retries + 1):
                attempts = attempt + 1
                try:
                    raw_response = _chat_complete(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        timeout_seconds=timeout_seconds,
                    )
                    parsed = _parse_rank_response(raw_response)
                    valid = True
                    error_text = None
                    break
                except Exception as exc:  # Keep broad to retry API or parse failures.
                    error_text = str(exc)

            record = EvalRecord(
                sample_id=sample_id,
                sample_path=str(doc_file),
                valid=valid,
                parsed=parsed,
                raw_response=raw_response,
                attempts=attempts,
                error=error_text,
            )
            records.append(record)

            # Keep completed API responses even if a long run is interrupted.
            with responses_jsonl.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "sample_path": record.sample_path,
                            "valid": record.valid,
                            "parsed": record.parsed,
                            "raw_response": record.raw_response,
                            "attempts": record.attempts,
                            "error": record.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()

            status = "valid" if valid else f"invalid: {error_text}"
            print(f"[{idx}/{len(doc_files)}] {sample_id}: {status}")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving completed results...")

    _write_jsonl(
        responses_jsonl,
        [
            {
                "sample_id": rec.sample_id,
                "sample_path": rec.sample_path,
                "valid": rec.valid,
                "parsed": rec.parsed,
                "raw_response": rec.raw_response,
                "attempts": rec.attempts,
                "error": rec.error,
            }
            for rec in records
        ],
    )
    _write_valid_csv(valid_csv, records)

    invalid_count = sum(1 for rec in records if not rec.valid)
    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": not interrupted,
        "interrupted": interrupted,
        "docs_dir": str(docs_path),
        "total_samples": len(records),
        "valid_samples": len(records) - invalid_count,
        "invalid_samples": invalid_count,
        "model": model,
        "base_url": resolved_base_url,
        "token_env": token_env,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "sleep_seconds": sleep_seconds,
        "metric_keys": METRIC_KEYS,
        "annotator_mapping": annotator_mapping,
        "outputs": {
            "responses_jsonl": str(responses_jsonl),
            "parsed_valid_csv": str(valid_csv),
        },
    }
    run_config.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved responses to: {responses_jsonl}")
    print(f"Saved parsed valid rows to: {valid_csv}")
    print(f"Saved eval run config to: {run_config}")


if __name__ == "__main__":
    main()

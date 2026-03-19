#!/usr/bin/env python3
"""Parse structured training CLI logs and emit a compact run summary."""

from __future__ import annotations

import argparse
import ast
import json
import operator
from pathlib import Path
import re
import statistics
from typing import Any


NUM = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
HEADER_RE = re.compile(r"(?P<label>[A-Za-z][A-Za-z0-9_-]*) E(?P<epoch>\d+)\s+\|\s+(?P<fields>.+)$")
SAVE_RE = re.compile(r"Saved checkpoint:\s+(?P<path>.+)$")


def _to_float(text: str | None) -> float | None:
    if not text or text == "skipped":
        return None
    return float(text.replace(",", ""))


def _extract_named_float(line: str, name: str, *, defect_safe: bool = False) -> float | None:
    if defect_safe:
        match = re.search(rf"(?<!defect_)\b{re.escape(name)}=({NUM})", line)
    else:
        match = re.search(rf"\b{re.escape(name)}=({NUM})", line)
    return float(match.group(1)) if match else None


def _extract_list(line: str) -> list[int] | None:
    try:
        payload = line.split(":", 1)[1].split("/", 1)[0].strip()
        return list(ast.literal_eval(payload))
    except Exception:
        return None


def parse_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    command = next(
        (
            line.strip()
            for line in lines
            if (
                "$ " in line
                or line.strip().startswith("uv run ")
                or line.strip().startswith("python ")
                or line.strip().startswith("python3 ")
            )
        ),
        "",
    )
    requested_epochs = None
    command_epochs = re.search(r"--epochs\s+(\d+)", command)
    if command_epochs:
        requested_epochs = int(command_epochs.group(1))

    records: list[dict[str, Any]] = []
    save_events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    capture_code_activity = False

    for line in lines:
        match = HEADER_RE.search(line)
        if match:
            header_fields: dict[str, Any] = {}
            for part in match.group("fields").split(" | "):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                header_fields[key] = value
            current = {
                "label": match.group("label"),
                "epoch": int(match.group("epoch")),
                "step": int(header_fields["step"].replace(",", ""))
                if "step" in header_fields
                else None,
                "header_fields": header_fields,
                "train_main_header": _to_float(header_fields.get("train")),
                "eval_main_header": _to_float(header_fields.get("eval")),
            }
            records.append(current)
            section = None
            capture_code_activity = False
            continue

        if current is None:
            save_match = SAVE_RE.search(line)
            if save_match:
                save_events.append({"epoch": None, "path": save_match.group("path")})
            continue

        if line.startswith("Train metrics:"):
            section = "train"
            continue
        if line.startswith("Eval metrics:"):
            section = "eval" if "skipped" not in line else None
            capture_code_activity = section == "eval"
            continue

        if line.startswith("  obs active codes/chart:") and capture_code_activity:
            current["eval_obs_active_codes_by_chart"] = _extract_list(line)
            continue
        if line.startswith("  act active codes/chart:") and capture_code_activity:
            current["eval_act_active_codes_by_chart"] = _extract_list(line)
            continue

        if line.startswith("  enclosure:"):
            for key in [
                "acc_act",
                "acc_base",
                "acc_both",
                "acc_obs",
                "alpha",
                "ce_act",
                "ce_base",
                "ce_both",
                "ce_obs",
                "loss_encoder",
                "loss_probe",
            ]:
                value = _extract_named_float(line, key, defect_safe=True)
                if value is not None:
                    current[f"{section}_enclosure_{key}"] = value
            continue

        if line.startswith("  loss:"):
            for key in [
                "act",
                "enclosure_encoder",
                "enclosure_probe",
                "main",
                "markov_shape",
                "markov_transition",
                "obs",
                "probe",
            ]:
                value = _extract_named_float(line, key)
                if value is not None:
                    current[f"{section}_loss_{key}"] = value
            continue

        if line.startswith("  act:"):
            for key in [
                "H_K",
                "H_K_given_X",
                "I_XK",
                "active_code_charts",
                "usage_active",
                "usage_perplexity",
                "routing_confidence_mean",
                "top1_prob_mean",
                "recon",
                "vq",
                "v_tangent_barrier",
            ]:
                value = _extract_named_float(line, key)
                if value is not None:
                    current[f"{section}_act_{key}"] = value
            continue

        if line.startswith("  obs:"):
            for key in [
                "H_K",
                "H_K_given_X",
                "I_XK",
                "active_code_charts",
                "usage_active",
                "usage_perplexity",
                "routing_confidence_mean",
                "top1_prob_mean",
                "recon",
                "vq",
                "v_tangent_barrier",
            ]:
                value = _extract_named_float(line, key)
                if value is not None:
                    current[f"{section}_obs_{key}"] = value
            continue

        if line.startswith("  markov:"):
            for key in [
                "L_transition",
                "chart_acc",
                "code_acc",
                "transition_acc",
                "transition_ce",
                "target_state_entropy",
                "next_state_entropy",
            ]:
                value = _extract_named_float(line, key)
                if value is not None:
                    current[f"{section}_markov_{key}"] = value
            for src, dst in [
                ("shape/L_align", "shape_align"),
                ("shape/agreement", "shape_agreement"),
                ("shape/align_ce", "shape_align_ce"),
                ("shape/align_kl", "shape_align_kl"),
            ]:
                value = _extract_named_float(line, src)
                if value is not None:
                    current[f"{section}_markov_{dst}"] = value
            continue

        save_match = SAVE_RE.search(line)
        if save_match:
            save_events.append({"epoch": current["epoch"], "path": save_match.group("path")})

    eval_records = [record for record in records if record.get("eval_loss_main") is not None]
    best_eval = (
        min(eval_records, key=operator.itemgetter("eval_loss_main")) if eval_records else None
    )
    last_record = records[-1] if records else None

    warnings: list[dict[str, Any]] = []

    if (
        requested_epochs is not None
        and last_record
        and last_record["epoch"] + 1 < requested_epochs
    ):
        warnings.append(
            {
                "type": "run_incomplete",
                "detail": (
                    f"Requested {requested_epochs} epochs but the last logged epoch is "
                    f"{last_record['epoch']}."
                ),
            },
        )

    first_alpha_one = next(
        (
            record["epoch"]
            for record in records
            if record.get("train_enclosure_alpha") is not None
            and abs(record["train_enclosure_alpha"] - 1.0) < 1e-12
        ),
        None,
    )
    if first_alpha_one is not None:
        warnings.append(
            {
                "type": "alpha_saturated",
                "detail": f"Enclosure alpha first reaches 1.0 at epoch {first_alpha_one}.",
            },
        )

    act_collapse_epochs = [
        record["epoch"]
        for record in records
        if (
            (record.get("train_act_H_K") is not None and record["train_act_H_K"] < 0.05)
            or (
                record.get("train_act_active_code_charts") is not None
                and record["train_act_active_code_charts"] <= 1.1
            )
        )
    ]
    if act_collapse_epochs:
        warnings.append(
            {
                "type": "act_collapse",
                "detail": (
                    f"Train action routing looks collapsed at epochs {act_collapse_epochs}."
                ),
            },
        )

    obs_i_xk_values = [
        record["eval_obs_I_XK"]
        for record in eval_records
        if record.get("eval_obs_I_XK") is not None
    ]
    obs_top1_values = [
        record["eval_obs_top1_prob_mean"]
        for record in eval_records
        if record.get("eval_obs_top1_prob_mean") is not None
    ]
    if obs_i_xk_values and obs_top1_values:
        median_i_xk = statistics.median(obs_i_xk_values)
        median_top1 = statistics.median(obs_top1_values)
        if median_i_xk < 1e-3 and median_top1 < 0.08:
            warnings.append(
                {
                    "type": "obs_ungrounded",
                    "detail": (
                        "Eval observation symbols look weakly grounded: median "
                        f"I_XK={median_i_xk:.6f}, median top1_prob_mean={median_top1:.4f}."
                    ),
                },
            )

    if best_eval and last_record and best_eval["epoch"] != last_record["epoch"]:
        if (
            last_record.get("train_loss_main") is not None
            and best_eval.get("eval_loss_main") is not None
            and eval_records[-1]["eval_loss_main"] > best_eval["eval_loss_main"]
        ):
            warnings.append(
                {
                    "type": "post_best_degradation",
                    "detail": (
                        f"Best eval is at epoch {best_eval['epoch']}, but the last eval is worse."
                    ),
                },
            )

    if best_eval:
        save_epochs = [event["epoch"] for event in save_events if event["epoch"] is not None]
        if save_epochs and best_eval["epoch"] not in save_epochs:
            warnings.append(
                {
                    "type": "best_eval_not_saved",
                    "detail": (
                        f"Best eval epoch {best_eval['epoch']} is not one of the saved "
                        f"checkpoint epochs {save_epochs}."
                    ),
                },
            )

    return {
        "path": str(path),
        "command": command,
        "requested_epochs": requested_epochs,
        "completed_epochs": None if last_record is None else last_record["epoch"] + 1,
        "run_label": None if not records else records[0]["label"],
        "start_epoch": None if not records else records[0]["epoch"],
        "last_epoch": None if last_record is None else last_record["epoch"],
        "last_step": None if last_record is None else last_record["step"],
        "save_events": save_events,
        "first_alpha_one_epoch": first_alpha_one,
        "best_eval": best_eval,
        "last_eval": None if not eval_records else eval_records[-1],
        "eval_records": eval_records,
        "warnings": warnings,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Training Log Summary")
    lines.append("")
    lines.append(f"- Path: `{summary['path']}`")
    if summary.get("command"):
        lines.append(f"- Command: `{summary['command']}`")
    if summary.get("requested_epochs") is not None:
        lines.append(f"- Requested epochs: `{summary['requested_epochs']}`")
    if summary.get("completed_epochs") is not None:
        lines.append(f"- Completed epochs logged: `{summary['completed_epochs']}`")
    if summary.get("last_step") is not None:
        lines.append(f"- Last global step: `{summary['last_step']}`")
    if summary.get("first_alpha_one_epoch") is not None:
        lines.append(
            f"- First epoch with `enclosure/alpha = 1.0`: `{summary['first_alpha_one_epoch']}`"
        )

    best_eval = summary.get("best_eval")
    if best_eval:
        lines.append(
            "- Best eval:"
            f" epoch `{best_eval['epoch']}`, `loss/main={best_eval['eval_loss_main']:.4f}`"
        )

    if summary.get("save_events"):
        lines.append("- Saved checkpoints:")
        for event in summary["save_events"]:
            epoch_text = "unknown" if event["epoch"] is None else str(event["epoch"])
            lines.append(f"  - epoch `{epoch_text}`: `{event['path']}`")

    if summary.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for warning in summary["warnings"]:
            lines.append(f"- `{warning['type']}`: {warning['detail']}")

    eval_records = summary.get("eval_records") or []
    if eval_records:
        lines.append("")
        lines.append("## Eval Table")
        lines.append(
            "| epoch | eval_main | eval_act | eval_obs | eval_enclosure | eval_markov | "
            "act_H_K | obs_I_XK | obs_top1 |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for record in eval_records:
            lines.append(
                "| "
                f"{record['epoch']} | "
                f"{record.get('eval_loss_main', float('nan')):.4f} | "
                f"{record.get('eval_loss_act', float('nan')):.4f} | "
                f"{record.get('eval_loss_obs', float('nan')):.4f} | "
                f"{record.get('eval_enclosure_loss_encoder', float('nan')):.4f} | "
                f"{record.get('eval_markov_L_transition', float('nan')):.4f} | "
                f"{record.get('eval_act_H_K', float('nan')):.4f} | "
                f"{record.get('eval_obs_I_XK', float('nan')):.6f} | "
                f"{record.get('eval_obs_top1_prob_mean', float('nan')):.4f} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to the training log")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    args = parser.parse_args()

    summary = parse_log(Path(args.log_path))
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_markdown(summary))


if __name__ == "__main__":
    main()

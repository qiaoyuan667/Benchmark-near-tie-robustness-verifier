#!/usr/bin/env python3
"""Fail closed on common code-release leaks and malformed Python files."""

from __future__ import annotations

import argparse
import ast
import io
import stat
import subprocess
import zipfile
import csv
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_PATH = ROOT / "provenance/public_files.sha256"
REPORT_PATH = ROOT / "provenance/release_audit.json"
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH_PARTS = {
    "annotation_batches",
    "external_cache",
    "external_sources",
    "__pycache__",
    ".pytest_cache",
}
IGNORED_OPERATIONAL_PATH_PARTS = {".git", ".venv", "build", "dist", "outputs"}
FORBIDDEN_SUFFIXES = {".pkl", ".npy", ".npz", ".parquet", ".pt", ".safetensors"}
SECRET_PATTERNS = {
    "absolute_user_path": re.compile(r"/(?:Users|home)/[^/$\s]+/"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned_api_key": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[=:]\s*['\"][^'\"]{8,}"
    ),
}
SENSITIVE_COLUMNS = {
    "question",
    "question_text",
    "problem",
    "rationale",
    "prompt",
    "model_id",
    "owner",
    "uploader",
    "item_id",
    "blinded_id",
    "pair_id",
}
REQUIRED_FILES = {
    "LICENSE",
    "configs/blinded_content_annotation_schema.json",
    "configs/blinded_content_rubric.md",
    "configs/analysis_protocol.json",
    "docs/DATA.md",
    "docs/PAPER_ARTIFACT_MAP.md",
    "requirements-pinned.txt",
    "results/primary/benchmark_summary.csv",
    "results/primary/family_shift_summary.csv",
    "results/primary/pairwise_exact_common_family_shift_summary.csv",
    "results/gap_sensitivity/gap_sensitivity_summary.csv",
    "results/population_robustness/robustness_summary.csv",
    "results/item_stability/stability_summary.csv",
    "results/item_stability/source_validation_summary.csv",
    "results/item_stability/source_shift_driver_validation_summary.csv",
    "results/content_audit/paired_binary_enrichment.csv",
    "tools/render_paper_figures.py",
}
BENCHMARK_KEYS = {"mmlu_pro", "bbh", "mmlu", "hellaswag", "winogrande"}
EXPECTED_DIMENSIONS = {
    "mmlu_pro": {"discovery": 64, "validation": 32},
    "bbh": {"discovery": 128, "validation": 128},
    "mmlu": {"discovery": 128, "validation": 128},
    "hellaswag": {"discovery": 16, "validation": 64},
    "winogrande": {"discovery": 8, "validation": 8},
}
EXPECTED_CANDIDATES = {
    "mmlu_pro": [1, 2, 4, 8, 16, 32, 64, 128],
    "bbh": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "mmlu": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "hellaswag": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "winogrande": [1, 2, 4, 8, 16, 32, 64, 128, 256],
}
EXPECTED_HEADLINES = {
    "MMLU-Pro": (0.4707792208, 0.1850649351, 0.2857142857, 0.0009990010),
    "BBH": (0.4213836478, 0.2524707996, 0.1689128482, 0.0009990010),
    "MMLU": (0.4068006182, 0.1632148377, 0.2435857805, 0.0009990010),
    "HellaSwag": (0.3092874476, 0.1140902585, 0.1951971891, 0.0009990010),
    "WinoGrande": (0.3381772515, 0.3469259985, -0.0087487470, 0.6893106893),
}
SEED_KEYS = {
    "family_split",
    "nested_owner_validation",
    "svd",
    "composition_bootstrap_base",
    "matched_random_base",
    "owner_bootstrap",
}



FORBIDDEN_PATH_PARTS.update({"__MACOSX", ".DS_Store", ".mypy_cache", ".ruff_cache", "raw_data", "raw_datasets", "cache"})
FORBIDDEN_SUFFIXES.update({".pickle", ".pth", ".pyc", ".pyo", ".log"})
SENSITIVE_COLUMNS.update({"model_name", "owner_id", "original_item_id", "dataset_item_index", "dataset_model_index", "affiliation", "institution", "email"})
ALLOWED_HIDDEN = {".gitignore", ".github", ".env.example"}
MANIFEST = "provenance/public_files.sha256"
REPORT = "provenance/release_audit.json"
EXCLUDED = {MANIFEST, REPORT}
LABEL_PATH = "results/content_audit/labels.csv"
LABEL_COLUMNS = {"annotator", "model", "blinded_id", "benchmark", "pair_id", "group",
                 "common_blueprint_cell", "discovery_advantaged_family", "validation_advantaged_family",
                 "quantitative_symbolic", "formal_rule_reasoning", "factual_domain_knowledge",
                 "contextual_reading", "commonsense_narrative", "spatial_temporal", "linguistic_wordplay",
                 "negation_exception", "code_structured_representation", "distractor_discrimination",
                 "reasoning_steps", "context_burden", "ambiguity_degree"}
SECRET_PATTERNS.update({
    "absolute_machine_path": re.compile(r"(?<![:\w])/(?:private|tmp|mnt|workspace|root|scratch|gpfs|lustre|cluster|data)/[^\s\"'<>]+"),
    "windows_user_path": re.compile(r"\b[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s\"']+"),
    "hosting_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{20,})\b"),
})
REQUIRED_FILES.update({
    "README.md", "pyproject.toml", "configs/upstream_inputs.json", "configs/routereval_item_layout.json",
    "docs/REPRODUCE.md", "docs/RESULTS.md", "docs/CONTENT_AUDIT.md", "docs/DIAGNOSTICS.md",
    "docs/MAINTAINER_GUIDE.md", "docs/RELEASE_CHECKLIST.md", "examples/make_synthetic_benchmark.py",
    "tools/audit_release.py", "tools/prepare_inputs.py", "tools/build_routereval_row_dataset_mapping.py",
    "tools/build_routereval_all_benchmarks_mapping.py", "tools/build_routereval_raw_global_matrix.py",
    "tools/build_aligned_all_benchmarks_dataset.py", "tools/export_routereval_raw_matrices.py",
    "results/gap_sensitivity/observed_gap_curves_by_fold.csv",
    "results/gap_sensitivity/matched_random_gap_curves_by_fold.csv",
    "results/direct_mirt/summary.csv", "results/direct_mirt/paired_cv_comparison.csv", LABEL_PATH,
    "results/discrimination/summary.csv", "results/discrimination/low_dif_fold_diagnostics.csv",
    "results/discrimination/original_matched_random_controls.csv",
    "results/discrimination/discrimination_matched_random_controls.csv",
    "results/within_family/summary.csv", "results/within_family/observed_fold_metrics.csv",
    "results/within_family/matched_random_controls.csv", "results/capability_profile/summary.csv",
    "results/capability_profile/protocol.json", "tests/test_artifact_integrity.py",
    "tests/test_release_diagnostics.py", "tests/test_content_replay.py", "tests/test_maintainer.py", "tests/test_direct_mirt.py",
})
for module in ("core/family_dif", "core/spectral_mirt", "core/direct_mirt", "ranking/mirt_primary",
               "ranking/direct_mirt", "ranking/exact_common", "ranking/summarize_v3",
               "robustness/family_owner_v3", "robustness/gap_sensitivity_v3",
               "interpretation/item_stability_v3", "interpretation/blinded_content_v3",
               "interpretation/content_replay", "diagnostics/discrimination",
               "diagnostics/within_family", "diagnostics/capability_profile", "maintainer"):
    REQUIRED_FILES.add("src/family_dif_benchmark_audit/" + module + ".py")
for benchmark in BENCHMARK_KEYS:
    for family in ("primary", "direct_mirt"):
        for filename in ("protocol.json", "dimension_cv.csv", "dimension_selection.csv", "ranking_impact_metrics.csv", "matched_random_anchor_controls.csv"):
            REQUIRED_FILES.add(f"results/{family}/per_benchmark/{benchmark}/{filename}")
    for filename in ("controls.csv", "observed.csv", "all_swapped.csv", "matching_quality.csv"):
        REQUIRED_FILES.add(f"results/capability_profile/per_benchmark/{benchmark}/{filename}")


def issue(path, kind, **detail):
    return {"path": path, "issue": kind, **detail}


def path_issues(name, archive=False):
    p = PurePosixPath(name)
    result = []
    if p.is_absolute() or ".." in p.parts or "\\" in name or re.match(r"^[A-Za-z]:", name):
        result.append(issue(name, "unsafe_path"))
    if FORBIDDEN_PATH_PARTS.intersection(p.parts):
        result.append(issue(name, "forbidden_path"))
    if archive and (IGNORED_OPERATIONAL_PATH_PARTS.intersection(p.parts) or any(x.endswith(".egg-info") or x.startswith(".venv-") for x in p.parts)):
        result.append(issue(name, "operational_metadata_in_archive"))
    if any(x.startswith(".") and x not in ALLOWED_HIDDEN for x in p.parts):
        result.append(issue(name, "unapproved_hidden_path"))
    if p.suffix.lower() in FORBIDDEN_SUFFIXES:
        result.append(issue(name, "forbidden_binary_or_log"))
    return result


def check_label_pseudonyms(rows):
    if len(rows) != 1000:
        return [issue(LABEL_PATH, "unexpected_label_panel_size")]
    for column, prefix, count in (("blinded_id", "record_", 500), ("pair_id", "pair_", 250)):
        if {r.get(column, "") for r in rows} != {f"{prefix}{i:04d}" for i in range(count)}:
            return [issue(LABEL_PATH, "nonrelease_label_identifiers")]
    cells = {r.get("common_blueprint_cell", "") for r in rows}
    if cells != {f"cell_{i:04d}" for i in range(len(cells))}:
        return [issue(LABEL_PATH, "nonrelease_blueprint_identifiers")]
    if len({(r.get("annotator"), r["blinded_id"]) for r in rows}) != 1000 or {r.get("annotator") for r in rows} != {"annotator_a", "annotator_b"}:
        return [issue(LABEL_PATH, "invalid_label_panel")]
    pairs = {}
    for row in rows:
        pairs.setdefault((row["annotator"], row["pair_id"]), []).append(row)
    if any(len(p) != 2 or {r.get("group") for r in p} != {"stable_high_dif", "matched_control"} for p in pairs.values()):
        return [issue(LABEL_PATH, "invalid_label_pairs")]
    return []


def content_issues(name, data, deny_terms=()):
    result, text = [], data.decode("utf-8", errors="replace")
    for index, term in enumerate(deny_terms, 1):
        if term.casefold() in text.casefold() or term.casefold() in name.casefold():
            result.append(issue(name, "private_deny_term", term_number=index))
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            result.append(issue(name, label))
    if name.endswith(".py"):
        try:
            ast.parse(text, filename=name)
        except SyntaxError as error:
            result.append(issue(name, "python_syntax_error", line=error.lineno, column=error.offset))
    if name.endswith(".json"):
        try:
            json.loads(text)
        except (ValueError, TypeError):
            result.append(issue(name, "invalid_json"))
    if name.endswith(".csv"):
        rows = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
        allowed = {"blinded_id", "pair_id"} if name == LABEL_PATH else set()
        columns = {c.strip().casefold().replace(" ", "_") for c in (rows.fieldnames or [])}
        leaked = (columns & SENSITIVE_COLUMNS) - allowed
        if leaked:
            result.append(issue(name, "sensitive_csv_columns", columns=sorted(leaked)))
        if name == LABEL_PATH:
            if set(rows.fieldnames or []) != LABEL_COLUMNS:
                result.append(issue(name, "unexpected_public_label_schema"))
            result.extend(check_label_pseudonyms(list(rows)))
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            pass
        else:
            try:
                metadata = PdfReader(io.BytesIO(data)).metadata or {}
                if str(metadata.get("/Author", "")).strip() not in ("", "Anonymous"):
                    result.append(issue(name, "nonanonymous_pdf_author"))
                serial = json.dumps({str(k): str(v) for k, v in metadata.items()})
                for index, term in enumerate(deny_terms, 1):
                    if term.casefold() in serial.casefold():
                        result.append(issue(name, "private_pdf_metadata", term_number=index))
                if any(p.search(serial) for p in SECRET_PATTERNS.values()):
                    result.append(issue(name, "sensitive_pdf_metadata"))
            except Exception:
                result.append(issue(name, "unreadable_pdf_metadata"))
    return result


def check_git_identity(root, check_local_identity=False):
    if not (root / ".git").exists():
        return []
    issues, command = [], ["git", "-C", str(root)]
    emails = {"anonymous@example.org", "anonymous@example.com", "anonymous@example.invalid", "anonymous@anonymous.invalid", "anonymous@users.noreply.github.com"}
    log = subprocess.run(command + ["log", "--all", "--format=%an%x00%ae%x00%cn%x00%ce"], capture_output=True, text=True)
    if log.returncode != 0:
        issues.append(issue(".git", "git_history_unreadable"))
    if log.returncode == 0:
        for line in log.stdout.splitlines():
            v = line.split("\0")
            if len(v) != 4 or v[0] != "Anonymous" or v[2] != "Anonymous" or any(v[i].lower() not in emails for i in (1, 3)):
                issues.append(issue(".git", "nonanonymous_commit_identity"))
                break
    if check_local_identity:
        for key, allowed in (("user.name", {"Anonymous"}), ("user.email", emails)):
            value = subprocess.run(command + ["config", "--local", "--get", key], capture_output=True, text=True)
            if value.stdout.strip() not in allowed:
                issues.append(issue(".git", "nonanonymous_or_missing_local_git_identity", field=key))
    return issues


def check_contract(root, names):
    issues = [issue(n, "missing_promised_file") for n in sorted(REQUIRED_FILES - names)]
    for benchmark in BENCHMARK_KEYS:
        path = root / f"results/primary/per_benchmark/{benchmark}/protocol.json"
        if not path.is_file():
            continue
        try:
            p = json.loads(path.read_text())
            for field, expected in (("selected_dimensions", EXPECTED_DIMENSIONS[benchmark]), ("candidate_dimensions", EXPECTED_CANDIDATES[benchmark])):
                if p.get(field) != expected:
                    issues.append(issue(str(path.relative_to(root)), "unexpected_" + field))
            if set(p.get("seeds", {})) != SEED_KEYS:
                issues.append(issue(str(path.relative_to(root)), "missing_seed_record"))
            if not p.get("inputs") or any(not str(r.get("path", "")).startswith("$" + "{PROJECT_ROOT}/") or not re.fullmatch(r"[0-9a-f]{64}", str(r.get("sha256", ""))) for r in p["inputs"]):
                issues.append(issue(str(path.relative_to(root)), "invalid_input_hash_record"))
        except (ValueError, TypeError, AttributeError):
            issues.append(issue(str(path.relative_to(root)), "invalid_primary_protocol"))
    headline = root / "results/primary/benchmark_summary.csv"
    if headline.is_file():
        rows = {r["benchmark"]: r for r in csv.DictReader(io.StringIO(headline.read_text()))}
        for benchmark, expected in EXPECTED_HEADLINES.items():
            row = rows.get(benchmark)
            columns = ("close_cross_family_flip_rate", "random_control_close_median", "excess_close_flip_rate", "random_control_close_p")
            if row is None or any(abs(float(row[c]) - v) > 5e-10 for c, v in zip(columns, expected)):
                issues.append(issue(str(headline.relative_to(root)), "headline_result_mismatch", benchmark=benchmark))
    return issues


def public_paths(root):
    """Walk the public tree without traversing local environments or outputs."""
    def operational(name):
        return name in IGNORED_OPERATIONAL_PATH_PARTS or name.endswith(".egg-info") or name.startswith(".venv-")
    for directory, subdirs, filenames in os.walk(root, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if not operational(name))
        for name in subdirs + sorted(filenames):
            if not operational(name):
                yield Path(directory) / name


def collect_tree(root, deny_terms=(), contract=True, check_local_identity=False):
    issues, hashes, python_count = [], {}, 0
    for path in sorted(public_paths(root)):
        rel = path.relative_to(root)
        if IGNORED_OPERATIONAL_PATH_PARTS.intersection(rel.parts) or any(p.endswith(".egg-info") or p.startswith(".venv-") for p in rel.parts):
            continue
        name = rel.as_posix()
        if path.is_symlink():
            issues.append(issue(name, "symlink_in_public_tree"))
            continue
        issues.extend(path_issues(name))
        if not path.is_file():
            continue
        data = path.read_bytes()
        issues.extend(content_issues(name, data, deny_terms))
        if name not in EXCLUDED:
            hashes[name] = hashlib.sha256(data).hexdigest()
        python_count += path.suffix == ".py"
    if contract:
        issues.extend(check_contract(root, set(hashes)))
    issues.extend(check_git_identity(root, check_local_identity=check_local_identity))
    return issues, hashes, python_count


def parse_manifest(text):
    result = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not m:
            raise ValueError("malformed checksum manifest")
        value, name = m.groups()
        if name in result or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or name in EXCLUDED:
            raise ValueError("unsafe or duplicate manifest entry")
        result[name] = value
    return result


def verify_manifest(path, actual):
    if not path.is_file():
        return [issue(MANIFEST, "missing_manifest_explicit_refresh_required")]
    try:
        expected = parse_manifest(path.read_text())
    except ValueError:
        return [issue(MANIFEST, "invalid_manifest")]
    return [issue(n, "checksum_or_inventory_mismatch") for n in sorted(set(expected) | set(actual)) if expected.get(n) != actual.get(n)]


def audit_zip(path, expected_hashes, deny_terms=()):
    issues, observed = [], {}
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [i for i in archive.infolist() if not i.is_dir()]
            prefixes = {PurePosixPath(i.filename).parts[0] for i in entries if PurePosixPath(i.filename).parts}
            wrapped = len(prefixes) == 1 and all(len(PurePosixPath(i.filename).parts) > 1 for i in entries)
            for info in archive.infolist():
                issues.extend(path_issues(info.filename, archive=True))
            for info in entries:
                name = PurePosixPath(*PurePosixPath(info.filename).parts[1:]).as_posix() if wrapped else info.filename
                if stat.S_ISLNK(info.external_attr >> 16):
                    issues.append(issue(name, "symlink_in_archive"))
                if name in observed:
                    issues.append(issue(name, "duplicate_archive_entry"))
                data = archive.read(info)
                issues.extend(content_issues(name, data, deny_terms))
                observed[name] = hashlib.sha256(data).hexdigest()
            actual = {k: v for k, v in observed.items() if k not in EXCLUDED}
            for name in sorted(set(expected_hashes) | set(actual)):
                if expected_hashes.get(name) != actual.get(name):
                    issues.append(issue(name, "archive_differs_from_audited_tree"))
            if MANIFEST not in observed:
                issues.append(issue(MANIFEST, "manifest_missing_from_archive"))
            else:
                n = next(i.filename for i in entries if i.filename.endswith(MANIFEST))
                if parse_manifest(archive.read(n).decode()) != expected_hashes:
                    issues.append(issue(MANIFEST, "archive_manifest_mismatch"))
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        issues.append(issue("archive", "invalid_or_unreadable_archive"))
    return issues


def redact(value, deny_terms):
    """Never repeat private terms, including terms embedded in filenames."""
    if isinstance(value, str):
        for term in deny_terms:
            value = re.sub(re.escape(term), "[redacted]", value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [redact(item, deny_terms) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, deny_terms) for key, item in value.items()}
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Read-only verification (default).")
    mode.add_argument("--refresh", action="store_true", help="Explicitly replace checksums after a clean content audit.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publisher-check", action="store_true",
                        help="Also require anonymous local Git configuration; not needed by readers.")
    parser.add_argument("--zip", dest="archive", type=Path, help="Check a ZIP against the audited public tree.")
    parser.add_argument("--deny-term", action="append", default=[], metavar="PRIVATE_TERM", help="Private case-insensitive exclusion; repeat as needed. Matches are never echoed.")
    args = parser.parse_args(argv)
    if any(not term.strip() for term in args.deny_term):
        parser.error("private exclusions must not be empty")
    root = args.root.resolve()
    publisher_check = args.publisher_check or args.refresh
    issues, hashes, count = collect_tree(root, args.deny_term, check_local_identity=publisher_check)
    if not args.refresh:
        issues.extend(verify_manifest(root / MANIFEST, hashes))
    if args.archive:
        issues.extend(audit_zip(args.archive, hashes, args.deny_term))
    try:
        import pypdf
        pdf_check = "enabled"
    except ImportError:
        pdf_check = "unavailable; install pypdf to inspect PDF properties"
    report = redact({"status": "fail" if issues else "pass", "mode": "refresh" if args.refresh else "verify",
                     "files": len(hashes), "python_files_parsed": count, "private_exclusions_supplied": len(args.deny_term),
                     "pdf_metadata_check": pdf_check, "git_identity_checked": (root / ".git").exists(),
                     "local_git_identity_checked": publisher_check and (root / ".git").exists(),
                     "archive_checked": args.archive is not None, "excluded_from_own_checksum": sorted(EXCLUDED), "issues": issues}, args.deny_term)
    if args.refresh and not issues:
        (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        (root / MANIFEST).write_text("".join(f"{hashes[n]}  {n}\n" for n in sorted(hashes)), encoding="utf-8")
        (root / REPORT).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

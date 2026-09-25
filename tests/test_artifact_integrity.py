"""Read-only artifact checks and independent reconstruction from published counts."""
import hashlib
import importlib.util
import json
import io
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_auditor", ROOT / "tools/audit_release.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
KEYS = ("mmlu_pro", "bbh", "mmlu", "hellaswag", "winogrande")
LABELS = ("MMLU-Pro", "BBH", "MMLU", "HellaSwag", "WinoGrande")


def table(name):
    return pd.read_csv(ROOT / "results" / name)


def rate(frame, prefix="close_cross_family"):
    return frame[prefix + "_flips"].sum() / frame[prefix + "_eligible_pairs"].sum()


def null_rates(frame, prefix="close_cross_family"):
    assert not frame.duplicated(["audit_half", "replicate"]).any()
    assert set(frame.audit_half) == {"discovery", "validation"}
    for _, fold in frame.groupby("audit_half"):
        assert set(fold.replicate) == set(range(1000))
    sums = frame.groupby("replicate")[[prefix + "_flips", prefix + "_eligible_pairs"]].sum()
    assert (sums[prefix + "_eligible_pairs"] > 0).all()
    assert (sums[prefix + "_flips"] <= sums[prefix + "_eligible_pairs"]).all()
    return sums[prefix + "_flips"] / sums[prefix + "_eligible_pairs"]


def tail(null, observed):
    return (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)


def test_primary_and_direct_summaries_recomputed_from_all_integer_replicates():
    main = table("primary/benchmark_summary.csv").set_index("benchmark")
    direct = table("direct_mirt/summary.csv").set_index("benchmark")
    for family in ("primary", "direct_mirt"):
        for key, label in zip(KEYS, LABELS):
            base = f"{family}/per_benchmark/{key}/"
            observed = table(base + "ranking_impact_metrics.csv")
            observed = observed[np.isclose(observed.anchor_fraction, .5)]
            assert len(observed) == 2
            control = table(base + "matched_random_anchor_controls.csv")
            actual, null = rate(observed), null_rates(control)
            if family == "primary":
                row = main.loc[label]
                expected = [row.close_cross_family_flip_rate, row.random_control_close_median,
                            row.excess_close_flip_rate, row.random_control_close_p]
            else:
                row = direct.loc[label]
                expected = [row.reversal_percent / 100, row.random_percent / 100, row.excess_pp / 100, row.p]
            np.testing.assert_allclose([actual, null.median(), actual - null.median(), tail(null, actual)],
                                       expected, atol=1e-12, rtol=0)


def test_discrimination_summaries_recomputed_from_counts():
    summaries = table("discrimination/summary.csv").set_index("benchmark")
    observed = table("discrimination/low_dif_fold_diagnostics.csv")
    original = table("discrimination/original_matched_random_controls.csv")
    controls = table("discrimination/discrimination_matched_random_controls.csv")
    for label in LABELS:
        actual = rate(observed[observed.benchmark == label])
        null = null_rates(controls[controls.benchmark == label])
        baseline = null_rates(original[original.benchmark == label])
        row = summaries.loc[label]
        np.testing.assert_allclose([actual, baseline.median(), null.median(), 100 * (actual - null.median()), tail(null, actual)],
                                   [row.low_dif_reversal_rate, row.original_random_median,
                                    row.discrimination_matched_random_median,
                                    row.discrimination_matched_excess_points, row.discrimination_matched_p], atol=1e-12, rtol=0)


def test_within_family_specificity_uses_paired_control_counts():
    summaries = table("within_family/summary.csv").set_index("benchmark")
    observed, controls = table("within_family/observed_fold_metrics.csv"), table("within_family/matched_random_controls.csv")
    for label in LABELS:
        o, c = observed[observed.benchmark == label], controls[controls.benchmark == label]
        cross, same = rate(o), rate(o, "close_same_family")
        cn, sn = null_rates(c), null_rates(c, "close_same_family")
        paired = cn - sn
        row = summaries.loc[label]
        np.testing.assert_allclose([100 * (cross - cn.median()), 100 * (same - sn.median()),
                                    100 * (cross - same - paired.median()), tail(paired, cross - same)],
                                   [row.cross_family_excess_points, row.same_family_excess_points,
                                    row.specificity_excess_points, row.specificity_p], atol=1e-12, rtol=0)


def test_capability_profile_all_scenarios_and_paired_contrasts_recomputed():
    summaries = table("capability_profile/summary.csv")
    scenarios = {"random_pairing", "optimal_unrestricted", "caliper_1.00", "caliper_0.50", "caliper_0.25"}
    assert len(summaries) == 25
    for key, label in zip(KEYS, LABELS):
        base = f"capability_profile/per_benchmark/{key}/"
        o, c, swapped = table(base + "observed.csv"), table(base + "controls.csv"), table(base + "all_swapped.csv")
        quality = table(base + "matching_quality.csv")
        assert set(c.scenario) == scenarios
        low, same = rate(o), rate(o, "close_same_family")
        for scenario in scenarios:
            controls = c[c.scenario == scenario]
            null = null_rates(controls)
            within = null_rates(controls, "close_same_family")
            complementary = null_rates(controls, "complement_close_cross_family")
            high = rate(swapped[swapped.scenario == scenario])
            row = summaries[(summaries.benchmark == label) & (summaries.scenario == scenario)].iloc[0]
            np.testing.assert_allclose([low, null.median(), 100 * (low - null.median()), tail(null, low), high,
                                       tail(np.abs(null - complementary), abs(low - high)),
                                       100 * (low - same - (null - within).median()), tail(null - within, low - same)],
                                      [row.low_dif_rate, row.paired_random_median, row.excess_pp, row.empirical_upper_p,
                                       row.all_swapped_rate, row.paired_contrast_two_sided_p,
                                       row.specificity_excess_pp, row.specificity_upper_p], atol=1e-12, rtol=0)
            q = quality[quality.scenario == scenario]
            assert len(q) == 2
            assert row.swappable_weight_fraction == pytest.approx(q.swappable_weight_fraction.mean())
            assert row.n_pairs_total == q.n_pairs.sum()


def test_gap_summary_recomputed_from_each_threshold_control_distribution():
    observed = table("gap_sensitivity/observed_gap_curves_by_fold.csv")
    controls = table("gap_sensitivity/matched_random_gap_curves_by_fold.csv")
    summary = table("gap_sensitivity/gap_sensitivity_summary.csv")
    for row in summary.itertuples():
        mask = (observed.benchmark == row.benchmark) & np.isclose(observed.gap_points, row.gap_points)
        o = observed[mask]
        c = controls[(controls.benchmark == row.benchmark) & np.isclose(controls.gap_points, row.gap_points)]
        assert not c.duplicated(["audit_half", "replicate"]).any()
        assert all(set(g.replicate) == set(range(1000)) for _, g in c.groupby("audit_half"))
        pooled = c.groupby("replicate")[["flips", "eligible_pairs"]].sum()
        null = pooled.flips / pooled.eligible_pairs
        actual = o.flips.sum() / o.eligible_pairs.sum()
        np.testing.assert_allclose([actual, null.median(), actual - null.median(), tail(null, actual)],
                                  [row.observed_flip_rate, row.random_median, row.excess_flip_rate,
                                   row.one_sided_randomization_p], atol=1e-12, rtol=0)


def test_default_audit_detects_changes_without_rewriting_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit, "REQUIRED_FILES", set())
    readme = tmp_path / "README.md"
    readme.write_text("Anonymous artifact\n")
    assert audit.main(["--root", str(tmp_path), "--refresh"]) == 0
    manifest = tmp_path / audit.MANIFEST
    original = manifest.read_bytes()
    report = (tmp_path / audit.REPORT).read_bytes()
    readme.write_text("Changed artifact\n")
    assert audit.main(["--root", str(tmp_path)]) == 1
    assert manifest.read_bytes() == original
    assert (tmp_path / audit.REPORT).read_bytes() == report
    assert "checksum_or_inventory_mismatch" in capsys.readouterr().out


def test_missing_manifest_does_not_trigger_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REQUIRED_FILES", set())
    assert audit.main(["--root", str(tmp_path)]) == 1
    assert not (tmp_path / audit.MANIFEST).exists()


def test_private_deny_terms_are_not_echoed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit, "REQUIRED_FILES", set())
    private = "synthetic_identity_marker"
    (tmp_path / (private + ".md")).write_text(private)
    assert audit.main(["--root", str(tmp_path), "--refresh", "--deny-term", private]) == 1
    captured = capsys.readouterr().out
    assert private not in captured
    assert "private_deny_term" in captured
    assert not (tmp_path / audit.MANIFEST).exists()


def test_archive_hidden_allowlist_and_forbidden_git(tmp_path):
    contents = {"README.md": b"Anonymous", ".gitignore": b"outputs/\n",
                ".github/workflows/tests.yml": b"name: tests\n", ".env.example": b"API_KEY=\n"}
    expected = {key: hashlib.sha256(value).hexdigest() for key, value in contents.items()}
    manifest = "".join(f"{value}  {name}\n" for name, value in sorted(expected.items()))
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, data in contents.items():
            z.writestr("artifact/" + name, data)
        z.writestr("artifact/" + audit.MANIFEST, manifest)
    assert audit.audit_zip(archive, expected) == []
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("artifact/.git/config", "local checkout metadata")
    assert any(i["issue"] == "operational_metadata_in_archive" for i in audit.audit_zip(archive, expected))


def test_sensitive_columns_only_allow_fresh_content_identifiers():
    labels = (ROOT / audit.LABEL_PATH).read_bytes()
    assert audit.content_issues(audit.LABEL_PATH, labels) == []
    assert any(i["issue"] == "sensitive_csv_columns" for i in audit.content_issues("results/other.csv", labels))
    invalid = labels.replace(b"record_0000", b"original_identifier")
    assert any(i["issue"] == "nonrelease_label_identifiers" for i in audit.content_issues(audit.LABEL_PATH, invalid))


def test_syntax_raw_binary_paths_and_tokens_are_rejected():
    assert any(i["issue"] == "python_syntax_error" for i in audit.content_issues("bad.py", b"def broken(\n"))
    assert audit.path_issues("raw_data/responses.npy")
    assert audit.path_issues("../outside.md", archive=True)
    token = ("sk-" + "a" * 24).encode()
    assert any(i["issue"] == "openai_style_key" for i in audit.content_issues("notes.txt", token))


def test_all_promised_files_present():
    names = {p.relative_to(ROOT).as_posix() for p in audit.public_paths(ROOT) if p.is_file()}
    assert not audit.REQUIRED_FILES - names


def test_local_build_artifacts_are_ignored_but_archive_rejects_them(tmp_path):
    for directory in ("build", "dist", "outputs", ".venv", ".venv-check", "package.egg-info"):
        location = tmp_path / directory
        location.mkdir()
        (location / "local.txt").write_text("not part of the public artifact")
        assert audit.path_issues(directory + "/local.txt", archive=True)
    issues, hashes, _ = audit.collect_tree(tmp_path, contract=False)
    assert not issues and not hashes
    leaked = tmp_path / "unexpected" / "responses.npy"
    leaked.parent.mkdir()
    leaked.write_bytes(b"not public")
    issues, _, _ = audit.collect_tree(tmp_path, contract=False)
    assert any(i["issue"] == "forbidden_binary_or_log" for i in issues)


@pytest.mark.skipif(shutil.which('git') is None, reason='Optional Git identity fixture requires Git')
def test_git_requires_anonymous_author_and_committer(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("config", "user.name", "Anonymous")
    git("config", "user.email", "anonymous@example.org")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Anonymous test fixture")
    assert audit.check_git_identity(tmp_path) == []
    git("config", "user.name", "Synthetic Test Identity")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Identity validation fixture")
    findings = audit.check_git_identity(tmp_path)
    assert any(i["issue"] == "nonanonymous_commit_identity" for i in findings)
    assert "Synthetic Test Identity" not in json.dumps(findings)


@pytest.mark.skipif(shutil.which('git') is None, reason='Optional Git identity fixture requires Git')
def test_reader_clone_does_not_require_publisher_git_configuration(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    command = ["git", "-C", str(tmp_path)]
    subprocess.run(command + ["-c", "user.name=Anonymous", "-c",
                   "user.email=anonymous@example.invalid", "-c", "commit.gpgsign=false",
                   "commit", "--allow-empty", "-m", "Anonymous fixture"],
                   check=True, capture_output=True)
    assert audit.check_git_identity(tmp_path) == []
    assert audit.check_git_identity(tmp_path, check_local_identity=True)
    subprocess.run(command + ["config", "user.name", "Reader Identity"], check=True)
    assert audit.check_git_identity(tmp_path) == []


def test_pdf_author_metadata_checked_when_dependency_is_available():
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Author": "Synthetic Test Identity"})
    buffer = io.BytesIO()
    writer.write(buffer)
    findings = audit.content_issues("figure.pdf", buffer.getvalue())
    assert any(i["issue"] == "nonanonymous_pdf_author" for i in findings)

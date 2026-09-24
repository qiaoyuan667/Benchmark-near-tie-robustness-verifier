"""Content replay and provider-neutral annotation boundary tests."""
import json
import shlex
import sys
from pathlib import Path

import pandas as pd
import pytest

from family_dif_benchmark_audit.interpretation import blinded_content_v3 as audit
from family_dif_benchmark_audit.interpretation.content_replay import LABEL_COLUMNS, validate_labels


def published_labels():
    root = Path(__file__).resolve().parents[1]
    return pd.read_csv(root / "results/content_audit/labels.csv")


def test_public_labels_have_exact_schema_and_paper_coverage():
    labels = published_labels()
    validate_labels(labels)
    assert list(labels.columns) == LABEL_COLUMNS
    assert labels.blinded_id.nunique() == 500
    assert labels.pair_id.nunique() == 250
    assert labels.blinded_id.str.match(r"record_\d{4}$").all()
    assert labels.pair_id.str.match(r"pair_\d{4}$").all()
    assert labels.common_blueprint_cell.str.match(r"cell_\d{4}$").all()


def test_public_labels_reject_leaked_text_and_broken_pairs():
    labels = published_labels()
    with pytest.raises(ValueError, match="public columns"):
        validate_labels(labels.assign(rationale="unreleased free text"))
    with pytest.raises(ValueError, match="matched pair"):
        validate_labels(labels.iloc[1:])


def test_label_validation_does_not_truncate_fractional_values():
    labels = published_labels()
    labels[audit.BINARY_AXES[0]] = labels[audit.BINARY_AXES[0]].astype(float)
    labels.loc[0, audit.BINARY_AXES[0]] = 0.5
    with pytest.raises(ValueError, match="binary label"):
        validate_labels(labels)
    row = {"blinded_id": "example", **{axis: 0 for axis in (*audit.BINARY_AXES, *audit.ORDINAL_AXES)}}
    row[audit.BINARY_AXES[0]] = 0.5
    with pytest.raises(ValueError, match="binary annotation"):
        audit.validate_payload({"annotations": [row]}, ["example"])


def test_published_labels_reproduce_reliability():
    labels = published_labels()
    binary, ordinal = audit.reliability_results(labels)
    assert binary.cohen_kappa.dropna().median() == pytest.approx(0.7154544707370809)
    assert ordinal.spearman_r.median() == pytest.approx(0.8000840608368447)


def test_backend_receives_json_and_returns_validated_annotations(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(audit, "OUTPUT", tmp_path)
    monkeypatch.setattr(audit, "SCHEMA", root / "configs/blinded_content_annotation_schema.json")
    row = {"blinded_id": "example", **{axis: 0 for axis in (*audit.BINARY_AXES, *audit.ORDINAL_AXES)},
           "primary_reasoning_type": "other", "confidence": 1.0, "rationale": "Test fixture."}
    payload = json.dumps({"annotations": [row]})
    # A local subprocess only; it verifies the transport request, not a model.
    program = ("import json,sys; r=json.load(sys.stdin); "
               "assert r['model']=='gpt-5.5'; assert r['reasoning_effort']=='low'; "
               "assert 'Binary axes' in r['prompt']; assert r['response_schema']['type']=='object'; "
               "print(" + repr(payload) + ")")
    command = shlex.join([sys.executable, "-c", program])
    result = audit.run_annotation_batch("annotator_a", "gpt-5.5", 0,
                                        [{"blinded_id": "example", "question_text": "Local test"}], command)
    assert json.loads(result.read_text()) == json.loads(payload)


def test_aligned_question_loader_checks_coverage_and_duplicates(tmp_path, monkeypatch):
    questions = tmp_path / "questions.csv"
    signatures = pd.DataFrame({"benchmark": ["BBH", "MMLU"], "dataset_item_index": [0, 0]})
    pd.DataFrame({"benchmark": ["BBH", "MMLU"], "dataset_item_index": [0, 0],
                  "question_text": ["First item", "Second item"]}).to_csv(questions, index=False)
    monkeypatch.setattr(audit, "QUESTION_FILE", questions)
    assert audit.load_question_texts(signatures) == {("BBH", 0): "First item", ("MMLU", 0): "Second item"}
    with pytest.raises(ValueError, match="missing 1 signature"):
        audit.load_question_texts(pd.DataFrame({"benchmark": ["BBH"], "dataset_item_index": [1]}))
    pd.DataFrame({"benchmark": ["BBH", "BBH"], "dataset_item_index": [0, 0],
                  "question_text": ["First item", "Duplicate item"]}).to_csv(questions, index=False)
    with pytest.raises(ValueError, match="duplicate item"):
        audit.load_question_texts(signatures)

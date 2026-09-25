"""The documented relative backend command runs from the caller's directory."""
import json
import shlex
import sys

from family_dif_benchmark_audit.interpretation import blinded_content_v3 as audit


def test_relative_backend_with_external_data_root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(audit, "HERE", data)
    monkeypatch.setattr(audit, "OUTPUT", data / "annotations")
    monkeypatch.chdir(tmp_path)
    values = {axis: 0 for axis in audit.BINARY_AXES + audit.ORDINAL_AXES}
    values.update(blinded_id="item_0000", rationale="Synthetic transport fixture")
    payload = {"annotations": [values]}
    (tmp_path / "backend.py").write_text(
        "import json, sys\nrequest = json.load(sys.stdin)\n"
        "assert request['response_schema']['type'] == 'object'\n"
        f"print({json.dumps(payload)!r})\n"
    )
    result = audit.run_annotation_batch(
        "annotator_a", "synthetic", 0,
        [{"blinded_id": "item_0000", "question_text": "Synthetic question"}],
        shlex.quote(sys.executable) + " backend.py",
    )
    assert json.loads(result.read_text()) == payload
    assert audit.SCHEMA.is_file()

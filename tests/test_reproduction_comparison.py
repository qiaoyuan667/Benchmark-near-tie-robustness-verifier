"""A comparison must not silently pass missing, reordered, or changed results."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT / 'tools/compare_reproduction.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.mark.parametrize('group', ['primary', 'direct_mirt', 'capability_profile'])
def test_per_benchmark_mapping(group):
    path = tool.generated_path(Path(group) / 'per_benchmark/bbh/table.csv', Path('outputs'))
    prefix = {'primary': 'outputs', 'direct_mirt': 'outputs/direct_mirt',
              'capability_profile': 'outputs/capability_profile'}[group]
    folder = 'bbh' if group == 'capability_profile' else 'bbh_family_ranking_impact'
    assert path == Path(prefix) / folder / 'table.csv'


def test_missing_changed_reordered_and_extra_fields(tmp_path):
    reference, output = tmp_path / 'reference', tmp_path / 'outputs'
    (reference / 'primary').mkdir(parents=True)
    destination = output / tool.LOCATIONS['primary']
    destination.mkdir(parents=True)
    expected = pd.DataFrame({'count': [3, 5], 'rate': [.3, .5]})
    expected.to_csv(reference / 'primary/test.csv', index=False)
    def check():
        return tool.compare(reference, output, ['primary'])[0]['status']
    assert check() == 'missing'
    expected.assign(private_extra=['a', 'b']).to_csv(destination / 'test.csv', index=False)
    assert check() == 'match'
    expected.iloc[::-1].to_csv(destination / 'test.csv', index=False)
    assert check() == 'mismatch'
    expected.assign(count=[3, 6]).to_csv(destination / 'test.csv', index=False)
    assert check() == 'mismatch'
    assert tool.compare(reference, output, ['primary'])[0]['differing_columns'] == ['count']
    (destination / 'test.csv').write_text('')
    assert check() == 'mismatch'


def test_absent_reference_is_error(tmp_path):
    with pytest.raises(ValueError, match='No reference'):
        tool.compare(tmp_path, tmp_path, ['primary'])


def test_command_exit_codes_and_report_protection(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(tool, 'ROOT', tmp_path)
    reference = tmp_path / 'results/primary'
    reference.mkdir(parents=True)
    pd.DataFrame({'count': [2]}).to_csv(reference / 'test.csv', index=False)
    args = ['--data-root', str(tmp_path / 'data'), '--analyses', 'primary']
    assert tool.main(args) == 1
    report = tmp_path / 'report.json'
    assert tool.main(args + ['--allow-missing', '--report', str(report)]) == 0
    assert json.loads(report.read_text())['complete'] is False
    with pytest.raises(SystemExit):
        tool.main(args + ['--report', str(report)])
    actual = tmp_path / 'data/outputs/v3_five_benchmark_mirt_primary'
    actual.mkdir(parents=True)
    pd.DataFrame({'count': [3]}).to_csv(actual / 'test.csv', index=False)
    assert tool.main(args + ['--allow-missing']) == 1
    pd.DataFrame({'count': [2]}).to_csv(actual / 'test.csv', index=False)
    assert tool.main(args) == 0

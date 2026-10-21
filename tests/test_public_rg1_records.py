"""Saved-data verification, without real-graph model training or inference."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def test_saved_real_graph_statistics():
    from verify_rg1_reference import verify
    result=verify()
    assert result['case_count']==32
    assert result['record_count']==1536
    assert len(result['measurements'])==512


def test_saved_data_changes_are_rejected(tmp_path):
    import pytest
    from verify_rg1_reference import verify
    data=tmp_path/'rg1_data'; data.mkdir()
    (data/'all_records.json').write_text('[]')
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify(tmp_path)

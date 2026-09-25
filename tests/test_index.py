"""Local archive index: ObsCore level-2 rows joined to the level-2 tree by file name.

The fake archive mirrors the production layout: ``repo/level2/<week>`` are
symbolic links into ``loads/<load>/data/level2/<week>``; one load's ObsCore
table still lists a superseded processing version of a file (stale row), a
file is listed by two loads (duplicate row), and one load has an empty
``obscore/`` directory, so its file is on disk but unindexed.
"""
import os

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("pyarrow")

from spherex_retrieval import index as sidx  # noqa: E402

A = "level2_2025W17_4B_0001_1D2_spx_l2b-v19-2025-240.fits"
C = "level2_2025W17_4B_0001_1D1_spx_l2b-v19-2025-241.fits"
C_OLD = "level2_2025W17_4B_0001_1D1_spx_l2b-v19-2025-240.fits"  # superseded, not on disk
D = "level2_2025W18_1B_0002_2D3_spx_l2b-v20-2025-262.fits"
E = "level2_2025W19_1B_0003_1D4_spx_l2b-v20-2025-262.fits"    # its load has no ObsCore table
# an exposure reprocessed in the same week: both versions on disk, the later one wins
G_OLD = "level2_2025W18_1B_0009_2D5_spx_l2b-v20-2025-262.fits"
G_NEW = "level2_2025W18_1B_0009_2D5_spx_l2b-v20-2025-268.fits"
SIZES = {A: 71_634_240 % 997, C: 71_637_120 % 997, D: 71_622_720 % 997, E: 11, G_OLD: 5, G_NEW: 6}


def _row(name, level=2, ra=10.0, dec=20.0):
    m = sidx.LEVEL2_NAME_RE.match(name)
    did = f"{m['week']}_{m['expo']}_{m['dither']}D{m['det']}"
    kind = "level2" if level == 2 else "level1"
    return {
        "dataproduct_subtype": f"spherex.{kind}", "calib_level": level,
        "obs_id": did[:-2], "obs_creator_did": did,
        "s_ra": ra, "s_dec": dec, "t_min": 60789.75, "t_max": 60789.752,
        "em_min": 7.455e-07, "em_max": 1.1167e-06,
        # production tables pad the polygon with runs of spaces
        "s_region": f"POLYGON ICRS {ra - 1} {dec + 1} {ra + 1}          {dec + 1} "
                    f"{ra + 1} {dec - 1} {ra - 1} {dec - 1}",
        "access_url": f"file:///data/{kind}/{m['week']}/{m['procver']}/{m['det']}/{name}",
    }


def _write_obscore(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    Table(rows=rows).write(path, format="ipac", overwrite=True)


def _put_file(root, load, name):
    m = sidx.LEVEL2_NAME_RE.match(name)
    d = root / "loads" / load / "data" / "level2" / m["week"] / m["procver"] / m["det"]
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"\0" * SIZES[name])


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    _put_file(root, "L1", A)
    _put_file(root, "L1", C)
    _put_file(root, "L2", D)
    _put_file(root, "L2", G_OLD)
    _put_file(root, "L2", G_NEW)
    _put_file(root, "L3", E)
    _write_obscore(root / "loads/L1/obscore/obscore_L1.tbl",
                   [_row(A, level=1), _row(A, ra=30.0), _row(C_OLD), _row(C, ra=31.0)])
    _write_obscore(root / "loads/L2/obscore/obscore_L2.tbl",
                   [_row(C, ra=99.0), _row(D, ra=40.0), _row(G_NEW, ra=41.0), _row(G_OLD, ra=41.0)])
    (root / "loads/L3/obscore").mkdir(parents=True)
    level2 = root / "repo" / "level2"
    level2.mkdir(parents=True)
    for load, week in [("L1", "2025W17_4B"), ("L2", "2025W18_1B"), ("L3", "2025W19_1B")]:
        os.symlink(f"../../loads/{load}/data/level2/{week}", level2 / week)
    return root


def test_build_index_joins_obscore_to_the_tree(archive):
    table, report = sidx.build_index(archive, workers=2)
    rows = table.to_pylist()
    by_name = {r["path"].rsplit("/", 1)[-1]: r for r in rows}
    assert set(by_name) == {A, C, D, G_NEW}
    assert report["superseded"] == [f"repo/level2/2025W18_1B/l2b-v20-2025-262/5/{G_OLD}"]
    assert report["files_listed"] == 6
    assert report["obscore_level2_rows"] == 7
    assert report["stale_rows"] == 1 and report["duplicate_rows"] == 1
    # E is on disk in no table, and not a FITS file: its header scan fails
    assert report["unindexed"] == [f"repo/level2/2025W19_1B/l2b-v20-2025-262/4/{E}"]
    assert report["header_rows"] == 0 and list(report["header_failures"]) == \
        [str(archive / f"repo/level2/2025W19_1B/l2b-v20-2025-262/4/{E}")]
    assert set(table.column("meta").to_pylist()) == {"obscore"}
    assert table.column("path").to_pylist() == sorted(table.column("path").to_pylist())
    a = by_name[A]
    assert a["path"] == f"repo/level2/2025W17_4B/l2b-v19-2025-240/2/{A}"
    assert (a["week"], a["expo"], a["dither"], a["detector"], a["procver"]) == \
        ("2025W17_4B", 1, 1, 2, "l2b-v19-2025-240")
    assert a["obs_creator_did"] == "2025W17_4B_0001_1D2" and a["obs_id"] == "2025W17_4B_0001_1"
    assert a["size"] == SIZES[A] and a["load"] == "L1"
    assert (a["s_ra"], a["c1_ra"], a["c1_dec"], a["c2_ra"]) == (30.0, 29.0, 21.0, 31.0)
    # the first table (sorted) keeps a file that two loads list
    assert by_name[C]["s_ra"] == 31.0 and by_name[C]["load"] == "L1"
    assert (archive / by_name[D]["path"]).exists()
    corners = sidx.index_corners(table)
    assert corners.shape == (4, 4, 2)


def test_procver_order():
    key = sidx.procver_key
    assert key("l2b-v20-2025-269") > key("l2b-v20-2025-267") > key("l2b-v19-2025-300")
    assert key("l2b_retry-v26-2026-197") > key("l2b-v26-2026-196")
    assert key("weird") == -1


def test_write_read_roundtrip_keeps_the_report(archive, tmp_path):
    table, report = sidx.build_index(archive, workers=1)
    out = sidx.write_index(table, tmp_path / "idx" / "index.parquet", report)
    back = sidx.read_index(out)
    assert back.num_rows == 4 and back.column_names == table.column_names
    meta = sidx.index_metadata(out)
    assert meta["indexed"] == 4 and meta["unindexed_count"] == 1 and meta["superseded_count"] == 1
    assert meta["header_failure_count"] == 1
    assert "unindexed" not in meta and "header_failures" not in meta


def test_cli_build_and_info(archive, tmp_path, capsys):
    out = tmp_path / "index.parquet"
    miss = tmp_path / "unindexed.txt"
    assert sidx.main(["build", str(archive), "-o", str(out), "--workers", "2",
                      "--unindexed-out", str(miss)]) == 0
    assert miss.read_text().strip().endswith(E)
    assert sidx.main(["info", str(out)]) == 0
    text = capsys.readouterr().out
    assert "rows: 4" in text and "per detector" in text


def test_read_ipac_columns_ragged_rows_and_nulls(tmp_path):
    p = tmp_path / "t.tbl"
    p.write_text(
        "\\tbl.relatedCols='obs_id'\n"
        "|  calib_level|      s_ra|                 s_region|\n"
        "|         long|    double|                     char|\n"
        "|             |       deg|                         |\n"
        "|         null|      null|                     null|\n"
        "             2      12.5  POLYGON ICRS 1 2 3 4 5 6 7 8\n"
        "             2       null POLYGON ICRS 1 2 3 4 5 6 7 8\n"
        "             1       7.0\n"
    )
    cols = sidx.read_ipac_columns(p, ("calib_level", "s_ra", "s_region"))
    assert list(cols["calib_level"]) == [b"2", b"2", b"1"]
    f = sidx._to_float(cols["s_ra"])
    assert f[0] == 12.5 and np.isnan(f[1]) and f[2] == 7.0
    assert cols["s_region"][2] == b""
    with pytest.raises(ValueError):
        sidx.parse_s_region(np.array([b"CIRCLE ICRS 1 2 3"]))

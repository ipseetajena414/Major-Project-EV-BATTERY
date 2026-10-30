"""Member 5 boundary tests. Run from repo root: python -m pytest tests -q"""
import tempfile
from pathlib import Path
import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"

from m5integration import member2_adapter as m2, member3_adapter as m3, member4_adapter as m4
from rag.document_loader import SourceMeta, load_text
from rag.text_chunker import chunk_document
from rag.vector_store import VectorStore
from rag.qa_engine import answer, EMPTY_MSG

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="ENVIRONMENT-BLOCKED: real torch is not installed")

OK_CONTRACT = {"status": "ok_experimental", "soh": {"current_observed_soh_pct": 88.0, "predicted_soh_pct": 85.5}}


# 1/2: Member 2 -> Member 3 SOH handling
def test_observed_soh_is_default_anchor_in_percent():
    s = m2.soh_for_range(OK_CONTRACT)
    assert s["allowed"] and s["soh_pct"] == 88.0 and s["soh_pct"] > 1.0


def test_predicted_soh_is_labelled_experimental():
    s = m2.soh_for_range(OK_CONTRACT, "predicted")
    assert s["soh_pct"] == 85.5 and "EXPERIMENTAL" in s["label"]


# 2: Member 3 receives accepted cycle + valid SOH
def test_member3_estimate_runs_with_real_cycle():
    soh = m2.soh_for_range(OK_CONTRACT)
    r = m3.estimate("tata_curvv_ev_55kwh_partial_reference.json", "udds_city.csv", soh, 25.0)
    assert r["estimated_range_km"] > 0 and r["soh_pct"] == 88.0
    assert r["quality_status"] == "REQUIRES_VALIDATION"


def test_lower_soh_gives_lower_range():
    hi = m3.estimate("tata_curvv_ev_55kwh_partial_reference.json", "udds_city.csv", {"allowed": True, "soh_pct": 95.0, "label": "t"}, 25)
    lo = m3.estimate("tata_curvv_ev_55kwh_partial_reference.json", "udds_city.csv", {"allowed": True, "soh_pct": 75.0, "label": "t"}, 25)
    assert lo["estimated_range_km"] < hi["estimated_range_km"]


# 4: blocked Member 2 result cannot become a range estimate
@pytest.mark.parametrize("contract", [None, {"status": "blocked", "soh": {}}, {"status": "error", "errors": [{"code": "x"}]}])
def test_blocked_member2_result_blocks_range(contract):
    soh = m2.soh_for_range(contract)
    assert soh["allowed"] is False and soh["soh_pct"] is None
    with pytest.raises(m3.RangeBlocked):
        m3.estimate("tata_curvv_ev_55kwh_partial_reference.json", "udds_city.csv", soh, 25.0)


# 5: RUL never consumed
def test_rul_is_never_a_decision_signal():
    assert m2.rul_signal({"rul": {"value": 123}}) is None


# 6: unsupported chemistries visible
def test_unsupported_chemistries_have_reasons():
    for chem in ("Na-ion", "Solid-State"):
        assert chem not in m2.SUPPORTED and m2.UNSUPPORTED_REASONS[chem]


# 3: Member 4 only gets supported inputs; safety rules are deterministic Python
def test_member4_bms_rules_are_deterministic():
    ok, err = m4.availability()
    assert ok, err
    a = m4.evaluate_request(3.0, 100, 5, 44, "full_protection")
    b = m4.evaluate_request(3.0, 100, 5, 44, "full_protection")
    assert a == b and a["decision"] == "reject" and a["target_soc"] == 80.0 and a["min_soc"] == 20.0


def test_member4_simulation_runs_and_protection_helps():
    u = m4.simulate("LFP", "unrestricted", "hot")
    p = m4.simulate("LFP", "full_protection", "hot")
    assert p["years_to_eol"] > u["years_to_eol"]


def test_vehicle_configs_flag_provenance():
    for v in m3.list_vehicles():
        assert m3.vehicle_info(v)["provenance_status"] != "UNKNOWN"


# 7: RAG cites only retrieved sources
def _store():
    s = VectorStore(tempfile.mkdtemp())
    txt = ("Sulfide solid electrolytes reach high ionic conductivity yet lithium dendrites limit the critical current density. " * 6)
    s.add(chunk_document(load_text(txt, SourceMeta("Test fixture A", year="2000"))))
    return s


def test_rag_empty_corpus_message():
    r = answer(VectorStore(tempfile.mkdtemp()), "anything")
    assert r["status"] == "empty_corpus" and r["message"] == EMPTY_MSG and r["passages"] == []


def test_rag_answers_cite_retrieved_sources_only():
    s = _store()
    r = answer(s, "critical current density dendrites")
    assert r["status"] == "retrieved" and r["passages"]
    assert {p["source"]["title"] for p in r["passages"]} <= {x["title"] for x in s.sources()}
    assert all(p["source"]["authors"] == "" for p in r["passages"])  # nothing invented


def test_rag_index_persists():
    d = tempfile.mkdtemp()
    s = VectorStore(d); s.add(chunk_document(load_text("sodium ion layered oxide cathode cycle life " * 10, SourceMeta("Persist"))))
    assert len(VectorStore(d).chunks) == len(s.chunks) > 0


# 8: missing module does not crash the dashboard
def test_dashboard_pages_render_without_exceptions():
    from streamlit.testing.v1 import AppTest
    for page in ("Overview", "EV Range", "Multi-Agent Simulation", "Literature Assistant (RAG)"):
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])


def test_member4_missing_is_a_visible_disabled_state(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(m4, "availability", lambda: (False, "ImportError: simulated"))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Multi-Agent Simulation").run()
    assert not at.exception
    assert any("simulated" in e.value for e in at.error)


# ---- corrections: dataset filtering, Member 1 flag, nested quality display ----
def test_dataset_id_is_applied_not_ignored():
    # wrong dataset for a real cell/cycle must fail, never silently fall back to the cell's real dataset
    frame, err = m2._cycle_rows("B0005", 100, "NOT_A_DATASET")
    assert frame is None and "NOT_A_DATASET" in err
    frame, err = m2._cycle_rows("B0005", 100, "NASA_Ames_PCoE")
    assert err is None and set(frame["dataset_id"]) == {"NASA_Ames_PCoE"}
    assert m2.member1_quality_flag("B0005", 100, "NOT_A_DATASET") is None


def test_member1_quality_flag_is_verbatim_and_separate():
    assert m2.member1_quality_flag("B0005", 100, "NASA_Ames_PCoE") == "VALID"
    multi = m2.member1_quality_flag("BATTERY_5", 100, "LFP_MultiStat_B5")
    assert multi and "TEMP_UNMEASURED" in multi


def test_quality_flag_reaches_member3():
    soh = {"allowed": True, "soh_pct": 90.0, "label": "t"}
    r = m3.estimate("tata_curvv_ev_55kwh_partial_reference.json", "udds_city.csv", soh, 25.0, quality_flag="VALID")
    assert r["quality_flag"] == "VALID"


@needs_torch
def test_real_contract_has_nested_quality_and_dataset_provenance():
    pred, err = m2.load_predictor()
    assert pred, err
    c, e = m2.health_for_cycle(pred, "B0005", 100, "NMC", dataset_id="NASA_Ames_PCoE")
    assert e is None and c["status"] == "ok_experimental"
    assert c["subject"]["dataset_id"] == "NASA_Ames_PCoE" and c["subject"]["cell_id"] == "B0005"
    q = c["quality"]
    for k in ("quality_flags", "warnings", "checks_passed", "checks_not_performed", "label_provenance"):
        assert k in q
    assert "quality_flags" not in c and "warnings" not in c  # schema untouched: nested only
    assert "member1_quality_flag" not in c and "quality_flag" not in c


@needs_torch
def test_battery_health_page_shows_nested_quality_fields():
    from streamlit.testing.v1 import AppTest
    pred, _ = m2.load_predictor()
    c, _ = m2.health_for_cycle(pred, "BATTERY_5", 100, "LFP", dataset_id="LFP_MultiStat_B5")
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.session_state["health_contract"] = c
    at.session_state["health_label"] = "LFP test"
    at.session_state["health_m1_flag"] = m2.member1_quality_flag("BATTERY_5", 100, "LFP_MultiStat_B5")
    at.sidebar.radio[0].set_value("Battery Health & SOH").run()
    text = " ".join(m.value for m in at.markdown)
    for label in ("quality_flags", "warnings", "checks_passed", "checks_not_performed", "label_provenance", "Member 1 quality_flag"):
        assert label in text, label
    assert not at.exception


def test_pdf_upload_path_extracts_text_and_is_retrievable(tmp_path):
    """Generates a small real PDF (test fixture, not a paper) and runs load_pdf -> chunk -> index -> answer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rag.document_loader import load_pdf
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.9, "Prussian blue analogue cathodes for sodium-ion batteries show good rate capability " * 1, fontsize=9)
    fig.text(0.05, 0.85, "and layered oxide cathodes deliver higher energy density but weaker air stability in sodium-ion cells " * 1, fontsize=9)
    fig.text(0.05, 0.80, "polyanionic cathodes trade energy density for long cycle life and thermal stability in sodium-ion cells", fontsize=9)
    pdf = tmp_path / "fixture.pdf"
    fig.savefig(pdf)
    doc = load_pdf(str(pdf), SourceMeta("PDF fixture", year="2000"))
    assert doc.pages, "no text extracted from fixture PDF"
    s = VectorStore(tempfile.mkdtemp())
    s.add(chunk_document(doc))
    r = answer(s, "layered oxide energy density air stability")
    assert r["status"] == "retrieved" and r["passages"][0]["source"]["title"] == "PDF fixture"
    assert r["passages"][0]["page"] == 1

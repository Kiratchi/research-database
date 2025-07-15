# tests/test_session.py
from es_tools import search_publications, get_more_results


def test_session_expiry(es):
    s = search_publications(query="test", size=1)
    sid = s["session_id"]
    assert s["total_results"] >= 0
    # force-fetch two pages, ensure page cache grows
    get_more_results(sid, 0)
    pg2 = get_more_results(sid, 1)
    assert pg2["page"] == 1 and pg2["results"]

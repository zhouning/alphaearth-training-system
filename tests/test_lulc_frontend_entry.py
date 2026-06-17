from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "ae_frontend" / "index.html"


def test_frontend_exposes_lulc_tool_tab_and_api_actions():
    html = FRONTEND.read_text(encoding="utf-8")

    assert "LULC 工具" in html
    assert "currentTab = 'lulc'" in html
    assert "currentTab === 'lulc'" in html
    assert "公共产品模式" in html
    assert "本地模型模式" in html
    assert "/api/ae/inference/lulc/modes" in html
    assert "/api/ae/inference/lulc/public" in html
    assert "/api/ae/inference/lulc/evaluate" in html
    assert "/api/ae/inference/lulc" in html
    assert "computed" in html.split("= Vue", maxsplit=1)[0]

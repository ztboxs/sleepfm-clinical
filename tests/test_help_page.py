from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_help_route_is_registered() -> None:
    source = (ROOT / "api/main.py").read_text(encoding="utf-8")
    assert '@app.get("/help")' in source
    assert 'FileResponse(os.path.join(STATIC_DIR, "docs.html"))' in source


def test_console_header_links_to_help() -> None:
    html = (ROOT / "api/static/index.html").read_text(encoding="utf-8")
    assert 'class="help-menu"' in html
    assert 'href="/help"' in html
    assert 'href="/docs"' in html


def test_documentation_shell_fetches_markdown_instead_of_copying_it() -> None:
    html = (ROOT / "api/static/docs.html").read_text(encoding="utf-8")
    assert 'id="docNavigation"' in html
    assert 'id="docContent"' in html
    assert "/api/v1/docs" in html
    assert "marked.parse" in html
    assert "LOO-CL 预训练" not in html

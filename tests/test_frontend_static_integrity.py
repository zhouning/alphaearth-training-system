import shutil
import subprocess
from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "ae_frontend" / "index.html"


def test_frontend_inline_application_script_is_valid_javascript(tmp_path):
    node = shutil.which("node")
    assert node is not None, "Node.js is required for frontend syntax validation"

    html = FRONTEND.read_text(encoding="utf-8")
    start = html.rfind("<script>")
    end = html.rfind("</script>")
    assert start != -1
    assert end > start

    script = html[start + len("<script>") : end]
    script_path = tmp_path / "alphaearth-index-script.js"
    script_path.write_text(script, encoding="utf-8")

    result = subprocess.run(
        [node, "--check", str(script_path)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

"""Tests for the CI review script in .github/scripts/claude_review.py."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "claude_review.py"
_spec = importlib.util.spec_from_file_location("claude_review", _SCRIPT)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

DIFF = """\
diff --git a/huehub/a.py b/huehub/a.py
index 111..222 100644
--- a/huehub/a.py
+++ b/huehub/a.py
@@ -10,3 +10,4 @@ def f():
     keep = 1
-    old = 2
+    new = 2
+    extra = 3
     tail = 4
diff --git a/old.txt b/old.txt
deleted file mode 100644
--- a/old.txt
+++ /dev/null
@@ -1,1 +0,0 @@
-gone
diff --git a/new.md b/new.md
new file mode 100644
--- /dev/null
+++ b/new.md
@@ -0,0 +1,2 @@
+line one
+line two
\\ No newline at end of file
"""


class TestCommentableLines:
    def test_added_and_context_lines(self) -> None:
        lines = cr.commentable_lines(DIFF)
        assert lines["huehub/a.py"] == {10, 11, 12, 13}

    def test_new_file(self) -> None:
        assert cr.commentable_lines(DIFF)["new.md"] == {1, 2}

    def test_deleted_file_has_no_lines(self) -> None:
        assert "old.txt" not in cr.commentable_lines(DIFF)


class TestBuildPrompt:
    def test_contains_all_parts(self) -> None:
        prompt = cr.build_prompt(
            "Fix it", "DIFF", {"a.py": "CODE"}, {"README.md": "DOCS"}
        )
        assert "<pr_title>Fix it</pr_title>" in prompt
        assert '<file path="a.py">\nCODE\n</file>' in prompt
        assert '<doc path="README.md">\nDOCS\n</doc>' in prompt
        assert prompt.index("<diff>") > prompt.index("<file")


class TestToReviewPayload:
    def test_splits_inline_and_outside(self) -> None:
        review = cr.Review(
            summary="Two issues.",
            findings=[
                cr.Finding(path="huehub/a.py", line=11, kind="bug", body="Wrong."),
                cr.Finding(path="huehub/b.py", line=5, kind="docs", body="Stale."),
            ],
        )
        payload = cr.to_review_payload(review, cr.commentable_lines(DIFF), "abc")
        assert payload["commit_id"] == "abc"
        assert payload["event"] == "COMMENT"
        assert payload["comments"] == [
            {
                "path": "huehub/a.py",
                "line": 11,
                "side": "RIGHT",
                "body": "**bug:** Wrong.",
            }
        ]
        assert "`huehub/b.py:5` **docs:** Stale." in payload["body"]

    def test_no_findings(self) -> None:
        payload = cr.to_review_payload(
            cr.Review(summary="Clean.", findings=[]), {}, "x"
        )
        assert payload["comments"] == []
        assert "Outside the diff" not in payload["body"]


class TestReadChangedFiles:
    def test_skips_missing_binary_and_huge(self, tmp_path: Path) -> None:
        (tmp_path / "ok.py").write_text("print()")
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00")
        (tmp_path / "big.txt").write_text("x" * (cr.MAX_FILE_CHARS + 1))
        names = {"ok.py", "bin.dat", "big.txt", "missing.py"}
        assert cr.read_changed_files(names, tmp_path) == {"ok.py": "print()"}


class _ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple]:
    fake_anthropic = types.SimpleNamespace(
        APIStatusError=_ApiError, APIConnectionError=ConnectionError
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("PR_NUMBER", "7")
    monkeypatch.setenv("PR_CHECKOUT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    calls: list[tuple] = []

    def fake_gh(*args: str, stdin: str | None = None) -> str:
        calls.append((args, stdin))
        if args[:2] == ("pr", "view"):
            return json.dumps({"title": "T", "headRefOid": "sha1"})
        if args[:2] == ("pr", "diff"):
            return DIFF
        return ""

    monkeypatch.setattr(cr, "_gh", fake_gh)
    return calls


class TestMain:
    def test_posts_review(
        self, fake_env: list[tuple], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        review = cr.Review(
            summary="One.",
            findings=[cr.Finding(path="new.md", line=2, kind="docs", body="x")],
        )
        monkeypatch.setattr(cr, "call_claude", lambda prompt: review)
        assert cr.main() == 0
        args, stdin = fake_env[-1]
        assert args[:2] == ("api", "repos/o/r/pulls/7/reviews")
        assert json.loads(stdin)["comments"][0]["path"] == "new.md"

    def test_api_error_fails_with_message(
        self,
        fake_env: list[tuple],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def boom(prompt: str) -> None:
            raise _ApiError(400, "credit balance is too low")

        monkeypatch.setattr(cr, "call_claude", boom)
        assert cr.main() == 1
        assert "credit balance is too low" in caplog.text

    def test_refusal_posts_comment(
        self, fake_env: list[tuple], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cr, "call_claude", lambda prompt: None)
        assert cr.main() == 0
        assert fake_env[-1][0][:2] == ("pr", "comment")

    def test_huge_diff_skipped(
        self, fake_env: list[tuple], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cr, "MAX_DIFF_CHARS", 10)
        monkeypatch.setattr(cr, "call_claude", pytest.fail)
        assert cr.main() == 0
        assert "Claude review skipped" in fake_env[-1][0][-1]

"""Offline checks for authenticated judge transport and redacted errors."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import urllib.response

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation import run_judge


TOKEN = "audit-dummy-credential"
PRIVATE_RESPONSE = "audit-private-response-content"


class OfflineHTTPSHandler(urllib.request.HTTPSHandler):
    """Exercise urllib processing without opening a connection."""

    def __init__(self, status=200, body=None, location=None):
        super().__init__()
        self.status = status
        self.body = body if body is not None else json.dumps({
            "model": "audit-model", "choices": [{"message": {"content": "1"}, "finish_reason": "stop"}]
        }).encode()
        self.location = location
        self.requests = []

    def https_open(self, request):
        self.requests.append(request)
        headers = {"Location": self.location} if self.location else {}
        response = urllib.response.addinfourl(io.BytesIO(self.body), headers, request.full_url, self.status)
        response.msg = "Offline test response"
        return response


class JudgeSecurityTests(unittest.TestCase):
    def test_valid_https_endpoints(self):
        cases = {
            "https://api.example.com/v1/": "https://api.example.com/v1/chat/completions",
            "https://api.example.com": "https://api.example.com/chat/completions",
            "https://localhost:8443/v1": "https://localhost:8443/v1/chat/completions",
            "https://127.0.0.1:8443": "https://127.0.0.1:8443/chat/completions",
            "https://[::1]:8443/v1": "https://[::1]:8443/v1/chat/completions",
        }
        for base, expected in cases.items():
            with self.subTest(base=base):
                self.assertEqual(run_judge.judge_endpoint(base), expected)

    def test_invalid_urls_are_rejected_without_echoing_the_value(self):
        invalid = [
            "", "http://api.example.com/v1", "https:///v1", "https://",
            "https://user:" + TOKEN + "@api.example.com/v1",
            "https://api.example.com/v1?api_key=" + TOKEN,
            "https://api.example.com/v1#" + TOKEN,
            "https://api.example.com/v1?", "https://api.example.com/v1#",
            "https://bad host/v1", "https://bad_host/v1", "https://-bad.example/v1",
            "https://bad..example/v1", "https://bad.example./v1\n",
            "https://999.1.1.1/v1", "https://[not-ipv6]/v1", "https://[::1]unexpected/v1",
            "https://api.example.com:bad/v1", "https://api.example.com:65536/v1",
            "https://api.example.com:0/v1", "https://api.example.com:/v1",
            "https://api.example.com\\@different.example/v1",
            "https://api.example.com/" + chr(127),
        ]
        for base in invalid:
            with self.subTest(base=base):
                with self.assertRaises(ValueError) as error:
                    run_judge.judge_endpoint(base)
                self.assertNotIn(TOKEN, str(error.exception))
                self.assertNotIn("api.example.com", str(error.exception))

    def test_redirects_never_issue_a_second_request(self):
        for status in (301, 302, 303, 307, 308):
            for target in ("https://different.example/collect", "https://api.example.com/other", "http://different.example/collect"):
                with self.subTest(status=status, target=target):
                    handler = OfflineHTTPSHandler(status, PRIVATE_RESPONSE.encode(), target)
                    opener = urllib.request.build_opener(handler, run_judge.NoRedirectHandler())
                    with self.assertRaises(SystemExit) as error:
                        run_judge.request_judgment(opener, "https://api.example.com/v1/chat/completions", TOKEN, {})
                    self.assertEqual(len(handler.requests), 1)
                    self.assertIn(str(status), str(error.exception))
                    self.assertNotIn(TOKEN, str(error.exception))
                    self.assertNotIn(PRIVATE_RESPONSE, str(error.exception))
                    self.assertNotIn(target, str(error.exception))

    def test_normal_authenticated_post_still_succeeds(self):
        handler = OfflineHTTPSHandler()
        opener = urllib.request.build_opener(handler, run_judge.NoRedirectHandler())
        value, choice, answer = run_judge.request_judgment(opener, "https://api.example.com/v1/chat/completions", TOKEN, {"model": "audit-model"})
        request = handler.requests[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
        self.assertEqual(json.loads(request.data), {"model": "audit-model"})
        self.assertEqual((value["model"], choice["finish_reason"], answer), ("audit-model", "stop", "1"))

    def test_network_and_response_errors_do_not_echo_private_content(self):
        failures = [urllib.error.URLError(TOKEN), ValueError(PRIVATE_RESPONSE), RuntimeError(TOKEN)]
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                class BrokenOpener:
                    def open(self, *_args, **_kwargs):
                        raise failure
                with self.assertRaises(SystemExit) as error:
                    run_judge.request_judgment(BrokenOpener(), "https://api.example.com/v1/chat/completions", TOKEN, {})
                self.assertNotIn(TOKEN, str(error.exception))
                self.assertNotIn(PRIVATE_RESPONSE, str(error.exception))
        for body in (PRIVATE_RESPONSE.encode(), b'{}', b'{"choices":[]}',
                     json.dumps({"choices": [{"message": {"content": PRIVATE_RESPONSE}, "finish_reason": "stop"}]}).encode()):
            handler = OfflineHTTPSHandler(body=body)
            opener = urllib.request.build_opener(handler, run_judge.NoRedirectHandler())
            with self.assertRaises(SystemExit) as error:
                run_judge.request_judgment(opener, "https://api.example.com/v1/chat/completions", TOKEN, {})
            self.assertNotIn(PRIVATE_RESPONSE, str(error.exception))
        handler = OfflineHTTPSHandler(status=401, body=(TOKEN + PRIVATE_RESPONSE).encode())
        opener = urllib.request.build_opener(handler, run_judge.NoRedirectHandler())
        with self.assertRaises(SystemExit) as error:
            run_judge.request_judgment(opener, "https://api.example.com/v1/chat/completions", TOKEN, {})
        self.assertIn("401", str(error.exception))
        self.assertNotIn(TOKEN, str(error.exception))
        self.assertNotIn(PRIVATE_RESPONSE, str(error.exception))

    def test_cli_preserves_output_schema_and_rejects_invalid_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            jobs, output = directory / "jobs.jsonl", directory / "result.jsonl"
            job = {"job_id": "audit-job", "input": {"candidate": "test"}}
            jobs.write_text(json.dumps(job) + "\n")
            argv = ["run_judge.py", "--jobs", str(jobs), "--output", str(output), "--model", "audit-model"]
            handler = OfflineHTTPSHandler()
            opener = urllib.request.build_opener(handler, run_judge.NoRedirectHandler())
            with patch.object(sys, "argv", argv), patch.dict("os.environ", {"JUDGE_BASE_URL": "https://api.example.com/v1", "JUDGE_API_KEY": TOKEN}), patch.object(urllib.request, "build_opener", return_value=opener), contextlib.redirect_stdout(io.StringIO()):
                run_judge.main()
            record = json.loads(output.read_text())
            self.assertEqual(record, {"job_id": "audit-job", "input": job["input"], "correct": 1,
                "requested_model": "audit-model", "returned_model": "audit-model", "raw_answer": "1", "finish_reason": "stop"})
            self.assertNotIn(TOKEN, output.read_text())
            invalid_output = directory / "invalid.jsonl"
            for invalid_key in (TOKEN + "\r\n", TOKEN + "\t", TOKEN + "\u00e9"):
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["run_judge.py", "--jobs", str(jobs), "--output", str(invalid_output)]), patch.dict("os.environ", {"JUDGE_BASE_URL": "https://api.example.com/v1", "JUDGE_API_KEY": invalid_key}), patch.object(urllib.request, "build_opener") as build, contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    run_judge.main()
                build.assert_not_called()
                self.assertFalse(invalid_output.exists())
                self.assertNotIn(TOKEN, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

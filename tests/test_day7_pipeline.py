import argparse
import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from day7_pipeline import (
    OpenAIController,
    MultiMethodRetriever,
    _evidence_satisfied,
    _prioritize_caption_chunks,
    _rank_fusion,
    _run_agent,
    _split_page,
    extract_pdf_page_images,
    load_questions,
)


class MockResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


class Day7PipelineTests(unittest.TestCase):
    def test_benchmark_meets_basic_split_minimums(self):
        questions = load_questions(ROOT / "data" / "nasa_science_plan_questions.json")
        for split, expected_answerable in (("dev", 15), ("test", 5)):
            selected = [q for q in questions if q["split"] == split]
            self.assertGreaterEqual(
                sum(q.get("answerable", True) for q in selected), expected_answerable
            )
            self.assertGreaterEqual(
                sum(not q.get("answerable", True) for q in selected), 2
            )
        self.assertGreaterEqual(len(questions), 24)

    def test_advanced_benchmark_meets_split_and_group_requirements(self):
        questions = load_questions(ROOT / "data" / "cassini_advanced_questions.json")
        for split, answerable_minimum, missing_minimum in (("dev", 15, 3), ("test", 24, 6)):
            selected = [q for q in questions if q["split"] == split]
            answerable = [q for q in selected if q.get("answerable", True)]
            missing = [q for q in selected if not q.get("answerable", True)]
            self.assertGreaterEqual(len(answerable), answerable_minimum)
            self.assertGreaterEqual(len(missing), missing_minimum)
        test_groups = {
            group: sum(
                q.get("answerable", True)
                and q["primary_group"] == group
                for q in questions
                if q["split"] == "test"
            )
            for group in ("table", "image", "chart", "multi_hop", "text")
        }
        self.assertGreaterEqual(test_groups["table"], 6)
        self.assertGreaterEqual(test_groups["image"], 5)
        self.assertGreaterEqual(test_groups["chart"], 5)
        self.assertGreaterEqual(test_groups["multi_hop"], 4)
        self.assertGreaterEqual(test_groups["text"], 4)

    def test_advanced_benchmark_gold_quotes_resolve_in_the_cassini_pdf(self):
        try:
            from pypdf import PdfReader
        except ImportError:
            self.skipTest("pypdf is not installed")
        pdf_path = os.environ.get("CASSINI_TEST_PDF")
        if not pdf_path or not Path(pdf_path).is_file():
            self.skipTest("Set CASSINI_TEST_PDF to the local Cassini PDF to validate gold quotes")
        reader = PdfReader(pdf_path)
        questions = load_questions(ROOT / "data" / "cassini_advanced_questions.json")
        failures = []
        for question in questions:
            if not question.get("answerable", True):
                continue
            for gold in question["gold"]:
                page_text = " ".join(
                    (reader.pages[gold["page"] - 1].extract_text() or "").split()
                )
                if gold["evidence"] not in page_text:
                    failures.append(f"{question['id']} page {gold['page']}")
        self.assertEqual(failures, [])

    def test_benchmark_gold_quotes_resolve_in_the_user_pdf(self):
        try:
            from pypdf import PdfReader
        except ImportError:
            self.skipTest("pypdf is not installed")
        pdf_path = os.environ.get("NASA_SCIENCE_PLAN_TEST_PDF")
        if not pdf_path or not Path(pdf_path).is_file():
            self.skipTest("Set NASA_SCIENCE_PLAN_TEST_PDF to the local NASA PDF to validate gold quotes")
        reader = PdfReader(pdf_path)
        questions = load_questions(ROOT / "data" / "nasa_science_plan_questions.json")
        failures = []
        for question in questions:
            if not question.get("answerable", True):
                continue
            for gold in question["gold"]:
                page_text = " ".join((reader.pages[gold["page"] - 1].extract_text() or "").split())
                if gold["evidence"] not in page_text:
                    failures.append(f"{question['id']} page {gold['page']}")
        self.assertEqual(failures, [])

    def test_sentence_chunking_keeps_overlap_and_respects_maximum(self):
        text = "First evidence sentence. " + ("long phrase " * 120) + "Final evidence sentence."
        chunks = _split_page(text, chunk_chars=120, overlap_chars=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertIn("First evidence sentence.", chunks[0])
        self.assertIn("Final evidence sentence.", chunks[-1])

    def test_hybrid_fusion_combines_rankings_and_deduplicates(self):
        dense = [
            {"id": "a", "text": "dense-a", "score": 0.9},
            {"id": "b", "text": "dense-b", "score": 0.8},
        ]
        lexical = [
            {"id": "b", "text": "lex-b", "score": 0.7},
            {"id": "c", "text": "lex-c", "score": 0.6},
        ]
        result = _rank_fusion(dense, lexical)
        self.assertEqual([row["id"] for row in result], ["b", "a", "c"])
        self.assertEqual(len({row["id"] for row in result}), 3)
        self.assertEqual(result[0]["strategy"], "hybrid")

    def test_tfidf_agent_retrieval_does_not_require_dense_index(self):
        chunks = [
            {
                "id": "evidence",
                "text": "Cassini planned a Titan probe flyby.",
                "source_id": "cassini",
                "source": "cassini.pdf",
                "page": 1,
                "metadata": {},
            }
        ]
        retriever = MultiMethodRetriever(chunks, None, "tfidf")

        results = retriever.search("Titan probe", filters=None, top_k=1)

        self.assertEqual([row["id"] for row in results], ["evidence"])

    def test_tfidf_agent_does_not_build_a_dense_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(json.dumps([{
                "id": "p1-c1",
                "text": "Test evidence.",
                "source_id": "test",
                "source": "test.pdf",
                "page": 1,
                "metadata": {},
            }]), encoding="utf-8")
            questions_path = root / "questions.json"
            questions_path.write_text(json.dumps([{
                "id": "q1",
                "split": "dev",
                "question": "What is in the report?",
                "answerable": True,
                "gold": [{
                    "source_id": "test",
                    "page": 1,
                    "evidence": "Test evidence.",
                }],
            }]), encoding="utf-8")
            args = argparse.Namespace(
                split="dev",
                finalize=False,
                max_searches=3,
                top_k=5,
                context_budget=4000,
                request_timeout=60,
                max_output_tokens=2048,
                run_name="tfidf",
                questions=questions_path,
                chunks=chunks_path,
                output=root / "results",
                image_dir=None,
                model=None,
                method="tfidf",
            )
            controller = SimpleNamespace(
                model="mock",
                max_image_pages=1,
                usage={
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_ms": [],
                },
            )
            with (
                patch("day7_pipeline.DenseQdrantIndex") as dense_index,
                patch("day7_pipeline.OpenAIController", return_value=controller),
                patch("day7_pipeline.run_agent", return_value={
                    "status": "answered",
                    "answer": "Test evidence. [p1-c1]",
                    "citations": ["p1-c1"],
                    "events": [],
                }),
            ):
                report = _run_agent(args)

        dense_index.assert_not_called()
        self.assertEqual(report["method"], "tfidf")

    def test_multi_hop_retrieval_requires_evidence_from_every_gold_item(self):
        question = {
            "evidence_mode": "all",
            "gold": [
                {"source_id": "cassini", "page": 41, "evidence": "Titan is the largest"},
                {"source_id": "cassini", "page": 42, "evidence": "Radius (km) 2575"},
            ],
        }
        first = {
            "source_id": "cassini",
            "page": 41,
            "text": "Titan is the largest Saturnian satellite.",
        }
        second = {
            "source_id": "cassini",
            "page": 42,
            "text": "Titan Radius (km) 2575.",
        }
        self.assertFalse(_evidence_satisfied([first], question))
        self.assertTrue(_evidence_satisfied([first, second], question))

    def test_explicit_figure_and_table_ids_prioritize_their_caption_chunks(self):
        table_chunk = {"id": "table", "text": "Table 3.2: Candidate data"}
        figure_chunk = {"id": "figure", "text": "Figure 8.1: Spacecraft drawing"}
        second_figure_chunk = {"id": "figure-2", "text": "Figure 3.11: Magnetosphere"}
        unrelated_chunk = {"id": "reference", "text": "Figure 8.1 is referenced here"}
        ranked = [{"id": "reference", "text": unrelated_chunk["text"], "score": 0.9}]
        result = _prioritize_caption_chunks(
            "Compare Table 3.2 with Figure 8.1.",
            ranked,
            [unrelated_chunk, figure_chunk, table_chunk],
        )
        self.assertEqual([row["id"] for row in result], ["table", "figure", "reference"])
        self.assertEqual(result[0]["strategy"], "caption")
        self.assertEqual(result[2]["score"], 0.9)
        multi_figure_result = _prioritize_caption_chunks(
            "Compare Figures 3.10 and 3.11.",
            [],
            [
                second_figure_chunk,
                {"id": "figure-1", "text": "Figure 3.10: Inner magnetosphere"},
            ],
        )
        self.assertEqual(
            [row["id"] for row in multi_figure_result],
            ["figure-1", "figure-2"],
        )

    def test_page_image_extraction_reports_pages_without_images(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_image = root / "source.jpg"
            Image.new("RGB", (2, 2), color="white").save(source_image, format="JPEG")
            page_with_image = SimpleNamespace(
                images=[SimpleNamespace(data=source_image.read_bytes())]
            )
            page_without_image = SimpleNamespace(images=[])
            pdf_path = root / "fake.pdf"
            pdf_path.write_bytes(b"fixture")
            with patch(
                "pypdf.PdfReader",
                return_value=SimpleNamespace(pages=[page_with_image, page_without_image]),
            ):
                result = extract_pdf_page_images(pdf_path, root / "pages")
            self.assertEqual(result["page_count"], 2)
            self.assertEqual(result["image_page_count"], 1)
            self.assertEqual(result["pages_without_embedded_image"], [2])
            self.assertTrue((root / "pages" / "page-1.jpg").is_file())

    def test_api_controller_parses_search_and_tracks_usage_without_real_request(self):
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"Artemis"}',
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 21, "completion_tokens": 4},
        }
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }):
            with patch("day7_pipeline.urllib.request.urlopen", return_value=MockResponse(payload)) as call:
                controller = OpenAIController(
                    "test-model", timeout=123, max_output_tokens=256
                )
                action = controller.decide("NASA?", [])
        self.assertEqual(action, {"action": "search", "query": "Artemis"})
        self.assertEqual(call.call_args.kwargs["timeout"], 123)
        self.assertEqual(controller.usage["requests"], 1)
        self.assertEqual(controller.usage["prompt_tokens"], 21)
        self.assertEqual(controller.usage["completion_tokens"], 4)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(
            body["tool_choice"],
            {"type": "function", "function": {"name": "search"}},
        )
        self.assertIsInstance(body["messages"][-1]["content"], str)

    def test_api_controller_attaches_retrieved_page_images(self):
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"Table 3.1"}',
                        }
                    }]
                }
            }],
            "usage": {},
        }
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page-42.jpg"
            Image.new("RGB", (2, 2), color="white").save(image_path, format="JPEG")
            image_bytes = image_path.read_bytes()
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
            }):
                with patch(
                    "day7_pipeline.urllib.request.urlopen",
                    return_value=MockResponse(payload),
                ) as call:
                    controller = OpenAIController(
                        "test-model", page_images={42: image_path}
                    )
                    controller.decide(
                        "What is shown?",
                        [{
                            "context": "Table evidence",
                            "context_ids": ["p42-c1-abc"],
                        }],
                    )
        body = json.loads(call.call_args.args[0].data)
        content = body["messages"][-1]["content"]
        image_item = next(item for item in content if item["type"] == "image_url")
        self.assertIn("Visual scan of PDF page 42", content[1]["text"])
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(
            image_item["image_url"]["url"],
            "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"),
        )

    def test_api_controller_reports_truncated_model_output(self):
        payload = {
            "choices": [{
                "message": {"content": ""},
                "finish_reason": "length",
            }],
            "usage": {},
        }
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }):
            with patch(
                "day7_pipeline.urllib.request.urlopen",
                return_value=MockResponse(payload),
            ):
                controller = OpenAIController("test-model")
                with self.assertRaisesRegex(ValueError, "finish_reason=length"):
                    controller.decide("NASA?", [])

    def test_local_ollama_endpoint_needs_no_openai_key_and_accepts_vision(self):
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"Figure 3"}',
                        }
                    }]
                }
            }],
            "usage": {},
        }
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page-42.jpg"
            Image.new("RGB", (2048, 1536), color="white").save(
                image_path, format="JPEG"
            )
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "",
                "OPENAI_BASE_URL": "http://localhost:11434/v1/",
            }):
                with patch(
                    "day7_pipeline.urllib.request.urlopen",
                    return_value=MockResponse(payload),
                ) as call:
                    controller = OpenAIController(
                        "qwen3-vl:2b", page_images={42: image_path}
                    )
                    action = controller.decide(
                        "What does Figure 3 show?",
                        [{
                            "context": "Figure evidence",
                            "context_ids": ["p42-c1-abc"],
                        }],
                    )

        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(action, {"action": "search", "query": "Figure 3"})
        self.assertEqual(controller.max_image_pages, 1)
        body = json.loads(request.data)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["max_tokens"], 2048)
        self.assertIn("A search has already been performed", body["messages"][0]["content"])
        content = body["messages"][-1]["content"]
        image_item = next(item for item in content if item["type"] == "image_url")
        self.assertIsInstance(image_item["image_url"], str)
        self.assertTrue(image_item["image_url"].startswith("data:image/jpeg;base64,"))
        encoded_image = image_item["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded_image))) as image:
            self.assertEqual(image.size, (1024, 768))

    def test_remote_endpoint_still_requires_an_api_key(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "https://example.com/v1",
        }):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIController("test-model")

    def test_agent_test_split_requires_explicit_finalization_before_api_key(self):
        args = argparse.Namespace(
            split="test",
            finalize=False,
            max_searches=3,
            top_k=3,
            context_budget=4000,
            run_name=None,
            request_timeout=60,
            max_output_tokens=2048,
        )
        with self.assertRaisesRegex(ValueError, "--finalize"):
            _run_agent(args)

    def test_agent_rejects_nonpositive_model_request_timeout(self):
        args = argparse.Namespace(
            split="dev",
            finalize=False,
            max_searches=3,
            top_k=5,
            context_budget=4000,
            run_name=None,
            request_timeout=0,
            max_output_tokens=2048,
        )
        with self.assertRaisesRegex(ValueError, "request_timeout"):
            _run_agent(args)


if __name__ == "__main__":
    unittest.main()

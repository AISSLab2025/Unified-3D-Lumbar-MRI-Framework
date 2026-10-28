"""Offline wiring checks; all HTTP and LLM calls are blocked or mocked."""
import io
import json
import unittest
from unittest.mock import Mock, patch

from spine_swarm import api_client, web_app
from spine_swarm.agents import normalize_patient_data
from spine_swarm.langchain_agents import LangChainClinicalPanel


class IntegrationChecks(unittest.TestCase):
    def test_all_image_types_route_without_llm_or_cross_type_fallback(self):
        panel = LangChainClinicalPanel.__new__(LangChainClinicalPanel)
        panel.pipeline_data = {
            "output_dir": "/study/output", "output_url": "http://example/static",
            "sagittal": {"segmentation": {
                "original_image": {"url": "http://example/original.png"},
                "overlay_image": {"url": "http://example/seg.png"}},
                "measurements": {
                    "vertebrae_height": {"image_overlay": {"url": "http://example/height.png"}, "measurements": {}},
                    "lumbar_lordosis": {"image_overlay": {"url": "http://example/angle.png"}, "measurements": {"Angle": 40}}}},
            "axial": {"classification": {"foraminal": [{"level_name": "L1-L2", "region_name": "LFS",
                "predicted_class": "Stenosis", "gradcam_path": "/study/output/nonexistent.png",
                "url": "http://example/static/classification/axial_combined/L1-L2_merged.png"}]},
                "roi_detection": {"L1-L2": [{"region_name": "LFS", "overlay_image_path": {"path": "/study/output/ROIs_detection/axial/axial_estimation_debug/detection.png"}}]}}
        }
        cases = [
            ("Show axial foraminal classification at L1-L2", "/classification/axial_combined/"),
            ("Show axial foraminal detection at L1-L2", "/detection.png"),
            ("Show sagittal lordosis measurement", "/angle.png"),
            ("Show sagittal segmentation", "/seg.png"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                reply, _ = panel.run_chat_query(question)
                self.assertIn(expected, reply)
                self.assertEqual(reply.count("!["), 1)
                self.assertNotIn("nonexistent", reply)
                self.assertNotIn("original.png", reply)
        for question in ("Show axial classification at T1-T2", "Show axial segmentation at L1-L2",
                         "Show sagittal disc height measurements"):
            reply, _ = panel.run_chat_query(question)
            self.assertEqual(reply.count("!["), 0)
            self.assertIn("No matching", reply)

    def test_axial_sources_reject_other_folders(self):
        data = {"axial": {
            "classification": {"foraminal": [{"level_name": "L1-L2", "region_name": "LFS",
                "gradcam_path": "http://example/static/classification/axial/old.png"}]},
            "roi_detection": {"L1-L2": [{"region_name": "LFS",
                "overlay_image_path": "http://example/static/ROIs_detection/axial/merged.png"}]}}}
        normalized = normalize_patient_data(data)
        self.assertEqual(normalized["Axial_foraminal_classification"][0]["gradcam_path"], "")
        self.assertEqual(normalized["Axial_ROIs_estimation"]["L1-L2"][0]["overlay_image_path"], "")

    def test_highest_sagittal_grade_returns_one_image_and_notes_ties(self):
        panel = LangChainClinicalPanel.__new__(LangChainClinicalPanel)
        panel.pipeline_data = {"sagittal": {"classification": {
            "left": [
                {"level_name": "L1-L2", "predicted_class": "Normal", "gradcam_path": "http://example/normal.png"},
                {"level_name": "L4-L5", "predicted_class": "Severe_Stenosis", "gradcam_path": "http://example/left.png"}],
            "right": [{"level_name": "L4-L5", "predicted_class": "Severe_Stenosis", "gradcam_path": "http://example/right.png"}]
        }}}
        # No LLM is attached: this request must remain a JSON lookup.
        question = "Show sagittal Grad-CAM for the most severe foraminal stenosis and state severity grade."
        reply, _ = panel.run_chat_query(question)
        self.assertIn("http://example/left.png", reply)
        self.assertNotIn("http://example/right.png", reply)
        self.assertEqual(reply.count("!["), 1)
        self.assertNotIn("http://example/normal.png", reply)
        self.assertIn("same severity", reply)
        self.assertIn("Severe left foraminal stenosis at L4-L5.", reply)
        for technical_term in ("JSON", "Grad-CAM", "Severe_Stenosis", "model outputs"):
            self.assertNotIn(technical_term, reply)
        reply, _ = panel.run_chat_query(question + " Left side only.")
        self.assertNotIn("http://example/right.png", reply)
        reply, _ = panel.run_chat_query(question + " Right side only.")
        self.assertIn("http://example/right.png", reply)
        self.assertEqual(reply.count("!["), 1)
        panel.pipeline_data["sagittal"]["classification"]["left"][1].pop("gradcam_path")
        reply, _ = panel.run_chat_query(question)
        self.assertIn("http://example/right.png", reply)
        self.assertEqual(reply.count("!["), 1)
        panel.pipeline_data["sagittal"]["classification"]["right"][0].pop("gradcam_path")
        reply, _ = panel.run_chat_query(question)
        self.assertEqual(reply.count("!["), 0)
        self.assertNotIn("http://example/normal.png", reply)

    def test_foraminal_selection_excludes_central_and_other_levels(self):
        panel = LangChainClinicalPanel.__new__(LangChainClinicalPanel)
        panel.use_llm = False
        panel.llm = None
        catalog = [
            {"id": f"{level}-{region}", "title": f"Axial Grad-CAM ({level} - {region})",
             "view": "axial", "level": level, "kind": "gradcam", "url": "unused"}
            for level in ("L1-L2", "L2-L3") for region in ("SCS", "LFS", "RFS")
        ]
        question = "Show me the Grad-CAM image for foraminal stenosis at L1–L2 in the axial view."
        self.assertEqual(
            {e["id"] for e in panel._prefilter_catalog(question, catalog)},
            {"L1-L2-LFS", "L1-L2-RFS"},
        )
        self.assertEqual(panel.select_image(question + " Right side.", catalog)["id"], "L1-L2-RFS")
        self.assertIsNone(panel.select_image(question.replace("L1–L2", "L5-S1"), catalog))
        self.assertIsNone(panel.select_image(question, [catalog[0]]))

    def test_upload_contract(self):
        payload = {"status": "success", "sagittal": {"classification": {"left": [], "right": []}}}
        response = Mock()
        response.json.return_value = payload
        with patch.object(api_client.requests, "post", return_value=response) as post:
            self.assertEqual(api_client.predict_zip("contract.zip", b"placeholder"), payload)
        self.assertEqual(post.call_args.kwargs["files"]["file"][0], "contract.zip")
        response.raise_for_status.assert_called_once()

    def test_failure_and_empty_payload_rejected(self):
        for payload in ({"status": "failed"}, {}, [], {"status": "success", "axial": {}}):
            with self.assertRaises(ValueError):
                api_client.validate_payload(payload)

    def test_schema_adapter(self):
        payload = {"sagittal": {"classification": {"left": [], "right": []}},
                   "axial": {"classification": {"central_canal_stenosis": []}}}
        normalized = normalize_patient_data(payload)
        self.assertIn("Sagittal_ROI_classification", normalized)
        self.assertIn("Axial_central_classification", normalized)
        self.assertNotIn("Sagittal_ROI_classification", payload)

    def test_provider_failure_does_not_generate_findings(self):
        panel = LangChainClinicalPanel.__new__(LangChainClinicalPanel)
        panel.use_llm = True
        panel.llm = Mock()
        panel.llm.invoke.side_effect = RuntimeError("offline provider")
        with self.assertRaisesRegex(RuntimeError, "offline provider"):
            panel._call_specialist_llm("connection-test", "", "")

    def test_failed_upload_clears_context_without_panel_call(self):
        body = json.dumps({"filename": "empty.json", "filecontent": "{}"}).encode()
        handler = web_app.SpineSwarmHTTPHandler.__new__(web_app.SpineSwarmHTTPHandler)
        handler.path = "/api/upload"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        web_app.current_pipeline_data = {"previous": True}
        with patch.object(web_app, "LangChainClinicalPanel") as panel:
            handler.do_POST()
            panel.assert_not_called()
        self.assertEqual(web_app.current_pipeline_data, {})
        self.assertIn("error", json.loads(handler.wfile.getvalue()))


if __name__ == "__main__":
    unittest.main()

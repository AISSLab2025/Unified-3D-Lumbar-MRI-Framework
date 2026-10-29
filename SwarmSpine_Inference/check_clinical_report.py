"""Offline report regression checks. No MRI inference, HTTP or model generation."""
import unittest
from unittest.mock import Mock

from spine_swarm.agents import normalize_patient_data
from spine_swarm.clinical_report import SECTIONS, render_clinical_report
from spine_swarm.langchain_agents import LangChainClinicalPanel


def study():
    return normalize_patient_data({
        "sagittal": {
            "classification": {"left": [
                {"level_name": "L1-L2", "predicted_class": "Severe_Stenosis"},
            ]},
            "measurements": {
                "angles": {"measurements": {"LLA": 36.03, "LSA": 12.60}},
                "spinal_height": {"measurements": {"superior_hight": "179.68 mm"}},
                "lumbar_lordosis": {"measurements": "Normal lordosis: 0.5"},
                "vertebrae_deformities": {"measurements_info": {"L5": {
                    "Wedge Classification": "Normal", "Wedge Deformity": 17.98,
                    "Biconcave Classification": "Mild", "Biconcave Deformity": 20.72,
                }}},
                "spondylolisthesis": {"measurements": {"L5": {
                    "Classification": "Anterior Displacement", "Angle": "33.21",
                }}},
            },
        },
        "axial": {
            "classification": {"foraminal": [
                {"level_name": "L1-L2", "region_name": "RFS", "predicted_class": "Normal"},
            ]},
            "measurements": {"L3-L4_herniation": {
                "disc_herniation": False, "herniation_ratio": 0.938,
            }},
        },
    })


class ClinicalReportChecks(unittest.TestCase):
    def test_findings_are_one_paragraph_without_measurement_details(self):
        report = render_clinical_report(study())
        findings = report.split("\n\nFINDINGS\n", 1)[1].split("\n\nIVD-LEVEL ANALYSIS", 1)[0]
        impression = report.split("\n\nIMPRESSION\n", 1)[1].split("\n\nRECOMMENDATION", 1)[0]
        self.assertNotIn("\n", findings)
        self.assertNotIn("ratio", findings)
        self.assertNotIn("%", findings)
        self.assertNotIn("No disc herniation", findings)
        self.assertIn("severe left neural foraminal stenosis", findings)
        self.assertIn("Mild biconcave deformity", findings)
        self.assertNotEqual(findings, impression)

    def test_matching_abnormalities_are_grouped_across_levels(self):
        data = study()
        data["Sagittal_ROI_classification"]["left_classification"].append({
            "level_name": "L2-L3", "predicted_class": "Severe_Stenosis",
        })
        report = render_clinical_report(data)
        findings = report.split("\n\nFINDINGS\n", 1)[1].split("\n\nIVD-LEVEL ANALYSIS", 1)[0]
        self.assertIn("At L1-L2 and L2-L3, there is severe left neural foraminal stenosis.", findings)

    def test_fixed_structure_and_recorded_values(self):
        report = render_clinical_report(study())
        self.assertEqual([line for line in report.splitlines() if line in SECTIONS], list(SECTIONS))
        self.assertEqual(sum(line.startswith("• At the ") for line in report.splitlines()), 10)
        self.assertIn("36.03°", report)
        self.assertIn("179.68 mm", report)
        self.assertIn("20.72%", report)
        self.assertIn("Anterior displacement of L5", report)
        self.assertNotIn("33.21", report)  # angle is not converted to slip severity
        self.assertIn("No disc herniation is identified; herniation ratio: 0.938", report)
        self.assertNotIn("Grad-CAM", report)

    def test_laterality_and_missing_data(self):
        report = render_clinical_report(study())
        self.assertIn("Severe left neural foraminal stenosis is present", report)
        self.assertIn("No right neural foraminal stenosis is identified", report)
        self.assertIn("Central canal stenosis: not assessed", report)
        self.assertIn("Wedge deformity: not assessed", report)

    def test_conflicting_classifications_are_not_overwritten(self):
        data = study()
        data["Axial_foraminal_classification"].append({
            "level_name": "L1-L2", "region_name": "LFS", "predicted_class": "Normal",
        })
        report = render_clinical_report(data)
        self.assertIn("assessment is discordant (sagittal: severe stenosis; axial: normal)", report)

    def test_empty_study_is_not_a_normal_report(self):
        with self.assertRaisesRegex(ValueError, "No recognized study"):
            render_clinical_report({})

    def test_false_zero_and_unknown_are_distinct(self):
        data = study()
        data["axial"]["measurements"]["L3-L4_herniation"] = {"herniation_ratio": 0}
        report = render_clinical_report(data)
        self.assertIn("Disc herniation: not assessed; herniation ratio: 0", report)

    def test_panel_feedback_cannot_become_the_clinical_report(self):
        panel = LangChainClinicalPanel.__new__(LangChainClinicalPanel)
        panel.pipeline_data = study()
        panel._call_specialist_llm = Mock(return_value="Excellent response! Strengths: well written.")
        report, nodes = panel.run_consensus_report()
        self.assertTrue(report.startswith("TECHNIQUE\n"))
        self.assertNotIn("Excellent response", report)
        self.assertEqual(nodes[-1]["output"], report)
        self.assertEqual(panel._call_specialist_llm.call_count, 4)


if __name__ == "__main__":
    unittest.main()

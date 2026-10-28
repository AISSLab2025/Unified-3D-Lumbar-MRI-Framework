import json
import logging
import urllib.request
import urllib.error
from spine_swarm.config import DEFAULT_MODEL, OLLAMA_API_BASE, UPLOADED_JSON_PATH

logger = logging.getLogger("SpineSwarm.Agents")

# Prompts
ORTHOPEDIC_PROMPT = """You are the Virtual Orthopedic Specialist.
Your focus is the biomechanical and structural alignment of the lumbar spine.
You will receive vertebrae measurements, listhesis (slippage) findings, vertebrae deformities, and lumbar lordosis angles.
Your tasks:
1. Assess vertebrae heights and highlight any deformities or wedge fractures.
2. Analyze spondylolisthesis (e.g., L4 on L5 slippage) and evaluate structural stability.
3. Assess sagittal balance using lumbar lordosis angles.
Provide your diagnostic assessment and reasoning concisely. Focus on biomechanical stability."""

NEURORADIOLOGIST_PROMPT = """You are the Virtual Neuroradiologist.
Your focus is the intervertebral discs (IVD) and neural foramina.
You will receive IVD heights, centroids, and sagittal/axial foraminal stenosis classification grades.
Your tasks:
1. Analyze IVD heights for degenerative disc disease (DDD) or loss of height.
2. Evaluate foraminal stenosis (narrowing of the openings where nerve roots exit) at L1-S1.
3. Highlight any disc herniation/bulge and assess potential nerve root compression.
Provide your diagnostic assessment and reasoning concisely. Focus on neural impingement and disc pathology."""

NEUROSURGEON_PROMPT = """You are the Virtual Neurosurgeon.
Your focus is the spinal canal, lateral recesses, and surgical indications.
You will receive central canal stenosis grades and lateral recess stenosis classifications.
Your tasks:
1. Assess central canal stenosis severity and risk of neurogenic claudication or cauda equina syndrome.
2. Assess lateral recess stenosis and its correlation with nerve root compression.
3. Formulate potential surgical recommendations (e.g., decompression, laminectomy) if severe stenosis is present.
Provide your diagnostic assessment and reasoning concisely. Focus on spinal cord/canal compromise and surgical utility."""

SEARCH_PROMPT = """You are the Medical Literature & Guideline Search Agent.
Your role is to retrieve evidence-based clinical practice guidelines and recent literature for diagnosed spinal conditions.
You will receive the consensus clinical findings of the panel.
Your tasks:
1. Identify relevant clinical guidelines (e.g., NASS, ACR Appropriateness Criteria, AAOS).
2. Summarize evidence-based next steps (conservative management like physical therapy/epidural injections vs. surgical consultation).
3. Provide literature citations or study results (e.g., SPORT trial) supporting the recommended approach.
Provide a clean summary of guidelines and literature references."""

MODERATOR_PROMPT = """You are the Consensus Moderator and Chief Radiologist.
Your role is to lead the medical panel debate, resolve diagnostic conflicts, and draft the final consensus report.
Your tasks:
1. Coordinate inputs from the Orthopedic Specialist, Neuroradiologist, and Neurosurgeon.
2. Resolve any visual discordances. For example, if sagittal view foraminal classification shows 'Severe' but axial view shows 'Normal' at the same level, audit the findings, explain the discrepancy, and determine the consensus grade.
3. Correlate biomechanical findings with neurological compromise (e.g., 'Degenerative L4-L5 listhesis is the cause of secondary central stenosis').
4. Grade your overall diagnostic confidence and produce a highly detailed, structured, and professional clinical diagnostic report.

You MUST format the final consensus report exactly with the following headings and structure:
1. Executive Summary:
[Provide a summary of the patient findings, symptoms, listhesis, lordosis, deformities, and muscle spasms]

2. Level-Based Analysis
[Include the markdown comparison table, Vertebrae Analysis, and IVDs Analysis exactly like this structure]
| Lumbar Level | Central Canal Stenosis | Lateral Recess Stenosis | Foraminal Stenosis (Left/Right) | Clinical Context |
| :--- | :--- | :--- | :--- | :--- |
| **L1-L2** | [Grade] | [Grade] | [LeftGrade] / [RightGrade] | [Context] |
...
Vertebrae Analysis:
At the L1 level, a wedge deformity of [value] is present, classified as [class]. A biconcave deformity of [value] is present, classified as [class]. Spondylolisthesis at L1 is [class].
...
IVDs Analysis:
At the L1-L2 level, a herniation is [presence], with a condition of [cond] and a ratio of [ratio]. Central canal stenosis (CCS) is [grade], left foraminal stenosis (LFS) is [grade], and right foraminal stenosis (RFS) is [grade]. The foraminal distances at L1-L2 are: Left Foraminal distance = [value]mm, Right Foraminal distance = [value]mm. The AP Distance at L1-L2 is [value] mm.
...

3. Impression:
[Provide diagnostic impression summary of the degenerative changes, listhesis levels, foraminal stenosis levels, disc herniations, and lordosis]

4. Panel Assessments:
Diagnostic Confidence: [Confidence]% ([Short explanation]).
Orthopaedic Expert: Approved
Neuroradiology Expert: Approved
Neurosurgery Expert: Approved
Chief Radiologist: Approved
"""

ORCHESTRATOR_PROMPT = """You are the SpineSwarm Radiology Coordinator.
Your role is to execute the 13 lumbar spine diagnostic pipeline tools sequentially using function calling.
Your tasks:
1. Start by calling tool_01_identifier_dataset with the patient_path.
2. Follow the pipeline view identification, slice selection, ROI detection, segmentation, projection, and stenosis classification.
3. Call each tool from tool_01 to tool_13 sequentially, using the output information from previous tools as logical parameters.
4. Once all 13 tools have been executed, summarize the diagnostic data context you compiled and return it."""

MODERATOR_CHAT_PROMPT = """You are the Consensus Moderator and Chief Radiologist leading the SpineSwarm clinical panel.
Your role is to compile the specialist assessments and answer the user's question directly.
Rules:
1. If the user asks a simple question, answer directly in a very simple 1-2 sentence response based on the patient findings.
2. If the user asks for a report (e.g., comprehensive report, complete report, consensus report), generate the Consensus Diagnostic Report using exactly these headings, and nothing else:
   1. Executive Summary:
   2. Level-Based Analysis: (the markdown level table, then *Vertebrae Analysis:* and *IVDs Analysis:*)
   3. Impression:
   4. Panel Assessments: (Diagnostic Confidence: [N]%, then Biomechanical Specialist / Neural Compression Specialist / Surgical Specialist verdicts and "Consensus Moderator (Chief Radiologist): Final Consensus Achieved")
3. If the user asks for an image or scan (e.g. "show me the vertebrae height measurements image"), check the available images list and return the Markdown image link: ![Title](URL).
Frame your response clearly as the consensus of the SpineSwarm panel."""


# Keys of the {"path": ..., "url": ...} objects the updated prediction API returns
# in place of the plain path strings used by the legacy schema.
_IMAGE_REF_KEYS = {"path", "url", "relpath", "name", "filename"}


def _is_image_ref(val) -> bool:
    """True for the {"path": ..., "url": ...} objects of the updated API schema."""
    return (
        isinstance(val, dict)
        and bool(val)
        and ("path" in val or "url" in val)
        and set(val.keys()) <= _IMAGE_REF_KEYS
    )


def _local_path_to_url(path: str, data: dict = None) -> str:
    """Map a server-local output path onto the static URL the API serves it from."""
    if not isinstance(data, dict) or not isinstance(path, str):
        return ""
    output_dir = data.get("output_dir") or ""
    output_url = data.get("output_url") or ""
    if not isinstance(output_dir, str) or not isinstance(output_url, str):
        return ""
    if not output_dir or not output_url:
        return ""
    norm = path.replace("\\", "/")
    base = output_dir.replace("\\", "/").rstrip("/")
    if base and norm.startswith(base + "/"):
        return output_url.rstrip("/") + "/" + norm[len(base) + 1:]
    return ""


def extract_path_or_url(val, data: dict = None) -> str:
    """Collapse an image reference to a single browser-usable string.

    The prediction API returns image references either as a plain path string
    (legacy schema) or as a {"path": ..., "url": ...} object (updated schema).
    Always returning a string keeps these values hashable, which the image
    de-duplication sets rely on. Server-local paths are rewritten to the static
    URL when the payload carries the output_dir/output_url mapping.
    """
    raw = ""
    if isinstance(val, dict):
        raw = val.get("url") or val.get("path") or ""
    elif isinstance(val, str):
        raw = val
    if not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if not raw or raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return _local_path_to_url(raw, data) or raw


def image_from_directory(ref, data, directory):
    """Accept an explicit image reference only from the required directory."""
    import posixpath
    from urllib.parse import unquote, urlsplit
    value = extract_path_or_url(ref, data)
    path = posixpath.normpath(unquote(urlsplit(value.replace("\\", "/")).path))
    return value if posixpath.dirname(path).endswith("/" + directory) else ""


def flatten_image_refs(obj, root: dict = None):
    """Deep-copy the payload, collapsing every image reference object to a string."""
    if root is None and isinstance(obj, dict):
        root = obj
    if _is_image_ref(obj):
        return extract_path_or_url(obj, root)
    if isinstance(obj, dict):
        return {k: flatten_image_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [flatten_image_refs(v, root) for v in obj]
    return obj

def normalize_patient_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}
        
    # Collapse the updated schema's {"path": ..., "url": ...} image objects to plain
    # strings before anything else reads them. This works on a deep copy, so the
    # caller's payload is never mutated by the normalization below.
    data = flatten_image_refs(data)
    normalized = dict(data)
    
    sagittal = data.get("sagittal") or {}
    axial = data.get("axial") or {}
    
    # 1. Slice Selection
    if "Sagittal_slice_selection" not in normalized and "slice_selection" in sagittal:
        normalized["Sagittal_slice_selection"] = sagittal["slice_selection"]
        
    # 2. Sagittal ROI Detection
    if "Sagittal_ROI_detection" not in normalized and "roi_detection" in sagittal:
        roi_det = sagittal["roi_detection"]
        if isinstance(roi_det, dict):
            normalized["Sagittal_ROI_detection"] = {
                "left_sagittal_slice_rois": roi_det.get("left", []),
                "right_sagittal_slice_rois": roi_det.get("right", []),
                "mid_sagittal_slice": sagittal.get("slice_selection", {}).get("mid", {})
            }
            
    # 3. Sagittal ROI Classification
    if "Sagittal_ROI_classification" not in normalized and "classification" in sagittal:
        cls_det = sagittal["classification"]
        if isinstance(cls_det, dict):
            normalized["Sagittal_ROI_classification"] = {
                "left_classification": cls_det.get("left", []),
                "right_classification": cls_det.get("right", [])
            }
    elif "Sagittal_ROI_classification" in normalized:
        sag_cls = normalized["Sagittal_ROI_classification"]
        if isinstance(sag_cls, dict):
            if "left_classification" not in sag_cls and "left" in sag_cls:
                sag_cls["left_classification"] = sag_cls["left"]
            if "right_classification" not in sag_cls and "right" in sag_cls:
                sag_cls["right_classification"] = sag_cls["right"]
                
    # 4. Sagittal Segmentation
    if "Sagittal_segmentation" not in normalized and "segmentation" in sagittal:
        normalized["Sagittal_segmentation"] = sagittal["segmentation"]
    sag_seg = normalized.get("Sagittal_segmentation")
    if isinstance(sag_seg, dict) and not sag_seg.get("output_path"):
        # The updated schema renamed output_path to overlay_image.
        seg_overlay = sag_seg.get("overlay_image") or sag_seg.get("original_image")
        if seg_overlay:
            sag_seg = dict(sag_seg)
            sag_seg["output_path"] = seg_overlay
            normalized["Sagittal_segmentation"] = sag_seg
        
    # 5. Sagittal Measurements and Diagnosis
    if "Sagittal_measurements_and_diagnosis" not in normalized and "measurements" in sagittal:
        sag_meas = dict(sagittal["measurements"]) if isinstance(sagittal["measurements"], dict) else {}
        vert_def = sag_meas.get("vertebrae_deformities")
        if isinstance(vert_def, dict) and "measurements" not in vert_def and "measurements_info" in vert_def:
            vert_def["measurements"] = vert_def["measurements_info"]
        normalized["Sagittal_measurements_and_diagnosis"] = sag_meas
            
    # 6. Axial Classification
    ax_cls = axial.get("classification", {}) if isinstance(axial, dict) else {}
    if "Axial_foraminal_classification" not in normalized:
        normalized["Axial_foraminal_classification"] = ax_cls.get("foraminal_stenosis") or ax_cls.get("foraminal", [])
    if "Axial_lateral_classification" not in normalized:
        normalized["Axial_lateral_classification"] = ax_cls.get("lateral_recess_stenosis") or ax_cls.get("lateral", [])
    if "Axial_central_classification" not in normalized:
        normalized["Axial_central_classification"] = ax_cls.get("central_canal_stenosis") or ax_cls.get("central", [])

    # Older API responses have a nonexistent gradcam_path and the actual
    # combined Grad-CAM URL on the classification entry itself.
    for key in ("Axial_foraminal_classification", "Axial_lateral_classification", "Axial_central_classification"):
        for item in normalized.get(key) or []:
            if isinstance(item, dict):
                refs = (item.get("gradcam_image"), item.get("url"), item.get("gradcam_path"))
                item["gradcam_path"] = next((url for ref in refs if (url := image_from_directory(
                    ref, data, "classification/axial_combined"))), "")

    # 7. Axial ROI estimation (level -> list of ROI entries carrying the overlay image)
    if "Axial_ROIs_estimation" not in normalized and isinstance(axial, dict):
        ax_roi = axial.get("roi_detection") or axial.get("ROIs_estimation")
        if isinstance(ax_roi, dict) and ax_roi:
            normalized["Axial_ROIs_estimation"] = ax_roi

    for items in (normalized.get("Axial_ROIs_estimation") or {}).values():
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                item["overlay_image_path"] = image_from_directory(
                    item.get("overlay_image_path"), data,
                    "ROIs_detection/axial/axial_estimation_debug")

    return normalized


def get_patient_context_str(data: dict) -> str:
    if not data:
        return "No patient diagnostic data loaded."
        
    data = normalize_patient_data(data)
    summary = []
    
    # 1. Spondylolisthesis & Lordosis
    sag_meas = data.get("Sagittal_measurements_and_diagnosis") or {}
    spond = sag_meas.get("spondylolisthesis", {}).get("measurements", {})
    spond_list = []
    for level, details in spond.items():
        if isinstance(details, dict):
            cls = details.get("Classification", "Normal")
            if cls != "Normal":
                spond_list.append(f"{level}: {cls} displacement ({details.get('Angle', 0)} degrees)")
    
    lordosis = sag_meas.get("lumbar_lordosis", {}).get("measurements", {})
    lordosis_str = "Normal"
    if isinstance(lordosis, dict):
        lordosis_str = f"{lordosis.get('Classification', 'Normal')} (Angle: {lordosis.get('Angle', 0)} degrees)"
        
    summary.append("### Biomechanical & Alignment findings:")
    summary.append(f"- Spondylolisthesis: {', '.join(spond_list) if spond_list else 'Normal'}")
    summary.append(f"- Lumbar Lordosis: {lordosis_str}")
    
    # 2. Vertebrae Heights & Deformities
    vert_h = sag_meas.get("vertebrae_height", {}).get("measurements", {})
    vert_def = sag_meas.get("vertebrae_deformities", {}).get("measurements", {})
    vert_list = []
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        h_info = vert_h.get(level, {})
        def_info = vert_def.get(level, {})
        if h_info or def_info:
            h_str = ""
            if isinstance(h_info, dict):
                dist = h_info.get("distance", {})
                h_str = f"Left Height: {dist.get('dist_left_mm', 0)}mm, Right Height: {dist.get('dist_right_mm', 0)}mm"
                
            def_str = ""
            if isinstance(def_info, dict):
                wedge_ratio = def_info.get("Wedge Deformity", 0)
                wedge_cls = def_info.get("Wedge Classification", "Normal")
                biconcave_ratio = def_info.get("Biconcave Deformity", 0)
                biconcave_cls = def_info.get("Biconcave Classification", "Normal")
                def_str = f"Wedge: {wedge_ratio} ({wedge_cls}), Biconcave: {biconcave_ratio} ({biconcave_cls})"
                
            vert_list.append(f"- {level}: {h_str} | {def_str}")
            
    summary.append("\n### Vertebrae Analysis:")
    summary.extend(vert_list)
    
    # 3. Intervertebral Discs & Stenosis Classifications
    ivd_h = sag_meas.get("ivd_height", {}).get("measurements", {})
    
    # Let's get axial and sagittal classifications
    left_sag = data.get("Sagittal_ROI_classification", {}).get("left_classification", [])
    right_sag = data.get("Sagittal_ROI_classification", {}).get("right_classification", [])
    
    axial_foraminal = data.get("Axial_foraminal_classification", [])
    if not axial_foraminal:
        axial_foraminal = data.get("axial_classification", {}).get("foraminal_stenosis", [])
        
    axial_lateral = data.get("Axial_lateral_classification", [])
    if not axial_lateral:
        axial_lateral = data.get("axial_classification", {}).get("lateral_recess_stenosis", [])
        
    axial_central = data.get("Axial_central_classification", [])
    if not axial_central:
        axial_central = data.get("axial_classification", {}).get("central_canal_stenosis", [])
        
    summary.append("\n### Intervertebral Discs & Stenosis Analysis:")
    for level in ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]:
        h_info = ivd_h.get(level, {})
        h_str = ""
        if isinstance(h_info, dict):
            dist = h_info.get("distance", {})
            h_str = f"Left Height: {dist.get('dist_left_mm', 0)}mm, Right Height: {dist.get('dist_right_mm', 0)}mm"
            
        left_sag_cls = "Normal"
        for item in left_sag:
            if isinstance(item, dict) and item.get("level_name") == level:
                left_sag_cls = item.get("predicted_class", "Normal")
        right_sag_cls = "Normal"
        for item in right_sag:
            if isinstance(item, dict) and item.get("level_name") == level:
                right_sag_cls = item.get("predicted_class", "Normal")
                
        foraminal_cls = "Normal"
        for item in axial_foraminal:
            if isinstance(item, dict) and item.get("level_name") == level:
                foraminal_cls = item.get("predicted_class", "Normal")
                
        lateral_cls = "Normal"
        for item in axial_lateral:
            if isinstance(item, dict) and item.get("level_name") == level:
                lateral_cls = item.get("predicted_class", "Normal")
                
        central_cls = "Normal"
        for item in axial_central:
            if isinstance(item, dict) and item.get("level_name") == level:
                central_cls = item.get("predicted_class", "Normal")
                
        summary.append(f"- {level}: {h_str}")
        summary.append(f"  * Sagittal Foraminal Stenosis: Left: {left_sag_cls}, Right: {right_sag_cls}")
        summary.append(f"  * Axial Foraminal Stenosis: {foraminal_cls}")
        summary.append(f"  * Axial Lateral Recess Stenosis: {lateral_cls}")
        summary.append(f"  * Axial Central Canal Stenosis: {central_cls}")
        
    return "\n".join(summary)


def find_images_in_json(query_text: str, data: dict) -> list:
    if not data or not isinstance(data, dict):
        return []
    found = []
    text_lower = query_text.lower()
    
    is_sagittal = "sagittal" in text_lower or "sag" in text_lower
    is_axial = "axial" in text_lower or "ax" in text_lower
    
    # Side filter
    side_filter = None
    if "left" in text_lower:
        side_filter = "left"
    elif "right" in text_lower:
        side_filter = "right"
        
    # Level filter
    levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    matched_level = None
    for lvl in levels:
        if lvl.lower() in text_lower or lvl.replace("-", "").lower() in text_lower:
            matched_level = lvl
            break
            
    # Check patient-wide scans / overlays
    meas = data.get("Sagittal_measurements_and_diagnosis") or {}
    
    # 1. Segmentation
    if "segmentation" in text_lower or "mask" in text_lower or "segresnet" in text_lower:
        seg = data.get("Sagittal_segmentation") or {}
        p = seg.get("output_path") or seg.get("overlay_image")
        if p:
            found.append(("Mid-Sagittal Segmentation Overlay", p))
            
    # 2. Vertebrae Height
    if "vertebrae height" in text_lower or "vertebral height" in text_lower or "vertebra height" in text_lower:
        p = meas.get("vertebrae_height", {}).get("image_overlay")
        if p:
            found.append(("Sagittal Vertebrae Height Measurements", p))
            
    # 3. IVD Height
    if "ivd height" in text_lower or "disc height" in text_lower or "intervertebral disc height" in text_lower:
        # Check to make sure we don't accidentally match vertebrae height if that was explicitly requested instead
        if not ("vertebra" in text_lower and "height" in text_lower and not "disc" in text_lower and not "ivd" in text_lower):
            p = meas.get("ivd_height", {}).get("image_overlay")
            if p:
                found.append(("Sagittal Intervertebral Disc Height Measurements", p))
                
    # 4. Deformity
    if "deformity" in text_lower or "deformities" in text_lower or "wedge" in text_lower or "biconcave" in text_lower:
        p = meas.get("vertebrae_deformities", {}).get("image_overlay")
        if p:
            found.append(("Sagittal Vertebrae Deformities Analysis", p))
            
    # 5. Spondylolisthesis / alignment
    if "spondylolisthesis" in text_lower or "slippage" in text_lower or "displacement" in text_lower or "alignment" in text_lower:
        p = meas.get("spondylolisthesis", {}).get("image_overlay")
        if p:
            found.append(("Sagittal Spondylolisthesis Measurement", p))
            
    # 6. Lordosis
    if "lordosis" in text_lower or "angle" in text_lower:
        p = meas.get("lumbar_lordosis", {}).get("image_overlay") or meas.get("angles", {}).get("image_overlay")
        if p:
            found.append(("Sagittal Lumbar Lordosis / Alignment", p))
            
    # 7. Spinal Height
    if "spinal height" in text_lower:
        p = meas.get("spinal_height", {}).get("image_overlay")
        if p:
            found.append(("Sagittal Spinal Height Measurements", p))

    # 8. Level-specific scan / ROI / slice / Grad-CAM requests
    is_general_image_request = any(k in text_lower for k in ["image", "scan", "overlay", "roi", "view", "show", "gradcam", "grad-cam", "slice", "detection"])
    
    # If the user is asking for a specific level, or if it's a general image query
    if matched_level or is_general_image_request:
        # SAGITTAL SPECIFIC MATCHES
        # If user explicitly asked for sagittal, or didn't explicitly ask for axial
        if is_sagittal or not is_axial:
            # Check ROI detection / Slice images
            sag_roi_det = data.get("Sagittal_ROI_detection") or {}
            if isinstance(sag_roi_det, dict):
                # Search both sides, prioritizing the requested side if any
                sides_to_check = [("Left", "left_sagittal_slice_rois"), ("Right", "right_sagittal_slice_rois")]
                if side_filter == "left":
                    sides_to_check = [("Left", "left_sagittal_slice_rois")]
                elif side_filter == "right":
                    sides_to_check = [("Right", "right_sagittal_slice_rois")]
                    
                for side_name, side_key in sides_to_check:
                    lst = sag_roi_det.get(side_key) or []
                    for item in lst:
                        if isinstance(item, dict):
                            lvl = item.get("level_name")
                            if matched_level is None or lvl == matched_level:
                                p = item.get("overlay_image_path")
                                if p:
                                    found.append((f"Sagittal ROI Detection ({side_name} {lvl})", p))

            # Check Grad-CAM / classification images
            sag_roi_cls = data.get("Sagittal_ROI_classification") or {}
            if isinstance(sag_roi_cls, dict):
                sides_to_check = [("Left", "left_classification"), ("Right", "right_classification")]
                if side_filter == "left":
                    sides_to_check = [("Left", "left_classification")]
                elif side_filter == "right":
                    sides_to_check = [("Right", "right_classification")]
                    
                for side_name, side_key in sides_to_check:
                    lst = sag_roi_cls.get(side_key) or []
                    for item in lst:
                        if isinstance(item, dict):
                            lvl = item.get("level_name") or f"L{item.get('level', 1)}-L{item.get('level', 1)+1}"
                            if matched_level is None or lvl == matched_level:
                                p = item.get("gradcam_path")
                                if p:
                                    found.append((f"Sagittal Grad-CAM ({side_name} {lvl})", p))

        # AXIAL SPECIFIC MATCHES
        # If user explicitly asked for axial, or didn't explicitly ask for sagittal
        if is_axial or not is_sagittal:
            # Check Axial ROI estimation / Slice images
            axial_est = data.get("Axial_ROIs_estimation") or {}
            if isinstance(axial_est, dict):
                for lvl, items in axial_est.items():
                    if matched_level is None or lvl == matched_level:
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    p = item.get("overlay_image_path")
                                    if p:
                                        found.append((f"Axial ROI Estimation ({lvl})", p))
                                        break  # All slices for the same level have the same overlay path

            # Check Axial Grad-CAM / classification images
            for key in ["Axial_foraminal_classification", "Axial_lateral_classification", "Axial_central_classification"]:
                items = data.get(key) or []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            lvl = item.get("level_name")
                            if matched_level is None or lvl == matched_level:
                                p = item.get("gradcam_path")
                                if p:
                                    region = item.get("region_name", "Central")
                                    found.append((f"Axial Grad-CAM ({lvl} - {region})", p))

    # Apply strict view filters if explicitly requested in the query
    seen = set()
    unique_found = []
    for title, raw_path in found:
        # Guard against un-normalized payloads: image references may still be
        # {"path": ..., "url": ...} objects, which are unhashable.
        path = extract_path_or_url(raw_path, data)
        if path and path not in seen:
            seen.add(path)
            
            # Strict view keyword filter check
            title_lower = title.lower()
            path_lower = path.lower()
            
            if is_sagittal and not is_axial:
                # Keep only sagittal images
                if "sagittal" in title_lower or "/sagittal/" in path_lower or "left" in title_lower or "right" in title_lower or "left" in path_lower or "right" in path_lower:
                    unique_found.append((title, path))
            elif is_axial and not is_sagittal:
                # Keep only axial images
                if "axial" in title_lower or "/axial/" in path_lower:
                    unique_found.append((title, path))
            else:
                # Keep everything if no filter or both mentioned
                unique_found.append((title, path))
                
    # Fallback to general mid-sagittal segmentation or vertebrae height if nothing matched
    if not unique_found and ("image" in text_lower or "scan" in text_lower or "overlay" in text_lower):
        if is_axial:
            # If they asked for axial specifically, try returning the first axial ROI estimation
            axial_est = data.get("Axial_ROIs_estimation") or {}
            for lvl, items in axial_est.items():
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    p = extract_path_or_url(items[0].get("overlay_image_path"), data)
                    if p:
                        unique_found.append((f"Axial ROI Estimation ({lvl})", p))
                        break
        else:
            # Return mid-sagittal segmentation
            seg = data.get("Sagittal_segmentation") or {}
            p = extract_path_or_url(seg.get("output_path") or seg.get("overlay_image"), data)
            if p:
                unique_found.append(("Mid-Sagittal Segmentation Overlay", p))
                
    return unique_found



LUMBAR_LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def _level_from_index(item: dict) -> str:
    """Resolve a level name, falling back to the numeric level index."""
    lvl = item.get("level_name")
    if lvl:
        return lvl
    idx = item.get("level")
    if isinstance(idx, int) and 0 <= idx < len(LUMBAR_LEVELS):
        return LUMBAR_LEVELS[idx]
    return ""


def build_image_catalog(data: dict) -> list:
    """Enumerate every diagnostic image in the payload as structured entries.

    Returns dicts of {id, title, view, level, kind, url}. This is the ground truth
    an image request is answered from: a selector may only choose an entry that
    exists here, so it cannot invent a scan the pipeline never produced.
    """
    if not isinstance(data, dict):
        return []
    data = normalize_patient_data(data)
    catalog = []

    def add(entry_id, title, view, level, kind, raw, **metadata):
        url = extract_path_or_url(raw, data)
        if url:
            catalog.append({"id": entry_id, "title": title, "view": view,
                            "level": level or "", "kind": kind, "url": url, **metadata})

    # --- Sagittal segmentation -------------------------------------------------
    sag_seg = data.get("Sagittal_segmentation") or {}
    if isinstance(sag_seg, dict):
        add("sag-seg-overlay", "Mid-Sagittal Segmentation Overlay", "sagittal", "",
            "segmentation", sag_seg.get("output_path") or sag_seg.get("overlay_image"))
        add("sag-seg-original", "Mid-Sagittal Slice (unlabelled)", "sagittal", "",
            "original", sag_seg.get("original_image"))

    # --- Sagittal measurement overlays ----------------------------------------
    meas = data.get("Sagittal_measurements_and_diagnosis") or {}
    for key, label in [
        ("vertebrae_height", "Vertebrae Height Measurements"),
        ("ivd_height", "Intervertebral Disc Height Measurements"),
        ("vertebrae_deformities", "Vertebrae Deformities Analysis"),
        ("spondylolisthesis", "Spondylolisthesis Measurement"),
        ("lumbar_lordosis", "Lumbar Lordosis / Sagittal Alignment"),
        ("angles", "Lumbar Angle Measurements"),
        ("spinal_height", "Spinal Height Measurements"),
    ]:
        val = meas.get(key)
        if isinstance(val, dict):
            add(f"sag-meas-{key}", f"Sagittal {label}", "sagittal", "",
                "measurement", val.get("image_overlay"), details=val.get("measurements") or val.get("measurements_info") or {})

    # --- Sagittal ROI detection and Grad-CAM ----------------------------------
    sag_det = data.get("Sagittal_ROI_detection") or {}
    if isinstance(sag_det, dict):
        for side, side_key in [("Left", "left_sagittal_slice_rois"),
                               ("Right", "right_sagittal_slice_rois")]:
            for item in sag_det.get(side_key) or []:
                if isinstance(item, dict):
                    lvl = _level_from_index(item)
                    add(f"sag-roi-{side.lower()}-{lvl}",
                        f"Sagittal ROI Detection ({side} {lvl})".replace(" )", ")"),
                        "sagittal", lvl, "roi", item.get("overlay_image_path"))

    sag_cls = data.get("Sagittal_ROI_classification") or {}
    if isinstance(sag_cls, dict):
        for side, side_key in [("Left", "left_classification"),
                               ("Right", "right_classification")]:
            for item in sag_cls.get(side_key) or []:
                if isinstance(item, dict):
                    lvl = _level_from_index(item)
                    add(f"sag-gradcam-{side.lower()}-{lvl}",
                        f"Sagittal Grad-CAM ({side} {lvl})".replace(" )", ")"),
                        "sagittal", lvl, "gradcam", item.get("gradcam_path"), grade=item.get("predicted_class"))

    # --- Axial segmentation (per level) ---------------------------------------
    axial = data.get("axial") if isinstance(data.get("axial"), dict) else {}
    ax_seg = data.get("Axial_segmentation") or axial.get("segmentation") or {}
    if isinstance(ax_seg, dict):
        for lvl, val in ax_seg.items():
            if isinstance(val, dict):
                add(f"ax-seg-overlay-{lvl}", f"Axial Segmentation Overlay ({lvl})",
                    "axial", lvl, "segmentation", val.get("overlay_image"))
                add(f"ax-seg-original-{lvl}", f"Axial Slice, unlabelled ({lvl})",
                    "axial", lvl, "original", val.get("original_image"))

    # --- Axial ROI estimation --------------------------------------------------
    ax_roi = data.get("Axial_ROIs_estimation") or {}
    if isinstance(ax_roi, dict):
        for lvl, items in ax_roi.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("overlay_image_path"):
                        region = item.get("region_name", "")
                        add(f"ax-roi-{lvl}-{region}", f"Axial ROI Detection ({lvl} {region})",
                            "axial", lvl, "roi", item.get("overlay_image_path"))

    # --- Axial Grad-CAM --------------------------------------------------------
    seen_gradcam = set()
    for key, region_label in [
        ("Axial_foraminal_classification", "Foraminal"),
        ("Axial_lateral_classification", "Lateral Recess"),
        ("Axial_central_classification", "Central Canal"),
    ]:
        for item in data.get(key) or []:
            if isinstance(item, dict):
                lvl = _level_from_index(item)
                region = item.get("region_name") or region_label
                entry_id = f"ax-gradcam-{lvl}-{region}"
                if entry_id in seen_gradcam:
                    continue
                seen_gradcam.add(entry_id)
                add(entry_id, f"Axial Grad-CAM ({lvl} - {region})",
                    "axial", lvl, "gradcam", item.get("gradcam_path"), grade=item.get("predicted_class"))

    # --- Axial herniation measurement overlays --------------------------------
    ax_meas = data.get("Axial_measurements") or axial.get("measurements") or {}
    if isinstance(ax_meas, dict):
        for key, val in ax_meas.items():
            if isinstance(val, dict):
                lvl = val.get("level") or key.split("_")[0]
                add(f"ax-meas-{key}", f"Axial Disc Herniation Measurement ({lvl})",
                    "axial", lvl, "measurement", val.get("image_overlay_path"),
                    details={k: v for k, v in val.items() if k != "image_overlay_path"})

    # De-duplicate by entry id, not by URL: several levels legitimately share one
    # rendered slice, and collapsing those would make a level unaskable.
    unique, seen_ids = [], set()
    for e in catalog:
        if e["id"] not in seen_ids:
            seen_ids.add(e["id"])
            unique.append(e)
    return unique


def format_image_catalog(catalog: list) -> str:
    """Number the catalogue for an LLM to choose from."""
    lines = []
    for i, e in enumerate(catalog, start=1):
        bits = [f"view={e['view']}", f"kind={e['kind']}"]
        if e["level"]:
            bits.append(f"level={e['level']}")
        lines.append(f"[{i}] {e['title']}  ({', '.join(bits)})")
    return "\n".join(lines)


def get_image_links_context(data: dict) -> str:
    if not data:
        return "No images available."
    links = []
    
    seg = data.get("Sagittal_segmentation") or {}
    seg_path = extract_path_or_url(seg.get("output_path") or seg.get("overlay_image"), data)
    if seg_path:
        links.append(f"- Mid-Sagittal Segmentation Overlay: {seg_path}")
        
    meas = data.get("Sagittal_measurements_and_diagnosis", {})
    for key, name in [
        ("vertebrae_height", "Vertebrae Height Measurements"),
        ("ivd_height", "Intervertebral Disc Height Measurements"),
        ("spondylolisthesis", "Spondylolisthesis Measurement"),
        ("lumbar_lordosis", "Lumbar Lordosis / Sagittal Alignment"),
        ("angles", "Lumbar Lordosis / Sagittal Alignment"),
        ("vertebrae_deformities", "Vertebrae Deformities Analysis")
    ]:
        val = meas.get(key)
        if isinstance(val, dict):
            p = extract_path_or_url(val.get("image_overlay"), data)
            if p:
                links.append(f"- {name}: {p}")
                
    left_sag = data.get("Sagittal_ROI_classification", {}).get("left_classification", [])
    for item in left_sag:
        if isinstance(item, dict):
            gp = extract_path_or_url(item.get("gradcam_path"), data)
            lvl = item.get("level_name") or f"L{item.get('level', 1)}-L{item.get('level', 1)+1}"
            if gp:
                links.append(f"- Sagittal Grad-CAM (Left {lvl}): {gp}")
                
    right_sag = data.get("Sagittal_ROI_classification", {}).get("right_classification", [])
    for item in right_sag:
        if isinstance(item, dict):
            gp = extract_path_or_url(item.get("gradcam_path"), data)
            lvl = item.get("level_name") or f"L{item.get('level', 1)}-L{item.get('level', 1)+1}"
            if gp:
                links.append(f"- Sagittal Grad-CAM (Right {lvl}): {gp}")
                
    for key, name in [
        ("Axial_foraminal_classification", "Axial Foraminal"),
        ("Axial_lateral_classification", "Axial Lateral Recess"),
        ("Axial_central_classification", "Axial Central Canal")
    ]:
        items = data.get(key) or []
        for item in items:
            if isinstance(item, dict):
                gp = extract_path_or_url(item.get("gradcam_path"), data)
                lvl = item.get("level_name")
                reg = item.get("region_name", "")
                if gp:
                    suffix = f" - {reg}" if reg else ""
                    links.append(f"- Axial Grad-CAM ({lvl}{suffix}): {gp}")
                    
    return "\n".join(links)



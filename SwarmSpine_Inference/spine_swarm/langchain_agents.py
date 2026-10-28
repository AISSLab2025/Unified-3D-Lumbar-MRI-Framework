import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from spine_swarm.config import DEFAULT_MODEL, OLLAMA_API_BASE, UPLOADED_JSON_PATH
from spine_swarm.agents import (
    MODERATOR_CHAT_PROMPT, get_patient_context_str, find_images_in_json,
    normalize_patient_data, build_image_catalog, format_image_catalog
)

logger = logging.getLogger("SpineSwarm.LangChainAgents")

# Try importing LangChain libraries; provide custom fallbacks if imports fail
try:
    from langchain_openai import ChatOpenAI
except (ImportError, Exception):
    try:
        from langchain.chat_models import ChatOpenAI
    except (ImportError, Exception):
        ChatOpenAI = None

try:
    from langchain_core.tools import Tool
except (ImportError, Exception):
    try:
        from langchain.tools import Tool
    except (ImportError, Exception):
        try:
            from langchain.agents import Tool
        except (ImportError, Exception):
            Tool = None

try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
except (ImportError, Exception):
    try:
        from langchain_classic.agents import initialize_agent, AgentType
        from langchain_classic.memory import ConversationBufferMemory
    except (ImportError, Exception):
        initialize_agent = None
        AgentType = None
        ConversationBufferMemory = None

# Prompts
ORTHOPEDIC_PROMPT = """
You are a Virtual Biomechanical Specialist.
Your expertise focuses on the biomechanical integrity and structural alignment of the lumbar spine. You will receive vertebral body heights, vertebral deformity findings, spondylolisthesis measurements, vertebral translation data, and lumbar sagittal alignment parameters.

Your tasks are to:

1. Evaluate vertebral body heights from L1 to L5 and identify any height loss or abnormal morphology.
2. Detect vertebral deformities, including wedge deformities, compression fractures, or chronic remodeling changes.
3. Assess spondylolisthesis and vertebral translation at all lumbar motion segments (L1-L2 through L5-S1), including the severity and potential biomechanical significance.
4. Evaluate biomechanical stability and structural integrity based exclusively on the provided vertebral morphology, spondylolisthesis measurements, vertebral translation values, and sagittal alignment parameters.
5. Analyze lumbar sagittal alignment using lumbar lordosis measurements and determine whether the alignment is normal, hypolordotic, or hyperlordotic.
6. Correlate identified alignment abnormalities with their potential biomechanical consequences, such as altered load distribution, segmental instability, or compensatory postural changes.

Provide a concise diagnostic assessment supported by brief biomechanical reasoning. Base conclusions exclusively on the supplied imaging findings and measurements.

Focus on:
- Structural integrity of the vertebral column
- Segmental and global spinal stability
- Sagittal balance and lumbar alignment
- Biomechanical implications of the detected abnormalities

Do not provide treatment recommendations or speculate beyond the available imaging findings.
"""

NEURORADIOLOGIST_PROMPT = """You are a Virtual Neural Compression Specialist.
Your expertise focuses on the localization, characterization, and severity grading of lumbar spinal stenosis.
You will receive classification results for:
* Central Canal Stenosis (SCS)
* Lateral Recess Stenosis:
  - Right Lateral Recess Stenosis (RLS)
  - Left Lateral Recess Stenosis (LLS)
* Foraminal Stenosis assessments derived from both sagittal and axial imaging:
  - Sagittal foraminal stenosis is reported separately for the LEFT and RIGHT sides and includes a detection confidence score.
  - Axial foraminal stenosis is reported as:
    - RFS: Right Foraminal Stenosis
    - LFS: Left Foraminal Stenosis
Stenosis Severity Grading:
- Normal: No significant narrowing.
- Stenosis: Presence of stenosis.
- Severe_Stenosis: Marked narrowing with substantial neural compression.

Use ONLY the following severity labels in all outputs:
- Normal
- Stenosis
- Severe_Stenosis

Severity hierarchy:
Normal < Stenosis < Severe_Stenosis

Do not generate alternative terms such as mild, moderate, severe, minimal, critical, or advanced stenosis.
For foraminal stenosis evaluation:
1. If multiple sagittal detections exist for the same lumbar level and side, retain only the detection with the highest confidence score and discard lower-confidence duplicates.
2. After resolving duplicate sagittal detections, integrate sagittal and axial foraminal stenosis findings when both are available.
3. For each lumbar level and side (LEFT or RIGHT), compare the stenosis severity grades derived from sagittal and axial assessments.
4. If different severity grades are present for the same level and side, assign the higher severity grade according to:
   Normal < Stenosis < Severe_Stenosis.
5. If sagittal and axial assessments demonstrate identical severity grades, retain that grade as the final diagnosis.
6. If only one imaging plane is available, use the available classification as the final diagnosis.

Your tasks are to:
1. Map central canal stenosis (SCS) at each lumbar level from L1-L2 through L5-S1.
2. Map right and left lateral recess stenosis (RLS and LLS) at each lumbar level from L1-L2 through L5-S1.
3. Determine right and left foraminal stenosis at each lumbar level by:
   - Selecting the highest-confidence sagittal detection when duplicate detections exist.
   - Integrating sagittal and axial findings.
   - Assigning the highest severity grade when discrepancies exist between imaging planes.
4. Document the severity grade of every identified stenotic region.
5. Evaluate narrowing of the central canal, lateral recesses, and neural foramina.
6. Identify lumbar levels and sides where the observed stenosis patterns may correspond to potential neural compression based on anatomical location.
7. Generate a precise level-by-level summary of all stenotic findings.

Provide a concise diagnostic assessment supported by brief imaging-based reasoning.
Focus exclusively on:
- Stenosis localization
- Severity grading
- Neural pathway narrowing
- Potential neural compression

Do not provide treatment recommendations, surgical planning, prognosis, or management advice. Restrict conclusions to imaging-based neural compression assessment."""

NEUROSURGEON_PROMPT = """You are a Virtual Surgical Specialist.
Your focus is the clinical implications of lumbar spinal stenosis and potential surgical considerations.
You will receive compiled central canal, lateral recess, and foraminal stenosis findings.

Your tasks are to:
1. Assess the clinical significance of central canal stenosis.
2. Assess the clinical significance of lateral recess stenosis.
3. Assess the clinical significance of foraminal stenosis.
4. Identify imaging findings that may be associated with neurogenic claudication, radiculopathy, or severe neural compromise requiring urgent clinical correlation.
5. Determine whether the imaging findings may warrant surgical evaluation if correlated with compatible clinical symptoms and failure of conservative treatment.
6. Identify the specific lumbar levels most responsible for neural compromise.
7. Suggest potential surgical considerations based solely on imaging severity while emphasizing that definitive treatment decisions require clinical correlation.

Provide a concise diagnostic assessment with brief supporting reasoning.
Focus on:
- Clinical implications of imaging findings
- Potential neural compromise
- Surgical considerations based on imaging severity
- Pathomechanical consequences of stenosis

Do not establish definitive surgical candidacy. Treatment recommendations require integration of clinical symptoms, neurological examination findings, and patient-specific factors."""

SEARCH_PROMPT = """You are the Medical Literature and Guideline Search Agent.
Your role is to retrieve evidence-based clinical practice guidelines and relevant medical literature.
You will receive the consensus clinical findings of the SpineSwarm panel.

Your tasks are to:
1. Identify relevant clinical guidelines, including NASS, ACR Appropriateness Criteria, AAOS, and other specialty society recommendations.
2. Summarize evidence-based management pathways for the identified spinal conditions.
3. Compare conservative treatment options with surgical management recommendations.
4. Retrieve landmark studies, systematic reviews, meta-analyses, randomized controlled trials, and high-quality evidence supporting management decisions.
5. Provide literature citations and key study findings, including trials such as SPORT when applicable.
6. Summarize guideline-supported next steps for patient care.
7. Prioritize specialty society guidelines and high-level evidence when available.

Provide a clean summary of guidelines, evidence, and literature references."""

MODERATOR_PROMPT = """You are the Consensus Moderator and Chief Radiologist leading the SpineSwarm clinical panel.
Your role is to moderate the panel debate, resolve diagnostic conflicts, synthesize specialist opinions, and generate a comprehensive consensus report.

You must:
1. Call your specialists as tools to obtain their assessments.
2. Resolve diagnostic discordances (e.g., sagittal versus axial classification discrepancies).
3. Correlate biomechanical findings with neurological compromise (e.g., L4-L5 spondylolisthesis contributing to stenosis).
4. Assign an overall diagnostic confidence score (0–100%) based on inter-agent agreement, completeness of imaging data, and consistency of findings across imaging planes.
5. Produce a highly detailed, structured, and professional clinical diagnostic report.

CRITICAL: Output ONLY the final structured consensus report. Do NOT write any conversational preambles, introductory sentences, greetings, or meta-commentary. Start directly with the section '1. Executive Summary:'.

You MUST format the consensus report exactly using the following structure:

1. Executive Summary:
[Provide a summary of the major imaging findings, including stenosis severity, spondylolisthesis, lumbar lordosis, vertebral deformities, and disc pathology.]

2. Level-Based Analysis:
| Lumbar Level | Central Canal Stenosis | Lateral Recess Stenosis (Left/Right) | Foraminal Stenosis (Left/Right) | Clinical Context |
| :--- | :--- | :--- | :--- | :--- |
| *L1-L2* | [Grade] | [LeftGrade] / [RightGrade] | [LeftGrade] / [RightGrade] | [Context] |
| *L2-L3* | [Grade] | [LeftGrade] / [RightGrade] | [LeftGrade] / [RightGrade] | [Context] |
| *L3-L4* | [Grade] | [LeftGrade] / [RightGrade] | [LeftGrade] / [RightGrade] | [Context] |
| *L4-L5* | [Grade] | [LeftGrade] / [RightGrade] | [LeftGrade] / [RightGrade] | [Context] |
| *L5-S1* | [Grade] | [LeftGrade] / [RightGrade] | [LeftGrade] / [RightGrade] | [Context] |

Grade = Normal / Stenosis / Severe_Stenosis []

*Vertebrae Analysis:*
At the [Level] vertebral body, describe vertebral deformities, vertebral height abnormalities, and spondylolisthesis findings, including severity classifications when applicable.

*IVDs Analysis:*
At each lumbar level, summarize disc pathology, central canal stenosis grade, lateral recess stenosis grade, foraminal stenosis grade, foraminal distances, and AP canal distance measurements when available.

3. Impression:
[Provide a concise summary of the major degenerative findings, stenosis severity and distribution, spondylolisthesis levels, disc pathology, vertebral deformities, and sagittal alignment.]

4. Panel Assessments:
Diagnostic Confidence: [Confidence]% ([Short explanation])

Biomechanical Specialist: [Approved / Minor Disagreement / Major Disagreement]
Neural Compression Specialist: [Approved / Minor Disagreement / Major Disagreement]
Surgical Specialist: [Approved / Minor Disagreement / Major Disagreement]
Consensus Moderator (Chief Radiologist): Final Consensus Achieved
"""

# Phrases that mean "give me the full Consensus Diagnostic Report" rather than a
# conversational answer. These route the chat through the same structured panel
# pipeline that batch_process.py uses.
REPORT_REQUEST_PHRASES = (
    "consensus report", "consensus diagnostic report", "diagnostic report",
    "clinical report", "radiology report", "radiological report", "full report",
    "complete report", "comprehensive report", "detailed report", "final report",
    "structured report", "entire report", "whole report", "report of the findings",
)

REPORT_REQUEST_VERBS = (
    "generate", "write", "produce", "create", "make", "give", "show",
    "provide", "prepare", "compile", "draft", "need", "want",
)


# A request to *see* a scan, as opposed to a clinical question about it.
IMAGE_VERBS = (
    "show", "display", "view", "see", "give me", "let me see", "open",
    "pull up", "bring up", "render", "visualise", "visualize", "present",
)

IMAGE_NOUNS = (
    "image", "images", "scan", "scans", "picture", "photo", "overlay",
    "gradcam", "grad-cam", "grad cam", "heatmap", "roi", "segmentation",
    "mask", "figure", "snapshot", "visual", "slice", "view",
)

# Words that make it a clinical question, so the panel should answer in prose
CLINICAL_INTENT = (
    "severity", "severe", "grade", "grading", "why", "how", "explain",
    "cause", "reason", "interpret", "describe", "compare", "worst",
    "mean", "meaning", "significance", "diagnos", "treat", "risk",
    "symptom", "pain", "recommend", "should", "summar",
)


# Request words that carry no signal about which image is wanted
_SELECT_STOPWORDS = {
    "show", "display", "give", "let", "see", "view", "open", "pull", "bring",
    "render", "present", "please", "can", "you", "the", "and", "for", "from",
    "with", "this", "that", "patient", "image", "images", "picture", "photo",
    "scan", "scans", "want", "need", "would", "like", "about", "what", "which",
    "get", "look", "there", "any", "some", "his", "her", "their",
}


# Clinical shorthand the catalogue titles spell out in full
_TERM_ALIASES = {
    "ivd": ("intervertebral", "disc"),
    "discs": ("intervertebral", "disc"),
    "gradcam": ("grad-cam",),
    "cam": ("grad-cam",),
    "heatmap": ("grad-cam",),
    "listhesis": ("spondylolisthesis",),
    "slippage": ("spondylolisthesis",),
    "deformity": ("deformities",),
    "vertebral": ("vertebrae",),
    "anatomical": ("segmentation",),
    "mask": ("segmentation",),
    "alignment": ("lordosis", "alignment"),
    "herniation": ("herniation",),
}


def _has_term(lowered: str, terms) -> bool:
    """Match each term at a word boundary, allowing suffixes ("diagnos" -> diagnostic).

    Plain substring matching is wrong here: "show" contains "how", which would
    misread every "show me the scan" as a clinical question.
    """
    return any(re.search(r"\b" + re.escape(t), lowered) for t in terms)


def is_image_only_request(text: str) -> bool:
    """True when the user only wants to look at a scan, with no clinical question.

    Such a request is answered with the image itself instead of convening the
    whole panel for prose the user did not ask for.
    """
    if not text:
        return False
    lowered = " ".join(str(text).lower().split())
    if is_report_request(lowered):
        return False
    if _has_term(lowered, CLINICAL_INTENT):
        return False
    return _has_term(lowered, IMAGE_VERBS) and _has_term(lowered, IMAGE_NOUNS)


def is_report_request(text: str) -> bool:
    """True when the user is asking for the full Consensus Diagnostic Report."""
    if not text:
        return False
    lowered = " ".join(str(text).lower().split())
    if any(phrase in lowered for phrase in REPORT_REQUEST_PHRASES):
        return True
    # "report" as a standalone word (not "reported"/"reporting") plus a request verb
    if re.search(r"\breports?\b", lowered):
        return any(verb in lowered for verb in REPORT_REQUEST_VERBS)
    return False


def get_available_patient_scans_summary(data: dict) -> str:
    if not data:
        return "No patient diagnostic data loaded."
        
    sag_levels = set()
    
    # Check Sagittal classification
    sag_roi = data.get("Sagittal_ROI_classification") or {}
    if isinstance(sag_roi, dict):
        for side in ["left_classification", "right_classification"]:
            for item in sag_roi.get(side) or []:
                if isinstance(item, dict):
                    lvl = item.get("level_name")
                    if not lvl:
                        level_num = item.get("level")
                        if level_num is not None:
                            lvl = f"L{level_num+1}-L{level_num+2}"
                    if lvl:
                        sag_levels.add(lvl)
                        
    # Check Sagittal detection
    sag_roi_det = data.get("Sagittal_ROI_detection") or {}
    if isinstance(sag_roi_det, dict):
        for side in ["left_sagittal_slice_rois", "right_sagittal_slice_rois"]:
            for item in sag_roi_det.get(side) or []:
                if isinstance(item, dict):
                    lvl = item.get("level_name")
                    if lvl:
                        sag_levels.add(lvl)
                        
    axial_levels = set()
    
    # Check Axial classification
    for key in ["Axial_foraminal_classification", "Axial_lateral_classification", "Axial_central_classification"]:
        items = data.get(key) or []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    lvl = item.get("level_name")
                    if lvl:
                        axial_levels.add(lvl)
                        
    # Check Axial estimation
    axial_est = data.get("Axial_ROIs_estimation") or {}
    if isinstance(axial_est, dict):
        for lvl in axial_est.keys():
            axial_levels.add(lvl)
            
    # Check patient-wide scans
    meas = data.get("Sagittal_measurements_and_diagnosis") or {}
    patient_wide = []
    if isinstance(meas, dict):
        if "vertebrae_height" in meas:
            patient_wide.append("vertebrae height measurements")
        if "ivd_height" in meas:
            patient_wide.append("intervertebral disc height measurements")
        if "vertebrae_deformities" in meas:
            patient_wide.append("vertebrae deformities analysis")
        if "spondylolisthesis" in meas:
            patient_wide.append("spondylolisthesis measurements")
        if "lumbar_lordosis" in meas or "angles" in meas:
            patient_wide.append("lumbar lordosis / angles")
    if "Sagittal_segmentation" in data:
        patient_wide.append("mid-sagittal segmentation")
        
    sag_list = sorted(list(sag_levels))
    axial_list = sorted(list(axial_levels))
    
    summary = []
    summary.append(f"- Sagittal levels with scan/ROI/Grad-CAM images: {', '.join(sag_list) if sag_list else 'None'}")
    summary.append(f"- Axial levels with scan/ROI/Grad-CAM images: {', '.join(axial_list) if axial_list else 'None'}")
    summary.append(f"- Patient-wide information: {', '.join(patient_wide) if patient_wide else 'None'}")
    
    return "\n".join(summary)


def format_chat_history(chat_history: list) -> str:
    if not chat_history:
        return "No previous conversation history."
    formatted = []
    # Limit to the last 10 turns to avoid token limits
    for msg in chat_history[-10:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


INTAKE_AGENT_PROMPT = """You are the SpineSwarm Intake and Clarifier Agent.
Your role is to pre-screen the user's question against the patient's diagnostic data and determine if it is clear or requires clarification.

Here is the exact diagnostic scan data available for this patient:
{patient_scans_summary}

Previous Conversation History:
{chat_history_str}

Guidelines:
1. Identify if the user's current question is level-specific or asking to see a level-specific scan (e.g. "What is the stenosis?", "Is there narrowing?", "Show me the ROI image", "Show me the Grad-CAM", "Show me the scan", "What is the displacement?") but has NOT specified a lumbar level (e.g., L1-L2, L2-L3, L3-L4, L4-L5, L5-S1) in either their current question or the immediate previous assistant/user turns.
   - If a level is NOT specified anywhere in the current context/history, you MUST ask a direct clarifying question. For example: "Which lumbar level would you like to see the sagittal ROI detection image for?" or "Which level are you referring to for your question?"
   - If the level is specified in the history or current question, that level is resolved. Check if it exists for the requested view (sagittal or axial) in the available patient data above. If it exists, or if the user's question is clear and has no ambiguity, reply with exactly "[CLEAR]" and nothing else.
2. If the user asks for a scan or image of a specific level, but that view/level does NOT exist in the patient's available scans listed above, you should politely clarify that only certain levels are available for that view, and list the available levels.
3. If the user asks a general question that does not require level clarification (e.g., "Write a comprehensive report", "What is the overall diagnostic confidence?", "What symptoms does the patient have?", "Why is this happening?"), reply with exactly "[CLEAR]" and nothing else.
4. Never suggest or ask for levels that do not exist in the patient's available data.
5. If the user's query is clear, you MUST respond with EXACTLY "[CLEAR]" and nothing else. Do not output explanations or extra text.

User's Current Question: "{user_question}"
Response:"""



def init_ollama_llm(model_name: str, base_url: str):
    # Try importing ChatOllama
    try:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.2)
    except ImportError:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name, base_url=base_url, temperature=0.2)
        except ImportError:
            pass
            
    # Fall back to ChatOpenAI with Ollama's OpenAI-compatible endpoint
    if ChatOpenAI is not None:
        # Try base_url first
        try:
            return ChatOpenAI(
                openai_api_key="ollama",
                base_url=f"{base_url}/v1",
                model_name=model_name,
                temperature=0.2
            )
        except Exception:
            try:
                return ChatOpenAI(
                    openai_api_key="ollama",
                    openai_api_base=f"{base_url}/v1",
                    model_name=model_name,
                    temperature=0.2
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChatOpenAI for Ollama: {e}")
                
    return None


def init_openai_llm(model_name: str, api_key: str = None):
    if ChatOpenAI is not None:
        try:
            import os
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is not set. Please provide your OpenAI API Key in the configuration panel.")
            return ChatOpenAI(
                openai_api_key=key,
                model_name=model_name,
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise e
    return None


class LangChainClinicalPanel:
    def __init__(self, pipeline_data: dict = None, api_key: str = None, model_name: str = None):
        self.pipeline_data = normalize_patient_data(pipeline_data or {})
        # Keep api_key argument for backward compatibility
        self.api_key = api_key or ""
        
        self.use_llm = False
        model = model_name if model_name else (DEFAULT_MODEL or "gemma3:12b")
        
        if model.startswith("gpt-") or "chatgpt" in model.lower():
            self.llm = init_openai_llm(model, api_key=self.api_key)
            if self.llm is not None:
                self.use_llm = True
                logger.info(f"Initialized OpenAI LLM with model: {model}")
        else:
            self.llm = init_ollama_llm(model, OLLAMA_API_BASE)
            if self.llm is not None:
                self.use_llm = True
                logger.info(f"Initialized Ollama LLM with model: {model}")
            else:
                logger.info("LLM provider initialization failed.")

        if not self.use_llm:
            raise RuntimeError("LLM provider unavailable. Install requirements and configure your model.")
        self.debate_history: List[Dict[str, str]] = []

    def run_intake_agent(self, user_question: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Analyze user question against patient data to ensure clarity; counter-question if ambiguous."""
        chat_history = chat_history or []
        if not self.use_llm or not self.llm:
            return self._rule_based_intake(user_question, chat_history)
            
        try:
            scans_summary = get_available_patient_scans_summary(self.pipeline_data)
            chat_history_str = format_chat_history(chat_history)
            intake_prompt = INTAKE_AGENT_PROMPT.format(
                patient_scans_summary=scans_summary,
                chat_history_str=chat_history_str,
                user_question=user_question
            )
            response = self.llm.invoke(intake_prompt)
            res_text = response.content.strip()
            if not res_text:
                return self._rule_based_intake(user_question, chat_history)
            return res_text
        except Exception as e:
            logger.error(f"Error calling Intake LLM: {e}")
            raise e

    def _rule_based_intake(self, user_question: str, chat_history: List[Dict[str, str]] = None) -> str:
        chat_history = chat_history or []
        q_lower = user_question.lower()
        history_lower = " ".join([m.get("content", "").lower() for m in chat_history])
        combined_text = history_lower + " " + q_lower
        
        # Extract available levels and views from data dynamically
        data = self.pipeline_data or {}
        
        sag_levels = set()
        left_sag = data.get("Sagittal_ROI_classification", {}).get("left_classification", [])
        right_sag = data.get("Sagittal_ROI_classification", {}).get("right_classification", [])
        for item in left_sag + right_sag:
            if isinstance(item, dict):
                lvl = item.get("level_name") or f"L{item.get('level', 1)}-L{item.get('level', 1)+1}"
                sag_levels.add(lvl.lower())
        sag_roi_det = data.get("Sagittal_ROI_detection") or {}
        if isinstance(sag_roi_det, dict):
            for side in ["left_sagittal_slice_rois", "right_sagittal_slice_rois"]:
                for item in sag_roi_det.get(side) or []:
                    if isinstance(item, dict):
                        lvl = item.get("level_name")
                        if lvl:
                            sag_levels.add(lvl.lower())

        axial_levels = set()
        for key in ["Axial_foraminal_classification", "Axial_lateral_classification", "Axial_central_classification"]:
            items = data.get(key) or []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        lvl = item.get("level_name")
                        if lvl:
                            axial_levels.add(lvl.lower())
        axial_est = data.get("Axial_ROIs_estimation") or {}
        if isinstance(axial_est, dict):
            for lvl in axial_est.keys():
                axial_levels.add(lvl.lower())
                
        # Check if user specified any level in current question or context history
        specified_level = None
        for lvl in ["l1-l2", "l2-l3", "l3-l4", "l4-l5", "l5-s1", "l1", "l2", "l3", "l4", "l5", "s1"]:
            if lvl in q_lower or lvl in history_lower:
                if "-" in lvl:
                    specified_level = lvl
                else:
                    for full_lvl in ["l1-l2", "l2-l3", "l3-l4", "l4-l5", "l5-s1"]:
                        if lvl in full_lvl:
                            specified_level = full_lvl
                            break
                break

        # Check if the user is asking for an image / scan
        is_image_request = any(k in q_lower for k in ["image", "scan", "overlay", "roi", "view", "show", "gradcam", "grad-cam"])
        
        # Check if patient-wide scans are requested
        is_patient_wide = any(k in q_lower for k in [
            "vertebrae height", "vertebral height", "vertebra height",
            "ivd height", "disc height", "deformity", "deformities",
            "spondylolisthesis", "lordosis", "segmentation", "mask"
        ])
        
        # Check for other clinical parameters in the JSON that require level/region clarification
        is_clinical_query = any(k in q_lower for k in [
            "stenosis", "narrowing", "displacement", "herniation", "bulge",
            "compression", "impingement", "ap distance", "distance",
            "wedge", "biconcave"
        ])
        
        if (is_image_request or is_clinical_query) and not specified_level and not is_patient_wide:
            # Clarify level-specific clinical query
            if "sagittal" in q_lower and "roi" in q_lower:
                available_str = ", ".join(sorted(list(sag_levels))).upper()
                return f"Which lumbar level ({available_str or 'L3-L4, L4-L5, or L5-S1'}) would you like to see the sagittal ROI detection image for?"
            elif "axial" in q_lower and "roi" in q_lower:
                available_str = ", ".join(sorted(list(axial_levels))).upper()
                return f"Which lumbar level ({available_str or 'L1-L2, L2-L3, L3-L4, L4-L5, or L5-S1'}) would you like to see the axial ROI detection image for?"
            elif "gradcam" in q_lower or "grad-cam" in q_lower:
                return "Which level (L1-L2 to L5-S1) and view (sagittal or axial) would you like to see the Grad-CAM scan for?"
            elif "stenosis" in q_lower or "narrowing" in q_lower:
                return "Could you please specify which lumbar level (L1-L2 to L5-S1) or which parameter (central, lateral, or foraminal stenosis) you want to know about?"
            elif "height" in q_lower:
                return "Are you asking about the vertebrae height or the intervertebral disc height? Please clarify."
            else:
                return "Which level (L1-L2, L2-L3, L3-L4, L4-L5, or L5-S1) are you referring to for your question?"
                
        # If user specified a level, but it doesn't match available data for that view
        if specified_level:
            if "sagittal" in q_lower and specified_level not in sag_levels:
                available_str = ", ".join(sorted(list(sag_levels))).upper()
                return f"The sagittal scan is not available at level {specified_level.upper()}. Available sagittal levels: {available_str}."
            if "axial" in q_lower and specified_level not in axial_levels:
                available_str = ", ".join(sorted(list(axial_levels))).upper()
                return f"The axial scan is not available at level {specified_level.upper()}. Available axial levels: {available_str}."
                
        return "[CLEAR]"


    def _call_specialist_llm(self, agent_name: str, system_prompt: str, input_context: str) -> str:
        """Helper to invoke a specific specialist agent LLM."""
        if not self.use_llm or not self.llm:
            return self._generate_specialist_fallback(agent_name, input_context)
        
        try:
            response = self.llm.invoke(f"System: {system_prompt}\n\nUser: {input_context}")
            if response and hasattr(response, "content") and response.content:
                return response.content
            raise RuntimeError("The model returned an empty response.")
        except Exception as e:
            logger.warning(f"Error calling LLM for {agent_name}: {e}. No substitute findings will be generated.")
            raise RuntimeError(
                f"{agent_name}: LLM call failed ({type(e).__name__}): {e}. "
                "Check the model service and selected model. No substitute findings were generated."
            ) from e

    def _generate_specialist_fallback(self, agent_name: str, input_context: str) -> str:
        raise RuntimeError(f"{agent_name}: LLM unavailable or returned no response. No substitute findings were generated.")

    def run_debate(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Full panel debate producing the structured Consensus Diagnostic Report."""
        return self.run_consensus_report()

    def run_consensus_report(
        self,
        user_question: str = None,
        chat_history: List[Dict[str, str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Convene the panel and compile the structured Consensus Diagnostic Report.

        This is the single source of the report structure: the batch pipeline calls
        it through run_debate(), and the chatbot calls it directly when the user asks
        for a report, so both produce identically structured output.
        """
        self.debate_history = []
        
        # Extract patient data details for the prompts
        sag_meas = self.pipeline_data.get("Sagittal_measurements_and_diagnosis") or {}
        sag_roi = self.pipeline_data.get("Sagittal_ROI_classification") or {}
        axial_foraminal = self.pipeline_data.get("Axial_foraminal_classification") or []
        axial_lateral = self.pipeline_data.get("Axial_lateral_classification") or []
        axial_central = self.pipeline_data.get("Axial_central_classification") or []
        patient_context = get_patient_context_str(self.pipeline_data)
        
        # 1. Orthopedic Specialist
        ortho_context = f"""Review the sagittal vertebrae measurements and alignment:
- Spondylolisthesis measurements: {json.dumps(sag_meas.get('spondylolisthesis', {}), indent=2)}
- Vertebrae heights: {json.dumps(sag_meas.get('vertebrae_height', {}), indent=2)}
- Vertebrae deformities: {json.dumps(sag_meas.get('vertebrae_deformities', {}), indent=2)}
- Lumbar lordosis angle: {json.dumps(sag_meas.get('lumbar_lordosis', {}), indent=2)}"""
        ortho_findings = self._call_specialist_llm("Orthopedic Specialist", ORTHOPEDIC_PROMPT, ortho_context)
        self.debate_history.append({
            "id": "orthopedic",
            "name": "Orthopedic Specialist Agent",
            "role": "Spine Alignment & Biomechanics",
            "input": ortho_context,
            "output": ortho_findings
        })
        
        # 2. Neuroradiologist
        neurorad_context = f"""Review intervertebral discs and neural foramina:
- IVD heights: {json.dumps(sag_meas.get('ivd_height', {}), indent=2)}
- Sagittal left foraminal stenosis classification: {json.dumps(sag_roi.get('left_classification', []), indent=2)}
- Sagittal right foraminal stenosis classification: {json.dumps(sag_roi.get('right_classification', []), indent=2)}
- Axial foraminal stenosis classification: {json.dumps(axial_foraminal, indent=2)}"""
        neurorad_findings = self._call_specialist_llm("Neuroradiology Agent", NEURORADIOLOGIST_PROMPT, neurorad_context)
        self.debate_history.append({
            "id": "neuroradiologist",
            "name": "Neuroradiology Agent",
            "role": "Intervertebral Discs & Neural Foramina",
            "input": neurorad_context,
            "output": neurorad_findings
        })
        
        # 3. Neurosurgeon
        neuro_context = f"""Review spinal canal and lateral recesses:
- Central canal stenosis classification: {json.dumps(axial_central, indent=2)}
- Lateral recess stenosis classification: {json.dumps(axial_lateral, indent=2)}"""
        neuro_findings = self._call_specialist_llm("Neurosurgeon Agent", NEUROSURGEON_PROMPT, neuro_context)
        self.debate_history.append({
            "id": "neurosurgeon",
            "name": "Neurosurgeon Agent",
            "role": "Spinal Canal & Lateral Recesses",
            "input": neuro_context,
            "output": neuro_findings
        })
        
        # 4. Search / Literature Agent
        search_context = f"""Search clinical guidelines and clinical trials (e.g. SPORT, NASS) for the patient's diagnosed findings:
{patient_context}"""
        search_findings = self._call_specialist_llm("Online Search / Literature Agent", SEARCH_PROMPT, search_context)
        self.debate_history.append({
            "id": "search",
            "name": "Online Search / Literature Agent",
            "role": "Medical Guidelines & Clinical Studies",
            "input": search_context,
            "output": search_findings
        })
        
        # 5. Consensus Moderator compiles the findings
        moderator_context = f"""Moderate the panel and write the final consensus report.
1. Orthopedic Specialist Assessment:
{ortho_findings}
 
2. Neuroradiology Assessment:
{neurorad_findings}
 
3. Neurosurgeon Assessment:
{neuro_findings}
 
4. Medical Guidelines & Literature:
{search_findings}"""

        if user_question:
            moderator_context += f'''
 
5. Requesting Clinician's Question:
"{user_question}"
Address this request within the report, but do NOT change the required report structure.'''
            history_str = format_chat_history(chat_history or [])
            if history_str and history_str.strip():
                moderator_context += f'''
 
6. Conversation History:
{history_str}'''
        
        moderator_report = self._call_specialist_llm("Consensus Moderator", MODERATOR_PROMPT, moderator_context)
        self.debate_history.append({
            "id": "moderator",
            "name": "Consensus Moderator / Chief Radiologist",
            "role": "Panel Debate & Consensus Report",
            "input": moderator_context,
            "output": moderator_report
        })
        
        return moderator_report, self.debate_history

    def _prefilter_catalog(self, question: str, catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Narrow the catalogue by the view, level and kind stated in the question.

        Filtering against the JSON before involving the LLM keeps the choice
        anchored to what exists, and keeps the prompt small enough for a small
        local model to answer reliably.
        """
        q = " ".join(question.lower().split())
        q = re.sub(r"[–—−]", "-", q)
        q = re.sub(r"\s*-\s*", "-", q)
        candidates = list(catalog)

        wants_axial = bool(re.search(r"\baxial|\btransverse", q))
        wants_sagittal = bool(re.search(r"\bsagittal|\bsag\b", q))
        if wants_axial and not wants_sagittal:
            candidates = [e for e in candidates if e["view"] == "axial"]
        elif wants_sagittal and not wants_axial:
            candidates = [e for e in candidates if e["view"] == "sagittal"]

        level = None
        level_match = re.search(r"\b([ctl]\d+)-?([ctls]\d+)\b", q)
        if level_match:
            level = "-".join(level_match.groups()).upper()
        if level:
            scoped = [e for e in candidates if e["level"] == level or (
                e["kind"] == "measurement" and level in e.get("details", {}))]
            candidates = scoped

        from spine_swarm.image_requests import requested_image_kind
        kind = requested_image_kind(q)
        if kind:
            scoped = [e for e in candidates if e["kind"] == kind]
            candidates = scoped

        if kind == "measurement":
            measurement_id = None
            for pattern, suffix in (
                (r"lordosis", "lumbar_lordosis"),
                (r"spondylolisthesis|listhesis|slippage", "spondylolisthesis"),
                (r"deform|wedge|biconcave", "vertebrae_deformities"),
                (r"(?:ivd|disc|intervertebral).*height", "ivd_height"),
                (r"vertebr.*height", "vertebrae_height"),
                (r"spinal.*height", "spinal_height"),
                (r"angle|\blla\b|\blsa\b", "angles"),
            ):
                if re.search(pattern, q):
                    measurement_id = "sag-meas-" + suffix
                    break
            if measurement_id:
                candidates = [e for e in candidates if e["id"] == measurement_id]
            elif "herniation" in q:
                candidates = [e for e in candidates if e["id"].startswith("ax-meas-")]
            elif re.search(r"foramin|canal|\bap distance\b", q):
                # Do not substitute a generic height/herniation measurement
                # when this API has no image for the requested distance.
                candidates = [e for e in candidates if re.search(
                    r"foramin|canal|\bap distance\b", e["title"].lower())]

        # Disease/region and side are constraints, not just ranking hints.
        # Otherwise an LLM can select SCS for a foraminal request.
        regions = []
        if re.search(r"\bforamin\w*|\blfs\b|\brfs\b", q):
            regions.append("foraminal")
        if re.search(r"\bcentral\b|\bspinal canal\b|\bscs\b|\bccs\b", q):
            regions.append("central")
        if re.search(r"\blateral recess\b|\blls\b|\brls\b", q):
            regions.append("lateral")

        def region_matches(entry, region):
            title = entry["title"].lower()
            if region == "foraminal":
                return bool(re.search(r"foramin|\blfs\b|\brfs\b", title)) or (
                    entry["view"] == "sagittal" and entry["kind"] in ("gradcam", "roi"))
            if region == "central":
                return bool(re.search(r"central|\bscs\b|\bccs\b", title))
            return bool(re.search(r"lateral|\blls\b|\brls\b", title))

        if regions and kind in ("gradcam", "roi"):
            candidates = [e for e in candidates if any(region_matches(e, r) for r in regions)]
        left = bool(re.search(r"\bleft\b|\blfs\b|\blls\b", q))
        right = bool(re.search(r"\bright\b|\brfs\b|\brls\b", q))
        if left != right and kind in ("gradcam", "roi"):
            pattern = r"\bleft\b|\blfs\b|\blls\b" if left else r"\bright\b|\brfs\b|\brls\b"
            candidates = [e for e in candidates if re.search(pattern, e["title"].lower())]

        # Rank what is left by how much of the request the title actually accounts
        # for, so the leading candidate is the best textual match rather than
        # whichever happened to be built first.
        terms = [w for w in re.findall(r"[a-z0-9-]{3,}", q) if w not in _SELECT_STOPWORDS]
        if terms:
            def score(entry):
                title = entry["title"].lower()
                hits = 0
                for w in terms:
                    if w in title or any(a in title for a in _TERM_ALIASES.get(w, ())):
                        hits += 1
                return hits
            candidates = sorted(candidates, key=score, reverse=True)

        return candidates

    def select_image(self, question: str, catalog: List[Dict[str, Any]]):
        """Pick the image the question asks for, choosing only from the catalogue.

        The LLM is given the numbered catalogue built from the patient JSON and may
        only answer with one of those numbers, so it cannot name a scan the
        pipeline never produced. It is consulted only when the JSON-side filtering
        leaves a genuine choice.
        """
        if not catalog:
            return None

        candidates = self._prefilter_catalog(question, catalog)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None

        if self.use_llm and self.llm:
            listing = format_image_catalog(candidates)
            prompt = (
                "You are selecting which existing MRI image to display.\n"
                "Choose the ONE entry that best matches the request.\n\n"
                f"Available images:\n{listing}\n\n"
                f"Request: \"{question}\"\n\n"
                "Answer with the entry number only (for example: 3). "
                "If none of them match, answer NONE. Do not explain."
            )
            try:
                response = self.llm.invoke(prompt)
                raw = (getattr(response, "content", "") or "").strip()
                match = re.search(r"\d+", raw)
                if match:
                    idx = int(match.group()) - 1
                    if 0 <= idx < len(candidates):
                        return candidates[idx]
                logger.info(f"Image selector returned no usable choice: {raw!r}")
            except Exception as e:
                logger.warning(f"Image selector LLM call failed: {e}")

        # No LLM, or it declined: fall back to the most specific filtered candidate
        return candidates[0]

    def _image_only_nodes(self, user_question: str, title: str, count: int) -> List[Dict[str, Any]]:
        """Debate-graph nodes for a request the moderator served straight from the record."""
        idle = "Not consulted - image retrieval only."
        specialists = [
            ("orthopedic", "Orthopedic Specialist Agent", "Spine Alignment & Biomechanics"),
            ("neuroradiologist", "Neuroradiology Agent", "Intervertebral Discs & Neural Foramina"),
            ("neurosurgeon", "Neurosurgeon Agent", "Spinal Canal & Lateral Recesses"),
            ("search", "Online Search / Literature Agent", "Medical Guidelines & Clinical Studies"),
        ]
        nodes = [{
            "id": "input",
            "name": "Patient Question",
            "role": "User Query Context",
            "input": f"Question: \"{user_question}\"",
            "output": "Image request - matched against the loaded patient record."
        }]
        nodes += [{"id": i, "name": n, "role": r, "input": idle, "output": idle}
                  for i, n, r in specialists]
        nodes.append({
            "id": "moderator",
            "name": "Consensus Moderator / Chief Radiologist",
            "role": "Panel Debate & Consensus Report",
            "input": f"Image lookup for: \"{user_question}\"",
            "output": f"Returned '{title}' ({count} image(s) matched)."
        })
        return nodes

    def run_chat_query(self, user_question: str, chat_history: List[Dict[str, str]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Deliberate the user's specific chat question across the LangChain agent framework."""
        chat_history = chat_history or []

        # A request for the full report is answered with the structured Consensus
        # Diagnostic Report from the panel pipeline (same as batch_process.py),
        # not a conversational summary. Level triage does not apply: the report
        # covers every lumbar level, so there is nothing to clarify.
        if is_report_request(user_question):
            report, panel_nodes = self.run_consensus_report(user_question, chat_history)
            report_nodes = [{
                "id": "input",
                "name": "Patient Question",
                "role": "User Query Context",
                "input": f"Question: \"{user_question}\"",
                "output": "Report request - convened the full panel for the Consensus Diagnostic Report."
            }] + panel_nodes
            return report, report_nodes

        # "Show me the L4-L5 Grad-CAM" wants the scan, not a panel discussion of it.
        # The image is chosen from the catalogue built out of the patient JSON, so
        # the answer is always a scan that exists rather than generated prose.
        from spine_swarm.image_requests import sagittal_severity_images, typed_image_reply
        severity_reply = sagittal_severity_images(user_question, self.pipeline_data)
        if severity_reply is not None:
            return severity_reply, self._image_only_nodes(
                user_question, "highest recorded sagittal foraminal grade", severity_reply.count("!["))

        typed_reply = typed_image_reply(user_question, self)
        if typed_reply is not None:
            return typed_reply, self._image_only_nodes(user_question, "recorded image lookup", typed_reply.count("!["))

        if is_image_only_request(user_question):
            catalog = build_image_catalog(self.pipeline_data)
            chosen = self.select_image(user_question, catalog)
            if chosen:
                title = chosen["title"]
                reply = f"{title}:\n\n![{title}]({chosen['url']})"
                related = [e["title"] for e in self._prefilter_catalog(user_question, catalog)
                           if e["id"] != chosen["id"]][:3]
                if related:
                    reply += "\n\nAlso available: " + ", ".join(related) + "."
                return reply, self._image_only_nodes(user_question, title, len(catalog))
            if catalog:
                # Asked for a scan the pipeline did not produce - say so rather
                # than letting the panel improvise an answer.
                views = sorted({e["view"] for e in catalog})
                kinds = sorted({e["kind"] for e in catalog})
                reply = (
                    "That scan is not in this patient's record. "
                    f"Available views: {', '.join(views)}. "
                    f"Available image types: {', '.join(kinds)}. "
                    "Please specify a lumbar level (L1-L2 to L5-S1)."
                )
                return reply, self._image_only_nodes(user_question, "none", 0)

        # Run triage check first
        intake_res = self.run_intake_agent(user_question, chat_history)
        if "[CLEAR]" not in intake_res:
            triage_nodes = [
                {
                    "id": "input",
                    "name": "Patient Question",
                    "role": "User Query Context",
                    "input": f"Question: \"{user_question}\"",
                    "output": "Triage agent flagged query as ambiguous or lacking level details."
                },
                {
                    "id": "orthopedic",
                    "name": "Orthopedic Specialist Agent",
                    "role": "Spine Alignment & Biomechanics",
                    "input": "Waiting for level clarification...",
                    "output": "Awaiting user specification."
                },
                {
                    "id": "neuroradiologist",
                    "name": "Neuroradiology Agent",
                    "role": "Intervertebral Discs & Neural Foramina",
                    "input": "Waiting for level clarification...",
                    "output": "Awaiting user specification."
                },
                {
                    "id": "neurosurgeon",
                    "name": "Neurosurgeon Agent",
                    "role": "Spinal Canal & Lateral Recesses",
                    "input": "Waiting for level clarification...",
                    "output": "Awaiting user specification."
                },
                {
                    "id": "search",
                    "name": "Online Search / Literature Agent",
                    "role": "Medical Guidelines & Clinical Studies",
                    "input": "Waiting for level clarification...",
                    "output": "Awaiting user specification."
                },
                {
                    "id": "moderator",
                    "name": "Consensus Moderator / Chief Radiologist",
                    "role": "Panel Debate & Consensus Report",
                    "input": "Clarification required from patient.",
                    "output": intake_res
                }
            ]
            return intake_res, triage_nodes

        patient_data_str = get_patient_context_str(self.pipeline_data)
        chat_history_str = format_chat_history(chat_history)
        
        # Specialist answers with context history
        ortho_q = f"Patient Findings Context:\n{patient_data_str}\n\nConversation History:\n{chat_history_str}\n\nUser Question: \"{user_question}\"\nAnalyze alignment based on the data and previous context, and answer in 1-2 sentences."
        ortho_ans = self._call_specialist_llm("Orthopedic Specialist", ORTHOPEDIC_PROMPT, ortho_q)
        
        neurorad_q = f"Patient Findings Context:\n{patient_data_str}\n\nConversation History:\n{chat_history_str}\n\nUser Question: \"{user_question}\"\nAnalyze disc/foramen narrowing based on the data and previous context, and answer in 1-2 sentences."
        neurorad_ans = self._call_specialist_llm("Neuroradiology Agent", NEURORADIOLOGIST_PROMPT, neurorad_q)
        
        neuro_q = f"Patient Findings Context:\n{patient_data_str}\n\nConversation History:\n{chat_history_str}\n\nUser Question: \"{user_question}\"\nAnalyze canal stenosis based on the data and previous context, and answer in 1-2 sentences."
        neuro_ans = self._call_specialist_llm("Neurosurgeon Agent", NEUROSURGEON_PROMPT, neuro_q)
        
        search_q = f"Patient Findings Context:\n{patient_data_str}\n\nConversation History:\n{chat_history_str}\n\nUser Question: \"{user_question}\"\nSearch guidelines for the relevant levels/stenosis based on the data and previous context, and answer in 1-2 sentences."
        search_ans = self._call_specialist_llm("Online Search / Literature Agent", SEARCH_PROMPT, search_q)
        
        # Default fallback Moderator compiler with context history
        fallback_compiler = lambda: self._call_specialist_llm(
            "Consensus Moderator",
            MODERATOR_CHAT_PROMPT,
            f"Conversation History:\n{chat_history_str}\n\nUser Question: \"{user_question}\"\nOrtho: {ortho_ans}\nNeurorad: {neurorad_ans}\nNeuro: {neuro_ans}\nSearch: {search_ans}"
        )
        
        final_reply = ""
        
        if self.use_llm and self.llm and initialize_agent is not None:
            try:
                tools = [
                    Tool(
                        name="orthopedic_specialist",
                        func=lambda q: ortho_ans,
                        description="Consult the orthopedic specialist for alignment."
                    ),
                    Tool(
                        name="neuroradiologist",
                        func=lambda q: neurorad_ans,
                        description="Consult the neuroradiologist for discs and foramina."
                    ),
                    Tool(
                        name="neurosurgeon",
                        func=lambda q: neuro_ans,
                        description="Consult the neurosurgeon for canal and lateral recesses."
                    ),
                    Tool(
                        name="medical_search_agent",
                        func=lambda q: search_ans,
                        description="Consult the literature search agent for clinical guidelines."
                    )
                ]
                
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                for msg in chat_history:
                    if msg.get("role") == "user":
                        memory.chat_memory.add_user_message(msg["content"])
                    else:
                        memory.chat_memory.add_ai_message(msg["content"])
                
                agent_executor = initialize_agent(
                    tools,
                    self.llm,
                    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=memory,
                    verbose=True
                )
                
                agent_prompt = f"""You are the Consensus Moderator leading the SpineSwarm clinical panel.
The patient has asked: "{user_question}"
Patient Findings Context:
{patient_data_str}
Previous Conversation History:
{chat_history_str}

Please consult the relevant specialists using their tools, compile their opinions, resolve any discrepancy, and formulate the final consensus response.
Guidelines:
1. If the user asks a simple question (e.g. "At which vertebrae is the stenosis?"), respond in a very simple, direct one or two lines of response based on the findings.
2. If the user asks for a report, generate the comprehensive Consensus Report with standard headings.
3. If they ask to view or show an image (e.g. "show me the vertebrae height measurements image"), return the markdown image link(s) for the matched images.
Frame your final answer clearly as the consensus of the SpineSwarm panel."""
                
                response = agent_executor.invoke({"input": agent_prompt})
                final_reply = response.get("output", "")
            except Exception as e:
                logger.warning(f"Error running LangChain initialize_agent: {e}. Falling back to default prompt compilation.")
                
        if not final_reply:
            final_reply = fallback_compiler()
            
        # Strip any LLM-hallucinated markdown images to prevent broken images
        import re
        final_reply = re.sub(r'!\[[\s\S]*?\]\s*\([\s\S]*?\)', '', final_reply)
            
        # Check for image request and append markdown image if found
        matched_image = self.select_image(user_question, build_image_catalog(self.pipeline_data)) if (
            _has_term(user_question.lower(), IMAGE_VERBS) and _has_term(user_question.lower(), IMAGE_NOUNS)
        ) else None
        if matched_image:
            img_markdown = f"\n\n![{matched_image['title']}]({matched_image['url']})"
            if img_markdown not in final_reply:
                final_reply += img_markdown
            
        # Compile debate nodes
        debate_nodes = [
            {
                "id": "input",
                "name": "Patient Question",
                "role": "User Query Context",
                "input": f"Question: \"{user_question}\"",
                "output": "Analyzed query against loaded DICOM JSON."
            },
            {
                "id": "orthopedic",
                "name": "Orthopedic Specialist Agent",
                "role": "Spine Alignment & Biomechanics",
                "input": ortho_q,
                "output": ortho_ans
            },
            {
                "id": "neuroradiologist",
                "name": "Neuroradiology Agent",
                "role": "Intervertebral Discs & Neural Foramina",
                "input": neurorad_q,
                "output": neurorad_ans
            },
            {
                "id": "neurosurgeon",
                "name": "Neurosurgeon Agent",
                "role": "Spinal Canal & Lateral Recesses",
                "input": neuro_q,
                "output": neuro_ans
            },
            {
                "id": "search",
                "name": "Online Search / Literature Agent",
                "role": "Medical Guidelines & Clinical Studies",
                "input": search_q,
                "output": search_ans
            },
            {
                "id": "moderator",
                "name": "Consensus Moderator / Chief Radiologist",
                "role": "Panel Debate & Consensus Report",
                "input": f"Moderator query with panel answers compiled.",
                "output": final_reply
            }
        ]
        
        return final_reply, debate_nodes

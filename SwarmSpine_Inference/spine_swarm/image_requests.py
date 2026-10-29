"""Select existing classification images without generating clinical findings."""
import re
from .agents import normalize_patient_data, extract_path_or_url, _level_from_index


def clinical_label(value):
    text = str(value).replace("_", " ")
    for term, replacement in {
        "Grad-CAM": "MRI", "ROI Detection": "Anatomical localization",
        "LFS": "Left foraminal", "RFS": "Right foraminal",
        "SCS": "Central canal", "LLS": "Left lateral recess",
        "RLS": "Right lateral recess", "Severe Stenosis": "Severe stenosis",
        "corner points": "Landmarks",
    }.items():
        text = re.sub(r"\b" + re.escape(term) + r"\b", replacement, text, flags=re.I)
    return text


def measurement_text(details):
    """Present saved values as readable text without internal coordinates."""
    rows = []
    def visit(value, labels):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("corner_points", "bbox", "image_overlay", "image_overlay_path"):
                    continue
                visit(child, labels + [clinical_label(key)])
        elif not isinstance(value, (list, tuple)):
            shown = "Yes" if value is True else "No" if value is False else "Unavailable" if value is None else clinical_label(value)
            rows.append("- " + (" / ".join(labels) + ": " if labels else "") + shown)
    visit(details, [])
    return "\n".join(rows)


def requested_image_kind(question):
    q = question.lower()
    if re.search(r"segment|\bmask\b", q):
        return "segmentation"
    if re.search(r"detect|\broi\b|bounding", q):
        return "roi"
    if re.search(r"measur|height|lordosis|angle|deform|spondylolisthesis|herniation", q):
        return "measurement"
    if re.search(r"classif|grad.?cam|heatmap|activation", q):
        return "gradcam"
    if re.search(r"image|show|display|scan", q) and re.search(r"stenosis|foramin|central canal|lateral recess", q):
        return "gradcam"
    return None


def typed_image_reply(question, panel):
    """Use typed catalog entries only; never ask an LLM to invent a path."""
    from .agents import build_image_catalog
    kind = requested_image_kind(question)
    if kind is None:
        return None
    catalog = [e for e in build_image_catalog(panel.pipeline_data) if e["kind"] == kind]
    candidates = panel._prefilter_catalog(question, catalog)
    if not candidates:
        return "No matching image is available for the requested examination, level, and side."
    chosen = candidates[0]
    title = clinical_label(chosen["title"])
    lines = [title]
    if chosen.get("grade"):
        lines.append(f"Assessment: {clinical_label(chosen['grade'])}.")
    if chosen.get("details"):
        details = chosen["details"]
        levels = re.findall(r"[lt]\d+\s*[-–—]\s*[ls]\d+", question.lower())
        if levels and isinstance(details, dict):
            key = re.sub(r"\s", "", levels[0]).replace("–", "-").replace("—", "-").upper()
            if key in details:
                details = {key: details[key]}
        lines.append("Measurements:\n\n" + measurement_text(details))
    if len(candidates) > 1:
        lines.append("Showing one matching image. Specify the side or level to narrow the selection.")
    lines.append(f"![{title}]({chosen['url']})")
    return "\n\n".join(lines)


def sagittal_severity_images(question, data):
    q = re.sub(r"[–—−]", "-", question.lower())
    if not ("sagittal" in q and re.search(r"foramin", q)
            and re.search(r"most severe|highest severity|worst", q)
            and re.search(r"grad.?cam|heatmap|image", q)):
        return None
    data = normalize_patient_data(data)
    classes = data.get("Sagittal_ROI_classification") or {}
    ranks = {"normal": 0, "stenosis": 1, "severe_stenosis": 2}
    levels = re.findall(r"l[1-5]\s*-\s*(?:l[2-5]|s1)", q)
    levels = {re.sub(r"\s", "", level).upper() for level in levels}
    left, right = bool(re.search(r"\bleft\b|\blfs\b", q)), bool(re.search(r"\bright\b|\brfs\b", q))
    records = []
    unknown = False
    for side, key in (("Left", "left_classification"), ("Right", "right_classification")):
        if left != right and (side == "Left") != left:
            continue
        for item in classes.get(key) or []:
            level = _level_from_index(item)
            if levels and level not in levels:
                continue
            grade = str(item.get("predicted_class", ""))
            rank = ranks.get(grade.lower().replace(" ", "_"))
            if rank is None:
                unknown = True
                continue
            records.append((rank, level, side, grade, extract_path_or_url(item.get("gradcam_path"), data)))
    if not records:
        return "Sagittal foraminal severity is unavailable for the requested level and side."
    maximum = max(r[0] for r in records)
    matches = list(dict.fromkeys(r for r in records if r[0] == maximum))
    lines = []
    # Select only among the highest-grade records, never a lower grade just
    # because it has an image. Record order breaks ties, not clinical severity.
    _, level, side, grade, url = next((r for r in matches if r[4]), matches[0])
    if grade.lower() == "normal":
        finding = f"No foraminal stenosis identified on the {side.lower()} at {level}."
    else:
        severity = "Severe " if grade.lower().replace(" ", "_") == "severe_stenosis" else ""
        finding = f"{severity}{side.lower()} foraminal stenosis at {level}."
        finding = finding[0].upper() + finding[1:]
    title = f"Sagittal MRI — {level}, {side.lower()} side"
    lines.append(finding)
    if len(matches) > 1:
        lines.append("Other images show the same severity; one representative image is displayed.")
    if unknown:
        lines.append("Severity is unavailable for some findings, so the comparison is incomplete.")
    lines.append(f"\n![{title}]({url})" if url else "The corresponding image is unavailable.")
    return "\n".join(lines)

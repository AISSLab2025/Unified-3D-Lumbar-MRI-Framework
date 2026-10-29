"""Render saved study findings without asking an LLM to invent report structure.

No severity thresholds, probability fusion or substitute patient values are used.
Conflicting recorded classifications remain explicit rather than being overwritten.
Input is the normalized payload produced by agents.normalize_patient_data.
"""
import math
import re

LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
VERTEBRAE = ("L1", "L2", "L3", "L4", "L5")
SECTIONS = (
    "TECHNIQUE", "FINDINGS", "IVD-LEVEL ANALYSIS", "VERTEBRA-LEVEL ANALYSIS",
    "SPINAL ALIGNMENT AND MEASUREMENTS", "IMPRESSION", "RECOMMENDATION",
)
REGIONS = {
    "SCS": "central canal", "LLS": "left lateral recess", "RLS": "right lateral recess",
    "LFS": "left neural foraminal", "RFS": "right neural foraminal",
}
GRADES = {"normal": "normal", "stenosis": "stenosis", "severe_stenosis": "severe stenosis"}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _level(value):
    return re.sub(r"\s+", "", re.sub(r"[–—−]", "-", str(value))).upper()


def _measurements(data, name):
    block = _dict(data.get(name))
    return _dict(block.get("measurements") or block.get("measurements_info"))


def _number(value, unit=""):
    """Accept finite numbers or numeric strings with the expected unit only."""
    if isinstance(value, bool) or value is None:
        return "not available"
    text = str(value).strip()
    if unit and text.endswith(unit):
        text = text[:-len(unit)].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return "not available"
    if not math.isfinite(float(text)):
        return "not available"
    return text + (unit if unit in ("%", "°") else " " + unit if unit else "")


def _classification_cells(data):
    cells = {}

    def add(item, region, plane):
        level = _level(item.get("level_name") or item.get("level") or "")
        if level not in LEVELS or region not in REGIONS:
            return
        raw = str(item.get("predicted_class") or "").lower().replace(" ", "_")
        # Unknown classifications are retained as unassessed, not normal.
        grade = GRADES.get(raw, "not assessed")
        pair = (plane, grade)
        entries = cells.setdefault((level, region), [])
        if pair not in entries:
            entries.append(pair)

    sagittal = _dict(data.get("Sagittal_ROI_classification"))
    for side, region in (("left", "LFS"), ("right", "RFS")):
        for item in _records(sagittal.get(side + "_classification")):
            add(item, region, "sagittal")
    for key in ("Axial_foraminal_classification", "Axial_lateral_classification", "Axial_central_classification"):
        for item in _records(data.get(key)):
            region = str(item.get("region_name") or "").upper()
            if region == "CCS":
                region = "SCS"
            # The central classifier has one region; never infer bilateral sides.
            if not region and key == "Axial_central_classification":
                region = "SCS"
            add(item, region, "axial")
    return cells


def _stenosis(entries, region):
    name = REGIONS[region]
    grades = {grade for _, grade in entries}
    if not grades or grades == {"not assessed"}:
        return f"{name.capitalize()} stenosis: not assessed."
    if len(grades) > 1:
        detail = "; ".join(f"{plane}: {grade}" for plane, grade in entries)
        return f"{name.capitalize()} assessment is discordant ({detail})."
    grade = next(iter(grades))
    if grade == "normal":
        return f"No {name} stenosis is identified."
    return f"{'Severe ' if grade == 'severe stenosis' else ''}{name} stenosis is present.".capitalize()


def _herniation(records):
    if not records:
        return "Disc herniation: not assessed; herniation ratio: not available."
    descriptions = []
    for item in records:
        present = item.get("disc_herniation")
        finding = ("Disc herniation is present" if present is True else
                   "No disc herniation is identified" if present is False else
                   "Disc herniation: not assessed")
        sentence = f"{finding}; herniation ratio: {_number(item.get('herniation_ratio'))}."
        if sentence not in descriptions:
            descriptions.append(sentence)
    return ("Multiple assessments: " if len(descriptions) > 1 else "") + " ".join(descriptions)


def _deformity(record, name):
    label = name.lower() + " deformity"
    grade = str(record.get(name + " Classification") or "").lower()
    if grade == "normal":
        finding = f"No {label} is identified"
    elif grade in ("mild", "moderate", "severe"):
        finding = f"{grade.capitalize()} {label} is present"
    else:
        finding = f"{label.capitalize()}: not assessed"
    return f"{finding}; {label} ratio: {_number(record.get(name + ' Deformity'), '%')}."


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _brief_stenosis(cells):
    """Group levels with the same abnormalities; preserve side and severity."""
    groups = {}
    affected = []
    discordant = []
    for level in LEVELS:
        phrases = []
        for regions, bilateral in (
            (("LFS", "RFS"), "bilateral neural foraminal"),
            (("LLS", "RLS"), "bilateral lateral recess"),
            (("SCS",), "central canal"),
        ):
            grades = [{g for _, g in cells.get((level, r), [])} for r in regions]
            if len(regions) == 2 and grades[0] == grades[1] and len(grades[0]) == 1:
                grade = next(iter(grades[0]))
                if grade in ("stenosis", "severe stenosis"):
                    phrases.append(("severe " if grade == "severe stenosis" else "") + bilateral + " stenosis")
                continue
            for region, values in zip(regions, grades):
                positives = values & {"stenosis", "severe stenosis"}
                if not positives:
                    continue
                if len(values) > 1:
                    phrases.append(REGIONS[region] + " stenosis with discordant assessments")
                    if level not in discordant:
                        discordant.append(level)
                else:
                    phrases.append(("severe " if "severe stenosis" in positives else "") + REGIONS[region] + " stenosis")
        if phrases:
            affected.append(level)
            groups.setdefault(tuple(phrases), []).append(level)
    sentences = [f"At {_join(levels)}, there is {_join(phrases)}." for phrases, levels in groups.items()]
    return sentences, affected, discordant


def render_clinical_report(data):
    """All headings and level bullets are code-owned, including for partial studies."""
    data = _dict(data)
    meas = _dict(data.get("Sagittal_measurements_and_diagnosis"))
    cells = _classification_cells(data)
    axial = _dict(data.get("axial"))
    axial_meas = _dict(data.get("Axial_measurements") or axial.get("measurements"))
    herniation = {level: [] for level in LEVELS}
    for key, item in axial_meas.items():
        if isinstance(item, dict):
            level = _level(item.get("level") or key.replace("_herniation", ""))
            if level in herniation:
                herniation[level].append(item)
    if not cells and not meas and not any(herniation.values()):
        raise ValueError("No recognized study measurements or classifications are available for the report.")

    summary, stenosis_levels, discordant_levels = _brief_stenosis(cells)
    herniation_levels = []
    uncertain_herniation = []
    discs = []
    for level in LEVELS:
        clauses = [_herniation(herniation[level])]
        for region in REGIONS:
            entries = cells.get((level, region), [])
            clause = _stenosis(entries, region)
            clauses.append(clause)
        if any(item.get("disc_herniation") is True for item in herniation[level]):
            if all(item.get("disc_herniation") is True for item in herniation[level]):
                herniation_levels.append(level)
            else:
                uncertain_herniation.append(level)
        discs.append(f"• At the {level} level: " + " ".join(clauses))

    herniation_summary = []
    if herniation_levels:
        herniation_summary.append(f"Disc herniation is present at {_join(herniation_levels)}.")
    if uncertain_herniation:
        herniation_summary.append(f"Disc herniation assessments are discordant at {_join(uncertain_herniation)}.")
    summary.extend(herniation_summary)

    deformities = _measurements(meas, "vertebrae_deformities")
    displacement = _measurements(meas, "spondylolisthesis")
    vertebrae = []
    vertebral_groups = {}
    for level in VERTEBRAE:
        record = _dict(deformities.get(level))
        clauses = [_deformity(record, name) for name in ("Wedge", "Biconcave")]
        for name in ("Wedge", "Biconcave"):
            severity = str(record.get(name + " Classification") or "").lower()
            if severity in ("mild", "moderate", "severe"):
                vertebral_groups.setdefault(f"{severity.capitalize()} {name.lower()} deformity", []).append(level)
        grade = str(_dict(displacement.get(level)).get("Classification") or "").lower()
        if grade == "normal":
            slip = "No vertebral displacement is identified."
        elif grade in ("anterior displacement", "posterior displacement"):
            slip = f"{grade.capitalize()} of {level} is identified."
            vertebral_groups.setdefault(grade.capitalize(), []).append(level)
        else:
            slip = "Vertebral displacement: not assessed."
        vertebrae.append(f"• At the {level} vertebral level: " + " ".join(clauses + [slip]))

    vertebral_summary = [f"{finding} is present at {_join(levels)}." for finding, levels in vertebral_groups.items()]
    summary.extend(vertebral_summary)

    angles = _measurements(meas, "angles")
    height = _measurements(meas, "spinal_height")
    height_value = height.get("superior_hight")
    if height_value is None:
        height_value = height.get("superior_height")
    lordosis = _dict(meas.get("lumbar_lordosis")).get("measurements")
    if isinstance(lordosis, dict):
        lordosis = lordosis.get("Classification")
    lordosis = lordosis.split(":")[0].strip() if isinstance(lordosis, str) else "not assessed"
    alignment = (
        f"Lumbar lordosis angle (LLA): {_number(angles.get('LLA'), '°')}; "
        f"lumbosacral angle (LSA): {_number(angles.get('LSA'), '°')}; "
        f"lumbar spinal height (L1-S1): {_number(height_value, 'mm')}; "
        f"lumbar lordosis: {lordosis or 'not assessed'}."
    )
    alignment_summary = []
    if lordosis and lordosis.lower() not in ("normal", "normal lordosis", "not assessed"):
        alignment_summary.append(f"Lumbar lordosis assessment: {lordosis}.")
    summary.extend(alignment_summary)
    findings = " ".join(summary) or (
        "No positive classification is documented in the available assessments. "
        "Unassessed structures are identified below and cannot be considered normal."
    )
    impression_parts = []
    if stenosis_levels:
        prefix = "Multilevel lumbar stenosis" if len(stenosis_levels) > 1 else "Lumbar stenosis"
        impression_parts.append(f"{prefix} is documented at {_join(stenosis_levels)}.")
    if discordant_levels:
        impression_parts.append(f"Stenosis assessments differ at {_join(discordant_levels)}; see level-specific analysis.")
    impression_parts.extend(herniation_summary + vertebral_summary + alignment_summary)
    impression = " ".join(impression_parts) or findings
    bodies = (
        "Automated analysis of the supplied lumbar MRI study includes assessment of the "
        "vertebrae, intervertebral discs (IVDs), neural compartments and lumbar alignment. "
        "Acquisition sequence details and complete-volume coverage are not verified in this report.",
        findings,
        "Intervertebral disc assessment covers disc herniation, the central canal, bilateral lateral "
        "recesses and bilateral neural foramina from L1-L2 through L5-S1.\n" + "\n".join(discs),
        "Vertebral assessment covers wedge deformity, biconcave deformity and vertebral "
        "displacement from L1 through L5.\n" + "\n".join(vertebrae),
        alignment,
        impression,
        "Correlation with clinical symptoms and neurological findings is advised. "
        "Specialist evaluation may be considered where clinically appropriate.",
    )
    return "\n\n".join(f"{heading}\n{body}" for heading, body in zip(SECTIONS, bodies))

"""Synthetic Nigerian specimen document generator.

Renders clearly-fake documents with known ground-truth fields so the harness is
runnable today with zero legal/privacy risk. Every document carries a
"SPECIMEN - NOT A REAL DOCUMENT" watermark. The same manifest/expected format
accepts real, legally-cleared samples later.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..models import (
    CaptureCondition,
    DocumentType,
    ExpectedFields,
    Manifest,
    Sample,
)
from ..schemas import schema_for
from . import degradations
from .faces import make_face

# --------------------------------------------------------------------------- #
# Fake data bank (clearly fictional)
# --------------------------------------------------------------------------- #
_FIRST = ["Ada", "Chidi", "Ngozi", "Emeka", "Bola", "Yusuf", "Aisha", "Tunde",
          "Funke", "Ibrahim", "Zainab", "Obinna", "Halima", "Segun", "Amaka"]
_MIDDLE = ["Oluwaseun", "Chukwuemeka", "Adaeze", "Olamide", "Ifeoma", "Babatunde",
           "Chinwe", "Abubakar", "Temitope", "Ucheoma"]
_SURNAME = ["Okafor", "Adeyemi", "Mohammed", "Eze", "Balogun", "Okonkwo", "Bello",
            "Afolabi", "Nwosu", "Abubakar", "Lawal", "Ogunleye", "Onyeka"]
_STREETS = ["12 Allen Avenue", "47 Awolowo Road", "8 Marina Street",
            "23 Ahmadu Bello Way", "5 Aminu Kano Crescent", "90 Ikorodu Road"]
_CITIES = [("Ikeja", "Lagos"), ("Wuse", "Abuja"), ("Garki", "Abuja"),
           ("Victoria Island", "Lagos"), ("GRA", "Port Harcourt"), ("Bompai", "Kano")]
_AUTHORITIES = {
    DocumentType.PASSPORT: "Nigeria Immigration Service",
    DocumentType.DRIVERS_LICENSE: "Federal Road Safety Corps",
    DocumentType.UTILITY_BILL: "Ikeja Electric PLC",
    DocumentType.BANK_STATEMENT: "Specimen Bank PLC",
}
_BANKS = ["Specimen Bank PLC", "Demo Trust Bank", "Sample MFB"]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _name_parts(rng: random.Random) -> tuple[str, str, str]:
    return rng.choice(_FIRST), rng.choice(_MIDDLE), rng.choice(_SURNAME)


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _dob(rng: random.Random) -> str:
    start = date(1965, 1, 1)
    d = start + timedelta(days=rng.randint(0, 365 * 45))
    return d.isoformat()


def _future_date(rng: random.Random, years_ahead_min=1, years_ahead_max=8) -> str:
    d = date.today() + timedelta(days=365 * rng.randint(years_ahead_min, years_ahead_max))
    return d.isoformat()


def _past_date(rng: random.Random, years_back_max=5) -> str:
    d = date.today() - timedelta(days=rng.randint(30, 365 * years_back_max))
    return d.isoformat()


def _address(rng: random.Random) -> str:
    city, state = rng.choice(_CITIES)
    return f"{rng.choice(_STREETS)}, {city}, {state}"


# --------------------------------------------------------------------------- #
# Field builders per document type
# --------------------------------------------------------------------------- #
def _build_fields(doc_type: DocumentType, rng: random.Random) -> dict[str, str]:
    first, middle, surname = _name_parts(rng)
    # full_name comes in two flavours: docs that render Surname + First Name
    # lines separately get the no-middle form (matches what's visible); docs that
    # render a single "Name:" line get the full first-middle-surname form.
    full_split = f"{first} {surname}"
    full_combined = f"{first} {middle} {surname}"
    gender = rng.choice(["Male", "Female"])
    dob = _dob(rng)
    addr = _address(rng)

    if doc_type == DocumentType.NIN_SLIP:
        return {"full_name": full_split, "surname": surname, "first_name": first,
                "date_of_birth": dob, "gender": gender, "nin": _digits(rng, 11),
                "address": addr, "issue_date": _past_date(rng)}
    if doc_type == DocumentType.NATIONAL_ID:
        return {"full_name": full_split, "surname": surname, "first_name": first,
                "date_of_birth": dob, "gender": gender, "nin": _digits(rng, 11),
                "document_number": _digits(rng, 9), "expiry_date": _future_date(rng)}
    if doc_type == DocumentType.PASSPORT:
        return {"full_name": full_split, "surname": surname, "first_name": first,
                "date_of_birth": dob, "gender": gender,
                "passport_number": rng.choice("ABCN") + _digits(rng, 8),
                "issue_date": _past_date(rng), "expiry_date": _future_date(rng),
                "issuing_authority": _AUTHORITIES[DocumentType.PASSPORT]}
    if doc_type == DocumentType.DRIVERS_LICENSE:
        return {"full_name": full_combined, "date_of_birth": dob, "gender": gender,
                "drivers_license_number": surname[:3].upper() + _digits(rng, 8),
                "issue_date": _past_date(rng), "expiry_date": _future_date(rng),
                "address": addr,
                "issuing_authority": _AUTHORITIES[DocumentType.DRIVERS_LICENSE]}
    if doc_type == DocumentType.VOTER_ID:
        return {"full_name": full_combined, "date_of_birth": dob, "gender": gender,
                "document_number": _digits(rng, 19), "address": addr}
    if doc_type == DocumentType.BANK_MANDATE:
        return {"full_name": full_combined, "bvn": _digits(rng, 11), "nin": _digits(rng, 11),
                "account_number": _digits(rng, 10), "date_of_birth": dob, "address": addr}
    if doc_type == DocumentType.UTILITY_BILL:
        return {"full_name": full_combined, "address": addr, "issue_date": _past_date(rng, 1),
                "issuing_authority": _AUTHORITIES[DocumentType.UTILITY_BILL]}
    if doc_type == DocumentType.BANK_STATEMENT:
        return {"full_name": full_combined, "account_number": _digits(rng, 10), "address": addr,
                "issue_date": _past_date(rng, 1), "issuing_authority": rng.choice(_BANKS)}
    return {"full_name": full_combined}


_TITLES = {
    DocumentType.NIN_SLIP: "FEDERAL REPUBLIC OF NIGERIA\nNATIONAL IDENTIFICATION NUMBER SLIP",
    DocumentType.NATIONAL_ID: "FEDERAL REPUBLIC OF NIGERIA\nNATIONAL IDENTITY CARD",
    DocumentType.PASSPORT: "FEDERAL REPUBLIC OF NIGERIA\nINTERNATIONAL PASSPORT",
    DocumentType.DRIVERS_LICENSE: "FEDERAL REPUBLIC OF NIGERIA\nDRIVER'S LICENCE (FRSC)",
    DocumentType.VOTER_ID: "INDEPENDENT NATIONAL ELECTORAL COMMISSION\nPERMANENT VOTER CARD",
    DocumentType.BANK_MANDATE: "ACCOUNT OPENING MANDATE FORM",
    DocumentType.UTILITY_BILL: "IKEJA ELECTRIC PLC\nELECTRICITY UTILITY BILL",
    DocumentType.BANK_STATEMENT: "SPECIMEN BANK PLC\nSTATEMENT OF ACCOUNT",
}

_LABELS = {
    "full_name": "Name", "surname": "Surname", "first_name": "First Name",
    "middle_name": "Middle Name", "date_of_birth": "Date of Birth", "gender": "Sex",
    "nin": "NIN", "bvn": "BVN", "passport_number": "Passport No",
    "drivers_license_number": "Licence No", "document_number": "Document No",
    "account_number": "Account No", "issue_date": "Date of Issue",
    "expiry_date": "Date of Expiry", "address": "Address",
    "issuing_authority": "Issued By",
}

# Order labels nicely; full_name only shown if surname/first not present.
_LINE_ORDER = ["surname", "first_name", "middle_name", "full_name", "date_of_birth",
               "gender", "nin", "bvn", "passport_number", "drivers_license_number",
               "document_number", "account_number", "address", "issue_date",
               "expiry_date", "issuing_authority"]


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without sized default
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_document(
    doc_type: DocumentType, seed: int
) -> tuple[Image.Image, dict[str, str], Optional[list[int]]]:
    rng = _rng(seed)
    fields = _build_fields(doc_type, rng)
    schema = schema_for(doc_type)
    portrait_expected = bool(schema and schema.portrait_expected)

    W, H = 1000, 640
    bg = (248, 247, 242)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Header band.
    draw.rectangle([0, 0, W, 90], fill=(20, 90, 60))
    title = _TITLES.get(doc_type, doc_type.value.upper())
    ty = 14
    for line in title.split("\n"):
        draw.text((24, ty), line, font=_font(22), fill=(255, 255, 255))
        ty += 30

    # Portrait box.
    portrait_bbox: Optional[list[int]] = None
    text_left = 36
    if portrait_expected:
        px, py, pw, ph = W - 210, 120, 160, 200
        face = make_face(pw, ph, seed=seed)
        img.paste(face, (px, py))
        draw.rectangle([px - 2, py - 2, px + pw + 2, py + ph + 2], outline=(120, 120, 120), width=2)
        portrait_bbox = [px, py, px + pw, py + ph]

    # Field lines.
    y = 120
    shown = [f for f in _LINE_ORDER if f in fields]
    # Avoid showing redundant full_name when surname+first already present.
    if "surname" in fields and "first_name" in fields and "full_name" in shown:
        shown.remove("full_name")
    for fname in shown:
        label = _LABELS.get(fname, fname)
        draw.text((text_left, y), f"{label}:", font=_font(20), fill=(60, 60, 60))
        draw.text((text_left + 230, y), fields[fname], font=_font(20), fill=(15, 15, 15))
        y += 38

    # Watermark.
    draw.text((text_left, H - 40), "SPECIMEN - NOT A REAL DOCUMENT",
              font=_font(18), fill=(190, 60, 60))

    return img, fields, portrait_bbox


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
DEFAULT_DOC_TYPES = [
    DocumentType.NIN_SLIP, DocumentType.NATIONAL_ID, DocumentType.PASSPORT,
    DocumentType.DRIVERS_LICENSE, DocumentType.VOTER_ID, DocumentType.BANK_MANDATE,
    DocumentType.UTILITY_BILL, DocumentType.BANK_STATEMENT,
]
DEFAULT_CONDITIONS = [
    CaptureCondition.CLEAN, CaptureCondition.MOBILE, CaptureCondition.LOW_LIGHT,
    CaptureCondition.GLARE, CaptureCondition.ROTATED, CaptureCondition.WHATSAPP,
]


def generate_dataset(
    out_root: Path,
    doc_types: Optional[list[DocumentType]] = None,
    conditions: Optional[list[CaptureCondition]] = None,
    per_type: int = 1,
    seed: int = 1000,
    manifest_name: str = "synthetic_v1",
) -> Manifest:
    """Generate images + expected files + a manifest under ``out_root``."""
    doc_types = doc_types or DEFAULT_DOC_TYPES
    conditions = conditions or DEFAULT_CONDITIONS
    out_root = Path(out_root)
    samples_dir = out_root / "samples"
    expected_dir = out_root / "expected"
    manifests_dir = out_root / "manifests"
    for d in (samples_dir, expected_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    counter = seed
    for doc_type in doc_types:
        schema = schema_for(doc_type)
        for i in range(per_type):
            counter += 1
            base_img, fields, portrait_bbox = render_document(doc_type, seed=counter)
            for cond in conditions:
                sample_id = f"{doc_type.value}-{i:02d}-{cond.value}"
                img = degradations.apply(cond, base_img)
                rel_path = f"samples/{sample_id}.jpg"
                img.convert("RGB").save(out_root / rel_path, format="JPEG", quality=92)

                expected = ExpectedFields(
                    sample_id=sample_id,
                    document_type=doc_type,
                    fields=fields,
                    required_fields=(schema.required_field_names if schema else []),
                    portrait_present=portrait_bbox is not None,
                    portrait_bbox=portrait_bbox,
                )
                (expected_dir / f"{sample_id}.json").write_text(
                    expected.model_dump_json(indent=2)
                )
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        document_type=doc_type,
                        image_path=rel_path,
                        capture_condition=cond,
                        source="synthetic",
                        tags=["synthetic", doc_type.value, cond.value],
                    )
                )

    manifest = Manifest(
        name=manifest_name,
        description="Synthetic Nigerian specimen documents for benchmark wiring.",
        samples=samples,
    )
    (manifests_dir / f"{manifest_name}.json").write_text(manifest.model_dump_json(indent=2))
    return manifest

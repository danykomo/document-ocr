"""Nigerian document field schemas — evidence-based.

Each schema lists the fields that are actually printed on the real document,
with citations to the issuing authority or a verification provider. We score
required-field recall against the fields a real document carries, not against
a wish list.

Sources used to build each schema are listed inline above the schema. Field
names use snake_case; the prompt asks providers for these exact keys and the
``_coerce_keys`` / ``text_kie`` layers map the document's printed labels onto
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .models import DocumentType
from .normalization import (
    validate_bvn,
    validate_expiry_not_past,
    validate_nin,
    validate_passport_number,
)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    # kind drives normalization + scoring strictness:
    #   name/address/text -> fuzzy partial credit
    #   date/number       -> strict (exact after normalisation)
    kind: str  # name | date | number | address | text | phone | money | code
    required: bool = False
    validator: Optional[Callable[[str], tuple[bool, str]]] = None
    description: str = ""


@dataclass(frozen=True)
class DocumentSchema:
    schema_id: str
    document_type: DocumentType
    fields: list[FieldSpec]
    portrait_expected: bool = False
    sides: tuple[str, ...] = ("front",)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def required_field_names(self) -> list[str]:
        return [f.name for f in self.fields if f.required]


def _f(*args, **kwargs) -> FieldSpec:  # tiny ctor alias for table readability
    return FieldSpec(*args, **kwargs)


SCHEMAS: dict[DocumentType, DocumentSchema] = {
    # ---------------------------------------------------------------------- #
    # NIN slip (NIMC standard / basic slip)
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - NIMC "NIN Slip Re-Issuance" page (nimc.gov.ng/nin/nin-slip-reissuance)
    #   - User-confirmed against 4 real NIN slips: printed fields are
    #     Tracking ID + Surname + First Name + Middle Name + Gender + NIN +
    #     Address + Portrait (right-hand side) + QR code. None of the four
    #     have a date of birth or an issue date printed on the slip.
    DocumentType.NIN_SLIP: DocumentSchema(
        schema_id="ng_nin_slip_v3",
        document_type=DocumentType.NIN_SLIP,
        portrait_expected=True,
        fields=[
            _f("tracking_id", "code"),
            _f("surname", "name", required=True),
            _f("first_name", "name", required=True),
            _f("middle_name", "name"),
            _f("gender", "text"),
            _f("nin", "number", required=True, validator=validate_nin),
            _f("address", "address"),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # NIMC National Identity (e-ID) Card — ICAO ID-1 compliant
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - NIMC "The e-ID Card" page (nimc.gov.ng/e-id-card) — ICAO-standards
    #     compliant card with MRZ, photo, signature, biometrics.
    #   - QoreID NIN-premium-slip response fields (firstname/lastname/middlename,
    #     birthdate, gender, photo, signature).
    DocumentType.NATIONAL_ID: DocumentSchema(
        schema_id="ng_national_id_v2",
        document_type=DocumentType.NATIONAL_ID,
        portrait_expected=True,
        sides=("front", "back"),
        fields=[
            _f("surname", "name", required=True),
            _f("first_name", "name", required=True),
            _f("middle_name", "name"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("nin", "number", required=True, validator=validate_nin),
            _f("document_number", "number"),
            _f("nationality", "text"),
            _f("issue_date", "date"),
            _f("expiry_date", "date", validator=validate_expiry_not_past),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # Nigerian International Passport — data page (ICAO 9303-compliant)
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - Nigeria Immigration Service "Passports" page (immigration.gov.ng/passports)
    #   - QoreID international-passport docs (passport_number, first/last/middle
    #     name, birth date, gender, issued location, issue date, expiry date, photo).
    #   - ICAO 9303 standard data page fields are mirrored in the visible zone.
    # Note: passport convention combines first+middle as "given names" — the
    # parser/coercer treats given_name/given_names as aliases for first_name.
    DocumentType.PASSPORT: DocumentSchema(
        schema_id="ng_passport_v2",
        document_type=DocumentType.PASSPORT,
        portrait_expected=True,
        fields=[
            _f("surname", "name", required=True),
            _f("first_name", "name", required=True),
            _f("middle_name", "name"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("passport_number", "number", required=True,
               validator=validate_passport_number),
            _f("nationality", "text"),
            _f("place_of_birth", "text"),
            _f("issue_date", "date"),
            _f("expiry_date", "date", required=True, validator=validate_expiry_not_past),
            _f("issuing_authority", "text"),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # FRSC National Driver's Licence (Nigeria)
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - FRSC driver's-licence verification portal (ndlverification.frsc.gov.ng)
    #   - DoJ EOIR "Features of the Nigerian driver's licence" (NGA102509.E.pdf)
    #     — explicitly lists: licence holder's name, address, blood group,
    #     facial marks, sex, glasses, date of birth, height, issue date,
    #     expiry date, issuing state of current + first licence, bar code,
    #     signature.
    DocumentType.DRIVERS_LICENSE: DocumentSchema(
        schema_id="ng_drivers_license_v2",
        document_type=DocumentType.DRIVERS_LICENSE,
        portrait_expected=True,
        fields=[
            _f("surname", "name", required=True),
            _f("first_name", "name", required=True),
            _f("middle_name", "name"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("drivers_license_number", "number", required=True),
            _f("issue_date", "date"),
            _f("expiry_date", "date", required=True, validator=validate_expiry_not_past),
            _f("address", "address"),
            _f("issuing_state", "text"),
            _f("blood_group", "text"),
            _f("height", "text"),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # INEC Permanent Voter Card (PVC)
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - INEC factsheet via Situation Room (situationroomng.org)
    #   - Dubawa "All you need to know about your PVC" — explicit field list:
    #     polling unit (PU) code, photo, full name, date of birth, gender,
    #     occupation, address (front); barcode, serial number, first name +
    #     VIN + registration date (back).
    DocumentType.VOTER_ID: DocumentSchema(
        schema_id="ng_voter_id_v2",
        document_type=DocumentType.VOTER_ID,
        portrait_expected=True,
        sides=("front", "back"),
        fields=[
            _f("full_name", "name", required=True),
            _f("date_of_birth", "date"),
            _f("gender", "text"),
            _f("occupation", "text"),
            _f("address", "address", required=True),
            _f("document_number", "number", required=True,  # VIN
               description="Voter Identification Number (VIN)"),
            _f("polling_unit_code", "code"),
            _f("registration_date", "date"),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # Bank Account Opening Form / Mandate
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - CBN BVN regulatory framework (cbn.gov.ng/Out/2021/CCD/REVISED ...)
    #   - CBN tier 1/2/3 KYC mandates (Dec 2023): BVN and/or NIN required.
    #   - Dojah "Navigating Nigeria's New KYC Rules" — BVN + NIN linkage.
    #   - Standard Nigerian KYC checklist: full name, DOB, address, BVN, NIN,
    #     phone, signature (NUBAN comes after account is opened).
    DocumentType.BANK_MANDATE: DocumentSchema(
        schema_id="ng_bank_mandate_v2",
        document_type=DocumentType.BANK_MANDATE,
        portrait_expected=False,
        fields=[
            _f("full_name", "name", required=True),
            _f("bvn", "number", required=True, validator=validate_bvn),
            _f("nin", "number", validator=validate_nin),
            _f("account_number", "number"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("phone_number", "phone"),
            _f("address", "address", required=True),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # Electricity Utility Bill (IKEDC / EKEDC / AEDC / IBEDC / etc.)
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - IBEDC "View Bills" page (ibedc.com/view-bills)
    #   - EKEDC customer portal (ekedp.com)
    #   - AEDC customer info portal (infocheck.abujaelectricity.com) — they all
    #     show: customer name, service address, account number (postpaid) or
    #     meter number (prepaid), billing period, units consumed, amount due,
    #     due date, disco name (issuing authority).
    DocumentType.UTILITY_BILL: DocumentSchema(
        schema_id="ng_utility_bill_v2",
        document_type=DocumentType.UTILITY_BILL,
        portrait_expected=False,
        fields=[
            _f("full_name", "name", required=True),
            _f("address", "address", required=True),
            _f("account_number", "number"),
            _f("meter_number", "number"),
            _f("billing_period", "text"),
            _f("amount_due", "money"),
            _f("due_date", "date"),
            _f("issuing_authority", "text", required=True),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # Bank Statement of Account
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - CBN NUBAN standard (cbn.gov.ng) — 10-digit account number.
    #   - First Bank, Oze, GTBank statement formats — header carries: bank name,
    #     branch, account holder name, account number (NUBAN), address,
    #     statement period (from-to), currency, opening + closing balance.
    DocumentType.BANK_STATEMENT: DocumentSchema(
        schema_id="ng_bank_statement_v2",
        document_type=DocumentType.BANK_STATEMENT,
        portrait_expected=False,
        fields=[
            _f("full_name", "name", required=True),
            _f("account_number", "number", required=True),
            _f("issuing_authority", "text", required=True,  # bank name
               description="Bank name"),
            _f("branch_name", "text"),
            _f("address", "address"),
            _f("statement_period", "text"),
            _f("opening_balance", "money"),
            _f("closing_balance", "money"),
        ],
    ),

    # ---------------------------------------------------------------------- #
    # CAC Certificate (Incorporation / Business Name)
    # ---------------------------------------------------------------------- #
    # Sources:
    #   - CAC public search portal (icrp.cac.gov.ng/public-search)
    #   - Korapay "Certificate of Incorporation (CAC) Verification" docs
    #   - useflexfinance / 9jadirectory guides — certificate shows: registered
    #     name, registration number (RC for companies, BN for business names,
    #     IT for incorporated trustees, LP for limited partnerships), date of
    #     incorporation/registration, registered address.
    DocumentType.CAC_DOCUMENT: DocumentSchema(
        schema_id="ng_cac_certificate_v1",
        document_type=DocumentType.CAC_DOCUMENT,
        portrait_expected=False,
        fields=[
            _f("company_name", "name", required=True),
            _f("registration_number", "number", required=True,  # RC/BN/IT/LP
               description="RC / BN / IT / LP registration number"),
            _f("incorporation_date", "date", required=True),
            _f("address", "address"),
        ],
    ),
}


def schema_for(document_type: DocumentType) -> Optional[DocumentSchema]:
    return SCHEMAS.get(document_type)

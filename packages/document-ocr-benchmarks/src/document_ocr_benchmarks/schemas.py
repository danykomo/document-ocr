"""Nigerian document field schemas.

Each schema declares the fields a document type should yield, which are
required, the validation rule for each, and whether a portrait is expected.
The harness uses these to (a) tell providers what to extract and (b) compute
required-field recall and portrait expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    kind: str  # name | date | number | address | text
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
    DocumentType.NIN_SLIP: DocumentSchema(
        # Real NIN slips print Surname / First Name / Middle Name as separate
        # lines — there is no combined "full name" on the document. The harness
        # still synthesises full_name from the parts in _coerce_keys when a
        # caller wants it, but it isn't a scoreable field here.
        schema_id="ng_nin_slip_v2",
        document_type=DocumentType.NIN_SLIP,
        portrait_expected=True,
        fields=[
            _f("surname", "name", required=True),
            _f("first_name", "name", required=True),
            _f("middle_name", "name"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("nin", "number", required=True, validator=validate_nin),
            _f("address", "address"),
            _f("issue_date", "date"),
        ],
    ),
    DocumentType.NATIONAL_ID: DocumentSchema(
        schema_id="ng_national_id_v1",
        document_type=DocumentType.NATIONAL_ID,
        portrait_expected=True,
        sides=("front", "back"),
        fields=[
            _f("full_name", "name", required=True),
            _f("surname", "name"),
            _f("first_name", "name"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("nin", "number", required=True, validator=validate_nin),
            _f("document_number", "number"),
            _f("expiry_date", "date", validator=validate_expiry_not_past),
        ],
    ),
    DocumentType.PASSPORT: DocumentSchema(
        schema_id="ng_passport_v1",
        document_type=DocumentType.PASSPORT,
        portrait_expected=True,
        fields=[
            _f("full_name", "name", required=True),
            _f("surname", "name", required=True),
            _f("first_name", "name"),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("passport_number", "number", required=True,
               validator=validate_passport_number),
            _f("issue_date", "date"),
            _f("expiry_date", "date", required=True, validator=validate_expiry_not_past),
            _f("issuing_authority", "text"),
        ],
    ),
    DocumentType.DRIVERS_LICENSE: DocumentSchema(
        schema_id="ng_drivers_license_v1",
        document_type=DocumentType.DRIVERS_LICENSE,
        portrait_expected=True,
        fields=[
            _f("full_name", "name", required=True),
            _f("date_of_birth", "date", required=True),
            _f("gender", "text"),
            _f("drivers_license_number", "number", required=True),
            _f("issue_date", "date"),
            _f("expiry_date", "date", required=True, validator=validate_expiry_not_past),
            _f("address", "address"),
            _f("issuing_authority", "text"),
        ],
    ),
    DocumentType.VOTER_ID: DocumentSchema(
        schema_id="ng_voter_id_v1",
        document_type=DocumentType.VOTER_ID,
        portrait_expected=True,
        fields=[
            _f("full_name", "name", required=True),
            _f("date_of_birth", "date"),
            _f("gender", "text"),
            _f("document_number", "number", required=True),
            _f("address", "address"),
        ],
    ),
    DocumentType.BANK_MANDATE: DocumentSchema(
        schema_id="ng_bank_mandate_v1",
        document_type=DocumentType.BANK_MANDATE,
        portrait_expected=False,
        fields=[
            _f("full_name", "name", required=True),
            _f("bvn", "number", validator=validate_bvn),
            _f("nin", "number", validator=validate_nin),
            _f("account_number", "number"),
            _f("date_of_birth", "date"),
            _f("address", "address"),
        ],
    ),
    DocumentType.UTILITY_BILL: DocumentSchema(
        schema_id="ng_utility_bill_v1",
        document_type=DocumentType.UTILITY_BILL,
        portrait_expected=False,
        fields=[
            _f("full_name", "name", required=True),
            _f("address", "address", required=True),
            _f("issue_date", "date"),
            _f("issuing_authority", "text"),
        ],
    ),
    DocumentType.BANK_STATEMENT: DocumentSchema(
        schema_id="ng_bank_statement_v1",
        document_type=DocumentType.BANK_STATEMENT,
        portrait_expected=False,
        fields=[
            _f("full_name", "name", required=True),
            _f("account_number", "number", required=True),
            _f("address", "address"),
            _f("issue_date", "date"),
            _f("issuing_authority", "text"),
        ],
    ),
}


def schema_for(document_type: DocumentType) -> Optional[DocumentSchema]:
    return SCHEMAS.get(document_type)

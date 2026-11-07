"""Grounding and validation for scriptwriter."""

from app.agents.fact_checker.schemas import VerificationStatus


def is_safe_for_use(claim_status: VerificationStatus) -> bool:
    if claim_status == VerificationStatus.VERIFIED:
        return True
    if claim_status == VerificationStatus.PARTIALLY_VERIFIED:
        return True  # but must be qualified
    return False


def validate_claim_references(claim_refs, valid_claim_ids: set, valid_source_ids: set):
    for ref in claim_refs:
        claim_id = ref.get("claim_id") if isinstance(ref, dict) else getattr(ref, "claim_id", None)
        if claim_id not in valid_claim_ids:
            raise ValueError(f"Invalid claim reference: {claim_id}")
        for sid in (ref.get("source_ids", ()) if isinstance(ref, dict) else getattr(ref, "source_ids", ())):
            if sid not in valid_source_ids:
                raise ValueError(f"Invalid source reference: {sid}")
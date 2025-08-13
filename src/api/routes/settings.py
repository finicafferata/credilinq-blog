from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json

from ...config.database import db_config


router = APIRouter()


class LinkItem(BaseModel):
    label: str = Field(..., max_length=200)
    url: str = Field(..., max_length=1000)


class CompanyProfile(BaseModel):
    companyName: Optional[str] = Field(None, max_length=200)
    companyContext: str = Field(..., max_length=10000)
    brandVoice: Optional[str] = Field(None, max_length=5000)
    valueProposition: Optional[str] = Field(None, max_length=5000)
    industries: List[str] = Field(default_factory=list, max_items=10)
    targetAudiences: List[str] = Field(default_factory=list, max_items=10)
    tonePresets: List[str] = Field(default_factory=list, max_items=10)
    keywords: List[str] = Field(default_factory=list, max_items=50)
    styleGuidelines: Optional[str] = Field(None, max_length=10000)
    prohibitedTopics: List[str] = Field(default_factory=list, max_items=50)
    complianceNotes: Optional[str] = Field(None, max_length=10000)
    links: List[LinkItem] = Field(default_factory=list, max_items=20)
    defaultCTA: Optional[str] = Field(None, max_length=1000)
    updatedAt: Optional[str] = None


def _ensure_table():
    """Create the app_settings table if it's missing (idempotent)."""
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
    except Exception:
        # Let callers handle DB connectivity errors; we only ensure existence.
        pass


def _upsert_setting(key: str, value: dict):
    _ensure_table()
    with db_config.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO app_settings(key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key)
            DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, json.dumps(value)),
        )
        conn.commit()


def _get_setting(key: str) -> Optional[dict]:
    _ensure_table()
    with db_config.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
        row = cur.fetchone()
        if not row:
            return None
        # row is a RealDictRow or tuple depending on cursor, support both
        raw = row[0] if isinstance(row, tuple) else row.get("value")
        try:
            return raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            return {}


def _build_company_profile_markdown(profile: CompanyProfile) -> str:
    parts: List[str] = []
    parts.append(f"# Company Profile\n")
    if profile.companyName:
        parts.append(f"**Company**: {profile.companyName}\n\n")
    parts.append("## Company Context\n")
    parts.append((profile.companyContext or '').strip() + "\n\n")
    if profile.valueProposition:
        parts.append("## Value Proposition\n")
        parts.append(profile.valueProposition.strip() + "\n\n")
    if profile.brandVoice:
        parts.append("## Brand Voice\n")
        parts.append(profile.brandVoice.strip() + "\n\n")
    if profile.industries:
        parts.append("## Industries\n")
        parts.append("- " + "\n- ".join(profile.industries) + "\n\n")
    if profile.targetAudiences:
        parts.append("## Target Audiences\n")
        parts.append("- " + "\n- ".join(profile.targetAudiences) + "\n\n")
    if profile.tonePresets:
        parts.append("## Tone Presets\n")
        parts.append("- " + "\n- ".join(profile.tonePresets) + "\n\n")
    if profile.keywords:
        parts.append("## Brand Keywords\n")
        parts.append(", ".join(profile.keywords) + "\n\n")
    if profile.styleGuidelines:
        parts.append("## Style Guidelines\n")
        parts.append(profile.styleGuidelines.strip() + "\n\n")
    if profile.prohibitedTopics:
        parts.append("## Prohibited Topics\n")
        parts.append("- " + "\n- ".join(profile.prohibitedTopics) + "\n\n")
    if profile.complianceNotes:
        parts.append("## Compliance Notes\n")
        parts.append(profile.complianceNotes.strip() + "\n\n")
    if profile.links:
        parts.append("## Useful Links\n")
        for link in profile.links:
            parts.append(f"- [{link.label}]({link.url})\n")
        parts.append("\n")
    if profile.defaultCTA:
        parts.append("## Default Call To Action\n")
        parts.append(profile.defaultCTA.strip() + "\n\n")
    return "".join(parts)


def _sync_company_profile_to_kb(profile: CompanyProfile):
    """Create or update a 'Company Profile' document in the Knowledge Base."""
    # Create a temp markdown file to reuse the existing pipeline
    import tempfile, os, uuid
    title = "Company Profile"
    content = _build_company_profile_markdown(profile)
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"company-profile-{uuid.uuid4().hex}.md")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Insert document row and schedule processing similar to upload flow
    doc_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    try:
        with db_config.get_db_connection() as conn:
            cur = conn.cursor()
            # Insert into existing documents table
            cur.execute(
                'INSERT INTO documents (id, title, "storagePath", "uploadedAt") VALUES (%s, %s, %s, %s)',
                (doc_id, title, file_path, created_at)
            )
            conn.commit()
    except Exception:
        # If document table is managed differently in the deployment env, skip the insert silently
        pass

    # Process synchronously here to guarantee immediate availability
    from ...agents.specialized.document_processor import DocumentProcessorAgent
    processor = DocumentProcessorAgent()
    try:
        processor.execute_safe({
            "document_id": doc_id,
            "file_path": file_path
        })
    except Exception:
        # Don't fail settings update if KB sync fails
        pass


def _normalize_profile_dict(data: dict) -> dict:
    """Normalize user-provided data to avoid validation errors and keep consistency."""
    def _clean_list(values, lower=False, drop_no_tokens=True):
        if not isinstance(values, list):
            return []
        cleaned = []
        for v in values:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            if drop_no_tokens and s.lower() in {"no", "none", "n/a"}:
                continue
            cleaned.append(s.lower() if lower else s)
        # de-duplicate preserving order
        seen = set()
        out = []
        for s in cleaned:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    # Normalize arrays
    for key in ["industries", "targetAudiences", "tonePresets", "keywords", "prohibitedTopics"]:
        data[key] = _clean_list(data.get(key, []))

    # Links: filter invalid and add https:// if missing
    links = []
    for link in data.get("links", []) or []:
        try:
            label = (link.get("label") or "").strip()
            url = (link.get("url") or "").strip()
            if not label or not url:
                continue
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            links.append({"label": label, "url": url})
        except Exception:
            continue
    data["links"] = links

    # Strings: trim
    for key in ["companyName", "companyContext", "brandVoice", "valueProposition", "styleGuidelines", "complianceNotes", "defaultCTA"]:
        if key in data and isinstance(data[key], str):
            data[key] = data[key].strip()
    return data


@router.get("/settings/company-profile")
def get_company_profile() -> CompanyProfile:
    data = _get_setting("company_profile") or {}
    # Provide minimal defaults so frontend can prefill
    if "companyContext" not in data:
        data["companyContext"] = ""
    return CompanyProfile(**data)


@router.put("/settings/company-profile")
def update_company_profile(profile: CompanyProfile):
    payload = _normalize_profile_dict(profile.model_dump())
    payload["updatedAt"] = datetime.utcnow().isoformat()
    try:
        _upsert_setting("company_profile", payload)
        # Sync to Knowledge Base as a visible document
        _sync_company_profile_to_kb(CompanyProfile(**payload))
        return {"message": "Company profile updated", "updatedAt": payload["updatedAt"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")


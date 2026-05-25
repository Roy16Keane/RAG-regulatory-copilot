from __future__ import annotations

import json
import re
from typing import Dict, List

from openai import OpenAI

from api.app.core.config import settings


client = OpenAI(api_key=settings.OPENAI_API_KEY)


ENTITY_SCHEMA_KEYS = [
    "organisations",
    "people_roles",
    "standards",
    "regulations",
    "risks",
    "controls",
    "procedures",
    "equipment",
    "locations",
    "dates",
    "key_terms",
]


def fallback_entity_extraction(text: str) -> Dict[str, List[str]]:
    return {
        "organisations": [],
        "people_roles": [],
        "standards": sorted(set(re.findall(r"\b(?:ISO|IEC|BS|EN|GDPR|MiFID|Basel|FCA|HSE)[\w\-\/: ]*\b", text)))[:30],
        "regulations": [],
        "risks": [],
        "controls": [],
        "procedures": [],
        "equipment": [],
        "locations": [],
        "dates": sorted(set(re.findall(r"\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)))[:30],
        "key_terms": [],
    }


def extract_entities_llm(text: str, filename: str, page: int | None = None) -> Dict:
    """
    Extract structured document intelligence metadata.
    Runs before chunk indexing.
    """

    text = text[:12000]

    system_prompt = """
You are a document intelligence metadata extractor for a RAG system.

Return ONLY valid JSON in this exact structure:

{
  "document_type": "",
  "summary": "",
  "section_titles": [],
  "entities": {
    "organisations": [],
    "people_roles": [],
    "standards": [],
    "regulations": [],
    "risks": [],
    "controls": [],
    "procedures": [],
    "equipment": [],
    "locations": [],
    "dates": [],
    "key_terms": []
  }
}
"""

    user_prompt = f"""
Filename: {filename}
Page: {page}

Text:
{text}
"""

    try:
        res = client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
        )

        content = res.choices[0].message.content
        parsed = json.loads(content)

        entities = parsed.get("entities", {})
        for key in ENTITY_SCHEMA_KEYS:
            entities.setdefault(key, [])

        parsed["entities"] = entities
        parsed.setdefault("section_titles", [])
        parsed.setdefault("summary", "")
        parsed.setdefault("document_type", "unknown")

        return parsed

    except Exception:
        return {
            "document_type": "unknown",
            "summary": "",
            "section_titles": [],
            "entities": fallback_entity_extraction(text),
        }


def merge_entities(entity_dicts: List[Dict]) -> Dict[str, List[str]]:
    merged = {key: [] for key in ENTITY_SCHEMA_KEYS}

    for item in entity_dicts:
        entities = item.get("entities", {})
        for key in ENTITY_SCHEMA_KEYS:
            values = entities.get(key, [])
            if isinstance(values, list):
                merged[key].extend(str(v) for v in values if v)

    return {
        key: sorted(set(values))
        for key, values in merged.items()
    }
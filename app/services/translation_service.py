
"""Translate recommendation text without changing recommendation logic."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request


SUPPORTED_LANGUAGES = {
    "en": "English",
    "kn": "Kannada",
    "te": "Telugu",
    "ta": "Tamil",
}

TRANSLATABLE_FIELDS = {
    "summary",
    "recommendation_text",
    "immediate_actions",
    "organic_management",
    "avoid",
    "message",
    "growth_stage_note",
    "organic_disclaimer",
}

_TRANSLATION_URL = (
    "https://translation.googleapis.com/language/translate/v2"
)

# In-memory cache: avoids repeat API charges for identical text
# during the lifetime of this backend process.
_translation_cache = {}


def _translate_batch(texts, target_language):
    """Translate a batch of English strings using Google Cloud."""

    if not texts:
        return []

    api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Google Translation is not configured. "
            "Set GOOGLE_TRANSLATE_API_KEY on the backend."
        )

    params = [
        ("key", api_key),
        ("source", "en"),
        ("target", target_language),
        ("format", "text"),
    ]

    params.extend(("q", text) for text in texts)

    url = _TRANSLATION_URL + "?" + urllib.parse.urlencode(params)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SmartCrop-AI"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(
            "Translation service is temporarily unavailable."
        ) from exc

    translations = (
        payload.get("data", {}).get("translations", [])
    )

    if len(translations) != len(texts):
        raise RuntimeError(
            "Translation service returned an unexpected response."
        )

    return [
        item["translatedText"]
        for item in translations
    ]


def translate_recommendation_plan(plan, target_language):
    """
    Return a translated copy of approved recommendation text.

    Disease labels, identifiers, scores, measurements, priorities,
    and all other non-approved fields are preserved unchanged.
    """

    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported language.")

    if not isinstance(plan, dict):
        raise ValueError("Recommendation plan must be an object.")

    if target_language == "en":
        return dict(plan)

    texts = []

    def collect(value, parent_key=None):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in TRANSLATABLE_FIELDS:
                    collect(item, key)
                elif isinstance(item, dict):
                    collect(item)
        elif parent_key in TRANSLATABLE_FIELDS:
            if isinstance(value, str) and value.strip():
                texts.append(value)
            elif isinstance(value, list):
                texts.extend(
                    item for item in value
                    if isinstance(item, str) and item.strip()
                )

    collect(plan)

    unique_missing = list(dict.fromkeys(
        text for text in texts
        if (target_language, text) not in _translation_cache
    ))

    # Keep each API request at a manageable batch size.
    for start in range(0, len(unique_missing), 100):
        batch = unique_missing[start:start + 100]
        translated = _translate_batch(batch, target_language)

        for original, result in zip(batch, translated):
            _translation_cache[(target_language, original)] = result

    def translate_value(value):
        if isinstance(value, str):
            if not value.strip():
                return value
            return _translation_cache.get(
                (target_language, value), value
            )

        if isinstance(value, list):
            return [
                translate_value(item)
                if isinstance(item, str) and item.strip()
                else item
                for item in value
            ]

        return value

    def rebuild(value):
        if isinstance(value, dict):
            result = {}

            for key, item in value.items():
                if key in TRANSLATABLE_FIELDS:
                    result[key] = translate_value(item)
                elif isinstance(item, dict):
                    result[key] = rebuild(item)
                else:
                    result[key] = item

            return result

        return value

    return rebuild(plan)

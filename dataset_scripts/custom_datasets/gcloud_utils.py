from typing import List
from google.cloud import translate


def translate_text(
    list_text: List[str] = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "YOUR_PROJECT_ID",
    source_language_code: str = "en-US", target_language_code: str = "vi"
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": list_text,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
        }
    )

    return response

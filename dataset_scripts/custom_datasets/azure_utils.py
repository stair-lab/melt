import json
import uuid
from typing import List

import requests

# Add your key and endpoint
key = "52ceb493d16e485ea561c200254862f3"
endpoint = "https://api.cognitive.microsofttranslator.com"

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = "southeastasia"

path = "/translate"
constructed_url = endpoint + path

headers = {
    "Ocp-Apim-Subscription-Key": key,
    # location required if you're using a multi-service or regional (not global) resource.
    "Ocp-Apim-Subscription-Region": location,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}

# You can pass more than one object in body.


def translate_text(
    list_text: List[str] = "YOUR_TEXT_TO_TRANSLATE",
    source_language_code: str = "en",
    target_language_code: str = "vi",
    *args,
    **kwargs
):
    params = {
        "api-version": "3.0",
        "from": source_language_code,
        "to": [target_language_code],
    }
    body = [
        {
            "text": text,
        }
        for text in list_text
    ]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    results = [x["translations"][0]["text"] for x in response]
    return results


if __name__ == "__main__":
    print(translate_text(["Hello world", "I love you"]))

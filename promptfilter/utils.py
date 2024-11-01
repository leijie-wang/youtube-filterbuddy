from openai import OpenAI
from django.conf import settings
import logging
import re


logger = logging.getLogger(__name__)

class ChatCompletion:

    def __init__(self):
        self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def chat_completion(self, system_prompt, user_prompt, type="json_object"):
        response = self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": type},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )
        answer = response.choices[0].message.content
    
        return answer

    def extract_xml(self, xml_str, tag):
    # extract the value of the tag from the xml string using regular expression
        pattern = re.compile(fr"<{tag}>(.*?)</{tag}>", re.DOTALL)
        result = pattern.findall(xml_str)
        return result[0].strip() if len(result) > 0 else None


def credentials_to_dict(credentials):
  return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
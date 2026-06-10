import requests
import json
import os
import sys


class _Error(Exception):
    pass


class GPTError(_Error):
    pass


def complete_4o(input_str):
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "gpt-4o",
        "temperature": 0,
    }
    url = os.environ["OPENAI_URL"]
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_TOKEN')}",
        "Content-Type": "application/json",
    }
    payload["messages"][0]["content"] = input_str
    res = requests.post(url, json=payload, headers=headers)
    if res.status_code != 200:
        raise GPTError(f"Failed to complete: {res.content}, {res.status_code}")
    ret = json.loads(res.content)
    return ret["choices"][0]["message"]["content"]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    msg = sys.stdin.read()
    print(complete_4o(msg))

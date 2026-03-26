import numpy as np
import json
import time
import sys
import os

try:
    import google.generativeai as genai
except Exception:
    genai = None
from anthropic import Anthropic

# OpenAI client for v1.x API (set by set_openai_key)
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY")}
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")
        _openai_client = OpenAI(**kwargs)
    return _openai_client

def get_openai_embedding(texts, model="text-embedding-ada-002"):
   client = get_openai_client()
   texts = [text.replace("\n", " ") for text in texts]
   r = client.embeddings.create(input=texts, model=model)
   return np.array([r.data[i].embedding for i in range(len(texts))])

def set_anthropic_key():
    pass

def set_gemini_key():
    if genai is None:
        raise ImportError("google.generativeai not installed")
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def set_openai_key():
    get_openai_client()


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            output = output.replace('json', '') # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):

    if model_name == 'claude-sonnet':
        model_name = "claude-3-sonnet-20240229"
    elif model_name == 'claude-haiku':
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0):

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
                model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):
    from openai import APIError, APIConnectionError, RateLimitError

    client = get_openai_client()
    completion = None
    while completion is None:
        wait_time = wait_time + 2
        try:
            # if model == 'chatgpt':
            #     messages = [{"role": "system", "content": query}]
            #     completion = client.chat.completions.create(
            #         model="gpt-3.5-turbo",
            #         temperature=temperature,
            #         max_tokens=num_tokens_request,
            #         n=num_gen,
            #         messages=messages
            #     )
            # else:
            messages = [{"role": "user", "content": query}]
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages
            )
        except APIError as e:
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except RateLimitError as e:
            print(f"OpenAI API rate limit: {e}")
            time.sleep(wait_time)
            pass
        except Exception as e:
            if "ServiceUnavailable" in type(e).__name__ or "503" in str(e):
                print(f"OpenAI API unavailable: {e}; waiting for {wait_time} seconds")
                time.sleep(wait_time)
                pass
            else:
                raise

    if model == 'davinci':
        outputs = [c.text.strip() for c in completion.choices]
        return outputs if num_gen > 1 else outputs[0]
    return completion.choices[0].message.content


def run_chatgpt_multimodal(text_prompt, image_paths, model='gpt-4o-mini',
                           num_tokens_request=1024, temperature=0.0, wait_time=1):
    """Call OpenAI-compatible API with text + images (vision)."""
    import base64
    from openai import APIError, APIConnectionError, RateLimitError

    client = get_openai_client()

    content = [{"type": "text", "text": text_prompt}]
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lower()
        mime = mime_map.get(ext, "image/png")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
        })

    messages = [{"role": "user", "content": content}]

    completion = None
    while completion is None:
        wait_time += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=num_tokens_request,
                messages=messages
            )
        except (APIError, APIConnectionError, RateLimitError) as e:
            print(f"Multimodal API error: {e}; waiting {wait_time}s")
            time.sleep(wait_time)
        except Exception as e:
            if "ServiceUnavailable" in type(e).__name__ or "503" in str(e):
                print(f"Multimodal API unavailable: {e}; waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                raise
    return completion.choices[0].message.content


def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):
    from openai import APIError, APIConnectionError, RateLimitError

    client = get_openai_client()
    completion = None
    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "system", "content": out})
    messages.append({"role": "user", "content": input})

    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=messages
            )
        except APIError as e:
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except RateLimitError as e:
            print(f"OpenAI API rate limit: {e}")
            time.sleep(wait_time)
            pass
        except Exception as e:
            if "ServiceUnavailable" in type(e).__name__ or "503" in str(e):
                print(f"OpenAI API unavailable: {e}; waiting for {wait_time} seconds")
                time.sleep(wait_time)
                pass
            else:
                raise

    return completion.choices[0].message.content

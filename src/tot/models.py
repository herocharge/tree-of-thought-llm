import os
import openai
import backoff 
import google.generativeai as genai

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key != "":
        genai.configure(api_key=api_key)
    else:
        print("Warning: [OPENAI|GEMINI]_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if model.startswith("gemini"):
        return gemini(prompt, model, temperature, max_tokens, n, stop)
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-4"):
    if backend.startswith("gemini"):
        return {"completion_tokens": -1, "prompt_tokens": -1, "cost": -1}
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


def gemini(prompt, model="gemini-1.5-flash", temperature=0.7, max_tokens=1000, n=1, stop=None):
    model = genai.GenerativeModel(model)
    outputs = []
    while n > 0:
        cnt = min(n, 8)
        n -= cnt
        res = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=cnt,
                stop_sequences=stop,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        # res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([(choice.content.parts[0].text) for choice in res.candidates])

    return (outputs)
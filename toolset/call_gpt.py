from openai import OpenAI
from toolset.config import OPENAI_API_KEY, GPT_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)

# Web search is part of the pipeline: the Literature Agent is retrieval-
# augmented by design (§2.2) and the name/structure lookups browse PubChem and
# RCSB. The evaluation scripts switch it off for their runs, because asked to
# reduce the hepatotoxicity of a named withdrawn drug the model could look up
# the literature on that exact compound, and the reported gain would partly
# measure retrieval rather than chemical reasoning. Set this to False to make
# every call in the process search-free:
#     import toolset.call_gpt as call_gpt
#     call_gpt.WEB_SEARCH_DEFAULT = False
WEB_SEARCH_DEFAULT = True


def gpt_api(prompt: str, model: str = GPT_MODEL, web_search: bool = None) -> str:
    """
    Call GPT, with web search per `web_search` (default: WEB_SEARCH_DEFAULT).
    """
    use_search = WEB_SEARCH_DEFAULT if web_search is None else web_search
    kwargs = {"model": model, "input": prompt, "temperature": 0}
    if use_search:
        kwargs["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**kwargs)

    return response.output_text.strip()

def parse_json_response(text):
    """
    Parse a JSON object out of an LLM reply, or None if nothing parses.

    The toolset prompts ask for ```json, and JSON's null/true/false are not
    Python literals, so ast.literal_eval raises "malformed node or string" on
    a perfectly valid reply -- which the structure-lookup prompts provoke
    deliberately by asking for null values when no structure exists. json.loads
    is therefore tried first, with literal_eval kept as the fallback for
    Python-style dicts. The code fence is optional: an unfenced reply, or one
    wrapped in a bare ```, still parses.
    """
    import json
    from ast import literal_eval

    if not isinstance(text, str):
        return None

    candidates = []
    for fence in ("```json", "```"):
        if fence in text:
            for part in text.split(fence)[1:]:
                candidates.append(part.split("```")[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1].strip())
    candidates.append(text.strip())

    for candidate in candidates:
        if not candidate:
            continue
        for parser in (json.loads, literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return None

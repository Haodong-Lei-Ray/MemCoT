def rag_view_agent_prompt(query_information, known_information,rag_results_text,benchmark):
    if benchmark == 'locomo' or benchmark == 'openclaw':
        prompt = f"""
{query_information}
{known_information}
RAG retrieval results:
{rag_results_text}

Task: Identify useful retrieval chunks based on "Constraint Matching".

Analysis Rules:
1. **Temporal Check**: If the query mentions a specific time or duration, ANY result mentioning a matching time/duration is HIGHLY relevant.
2. **Descriptive Matches**: If the query asks for a specific name (e.g., a place or person), but the text only provides a generic description (e.g., "my place", "that person", "my origin"), mark it as USEFUL. It confirms the context.
3. **Partial Information**: Do not discard a result just because it lacks the final answer. If it provides the *background* or *cause* of the queried event, it is useful.

Output a JSON object:
1. thinking: A brief explanation covering, including Why specific IDs were selected (mention the matching time or description).
3. missing_information: What specific missing_information is still missing. You can analyze the disadvantage of search_query.
4. useful_ids: List of indices (e.g. [1, 2]). Include IDs that contain matching timeframes or descriptions of the target, even if the specific name is missing.

Output ONLY valid JSON."""
    elif benchmark == 'longmemeval':
        prompt = f"""
{query_information}
{known_information}
RAG retrieval results:
{rag_results_text}

Task: Identify useful retrieval chunks based on "Constraint Matching".

Analysis Rules:
1. **Temporal Check**: If the query mentions a specific time or duration, ANY result mentioning a matching time/duration is HIGHLY relevant.
2. **Descriptive Matches**: If the query asks for a specific name (e.g., a place or person), but the text only provides a generic description (e.g., "my place", "that person", "my origin"), mark it as USEFUL. It confirms the context.
3. **Partial Information**: Do not discard a result just because it lacks the final answer. If it provides the *background* or *cause* of the queried event, it is useful.

Output a JSON object:
1. thinking: A brief explanation covering, including Why specific IDs were selected (mention the matching time or description).
3. missing_information: What specific missing_information is still missing. You can analyze the disadvantage of search_query.
4. useful_ids: List of indices (e.g. [1, 2]). Include IDs that contain matching timeframes or descriptions of the target, even if the specific name is missing.

Output ONLY valid JSON."""
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return prompt

def middle_view_agent_prompt(query_information, known_information,middle_context_text,benchmark):
    if benchmark == 'locomo' or benchmark == 'openclaw':
        prompt = f"""
{query_information}
{known_information}
{middle_context_text}

Task: Identify useful information to answer the query.

Analysis Rules:
1. **Temporal Check**: If the query mentions a specific time or duration, ANY result mentioning a matching time/duration is HIGHLY relevant.
1. **Descriptive Matches**: If the query asks for a specific name but the text only provides a generic description, mark it as USEFUL.
2. **Contextual Clues**: Surrounding turns may provide context, causation, or temporal references that help answer the query.

Output a JSON object:
1. thinking: Your reasoning about what useful information these context windows contain.
2. thinking_choice:  Why specific IDs were selected (mention the matching time or description); 
3. missing_information: What specific information is still missing to fully answer the query.
4. useful_ids: List of 0-based indices (e.g. [0, 2]) from the context windows that are useful.

Output ONLY valid JSON."""
    elif benchmark == 'longmemeval':
        prompt = f"""
{query_information}
{known_information}
{middle_context_text}

Task: Identify useful information to answer the query".

Analysis Rules:
1. **Temporal Check**: If the query mentions a specific time or duration, ANY result mentioning a matching time/duration is HIGHLY relevant.
1. **Descriptive Matches**: If the query asks for a specific name but the text only provides a generic description, mark it as USEFUL.
2. **Contextual Clues**: Surrounding turns may provide context, causation, or temporal references that help answer the query.

Output a JSON object:
1. thinking: Your reasoning about what useful information these context windows contain.
2. thinking_choice:  Why specific IDs were selected (mention the matching time or description); 
3. missing_information: What specific information is still missing to fully answer the query.
4. useful_ids: List of 0-based indices (e.g. [0, 2]) from the context windows that are useful.

Output ONLY valid JSON."""
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return prompt

def visual_ocr_agent_prompt(query_information, known_information,rag_information,session_list_str,benchmark):
    if benchmark == 'locomo' or benchmark == 'openclaw':
        prompt =f"""{query_information}
{known_information}
{rag_information}

You are viewing conversation session images from sessions: {session_list_str}.
Each image is a page from a conversation PDF. Dialogue entries are formatted as:
  {{dia_id}}- {{speaker}}: {{text}}
Some entries may include embedded images with captions.

Your task:
1. Carefully read ALL dialogue content visible in the images.
2. Identify dialogue entries (by their dia_id, e.g. "D1:5") that contain information relevant to answering the query.
3. Explain your reasoning.

Output a JSON object:
{{
    "thinking": ...,
    "useful_dia_ids": ...,
}}
Output ONLY valid JSON."""
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return prompt

def observation_agent_prompt(fail_queue_information, query,short_memory_text,conv_memory_text,thinking,queries_num,benchmark):
    if benchmark == 'locomo' or benchmark == 'openclaw':
        prompt =f"""
{fail_queue_information}
Query: {query}
{short_memory_text}
{conv_memory_text}
{fail_queue_information}
Output a JSON object: 
1.{thinking}
2.useful_id: List of dia_id strings from the useful results (e.g. [0, 2]). If can_answer, include those that support the answer. If not, include those with relevant partial info.
3.can_answer: true if the results contain enough information to answer the query, false otherwise.
4.action: Check the **Fail query**. You can Choose only one action to generate for each new query:
    1. Break: Break down last query into sub-queries to get shorter but more exact query. if Q=[Q_A,Q_B], you can just searcg Q_A firstly. Example: When Tom arrive at Shanghai for 3 years ago-> [Tom arrive at Shanghai,3 years ago]
    2. Delete: If Root Query Q = [Q_A,Q_B] and Short Memory include Q_A, focus on Q_B and New query Q'=Q-Q_A.
    Do not let new_queries as same as and Fail query.
    You can try more type action to avoid to me the same fail query.
5.new_queries: If can_answer is false, suggest {queries_num} new queries that are more likely to retrieve the missing information. These should be focused and based on the gaps identified in the report.
Output a JSON object exactly following this structure:
{{
    "thinking": ...,
    "useful_id": ...,
    "can_answer": ...,
    "action": ...,
    "new_queries": ...,
}}
Output ONLY valid JSON."""
    elif benchmark == 'longmemeval':
        prompt =f"""
{fail_queue_information}
Query: {query}
{short_memory_text}
{conv_memory_text}
{fail_queue_information}
Output a JSON object: 
1.{thinking}
2.useful_id: List of dia_id strings from the useful results (e.g. [0, 2]). If can_answer, include those that support the answer. If not, include those with relevant partial info.
3.can_answer: true if the results contain enough information to answer the query, false otherwise.
4.action: Check the **Fail query**. You can Choose only one action to generate for each new query:
    1. Break: Break down last query into sub-queries to get shorter but more exact query. if Q=[Q_A,Q_B], you can just searcg Q_A firstly. Example: When Tom arrive at Shanghai for 3 years ago-> [Tom arrive at Shanghai,3 years ago]
    2. Delete: If Root Query Q = [Q_A,Q_B] and Short Memory include Q_A, focus on Q_B and New query Q'=Q-Q_A.
    Do not let new_queries as same as and Fail query.
    You can try more type action to avoid to me the same fail query.
5.new_queries: If can_answer is false, suggest {queries_num} new queries that are more likely to retrieve the missing information.
Output a JSON object exactly following this structure:
{{
    "thinking": ...,
    "useful_id": ...,
    "can_answer": ...,
    "action": ...,
    "new_queries": ...,
}}
Output ONLY valid JSON."""
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return prompt


def answer_agent_prompt(additional_information_text, short_memory_text, query,benchmark):
    if benchmark == 'locomo' or benchmark == 'openclaw':
        prompt =f"""
CRITICAL RULES:
1. Ultra-Concise Answer: The "answer" MUST be an extremely short entity, number, or absolute date.
2. For yes/no questions (Would/Did/Is/Does...?), answer yes or no, or the given choice. For how many, answer english word, like two,three,twice.
{additional_information_text}
{short_memory_text}
Query: {query}
Output ONLY valid JSON.
1. thinking: Thinking hard and more for the answer. Calculated absolute date from session context if possible. Extracted target entity.
2. answer: Write the answer in the form of a short phrase. Answer with exact words from the context whenever possible. Cannot be empty. You can not say"No information available".
Output a JSON object exactly following this structure:
{{
    "thinking": "...",
    "answer": "..."
}}
"""
    elif benchmark == 'longmemeval':
        prompt =f"""{additional_information_text}
{short_memory_text}
Query: {query}
1. thinking: Thinking hard and more for the answer. You need to explain for all the short memory. If you design math problems, please write the calculation steps in the "thinking" field.
2. answer: Answer the query. Cannot be empty. You can not say"No information available".
"""
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return prompt


from task_eval.gpt_utils import QA_PROMPT, QA_PROMPT_CAT_5, CONV_START_PROMPT
from .format_tool import _build_full_conv_context_from_entry
from benchmark.longmemeval.src.generation.run_generation import prepare_prompt
import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")

def conv_answer_agent_prompt(full_conv, query, benchmark):
    if benchmark == 'locomo':
        full_conv_text = full_conv
        prompt = full_conv_text + "\n\n" + QA_PROMPT.format(query)
    elif benchmark == 'openclaw':
        lines = []
        for msg in full_conv:
            date_time = msg.get("date_time", "")
            role = msg.get("role", "")
            context = msg.get("context", "")
            dt_str = f"[{date_time}] " if date_time else ""
            lines.append(f"{dt_str}{role} said: {context}")
        full_conv_text = "\n\n".join(lines)
        prompt = full_conv_text + "\n\n" + QA_PROMPT.format(query)
    elif benchmark == 'longmemeval':
        if not isinstance(full_conv, dict):
            raise ValueError("longmemeval requires full_conv as entry dict")
        if "question" not in full_conv:
            raise ValueError("longmemeval full_conv missing required field: question")
        if "question_date" not in full_conv:
            raise ValueError("longmemeval full_conv missing required field: question_date")
        prompt = prepare_prompt(
            entry=full_conv,
            retriever_type="orig-session",
            topk_context=len(full_conv.get("haystack_sessions", [])),
            useronly=False,
            history_format="nl",
            cot=False,
            tokenizer=TOKENIZER,
            tokenizer_backend="openai",
            max_retrieval_length=1000000,
            merge_key_expansion_into_value="none",
            con=False,
            con_client=None,
            con_model=None,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return prompt

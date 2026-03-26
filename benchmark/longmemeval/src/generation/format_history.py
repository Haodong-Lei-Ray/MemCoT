"""
run_generation 风格的 history 格式化（orig-session + nl，con=False）。
供 run_generation、quick_test_gpt4omini、agent 复用，保证载入格式一致。
仅依赖 json，无重依赖。
"""
import json


def format_history_for_prompt(
    entry: dict,
    max_sessions: int | None = None,
    useronly: bool = False,
    history_format: str = "nl",
) -> str:
    """将 haystack 展开为 run_generation 风格的 history 文本（orig-session + nl，con=False）。"""
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    if len(dates) != len(sessions):
        raise ValueError("haystack_dates / haystack_sessions 长度不一致")
    if max_sessions is not None:
        dates = dates[:max_sessions]
        sessions = sessions[:max_sessions]

    history_string = ""
    for i, (chunk_date, session_entry) in enumerate(zip(dates, sessions)):
        if useronly:
            session_entry = [x for x in session_entry if x.get("role") == "user"]
        if history_format == "nl":
            sess_string = ""
            for turn_entry in session_entry:
                sess_string += "\n\n{}: {}".format(
                    turn_entry.get("role", "user"),
                    str(turn_entry.get("content", "")).strip(),
                )
        elif history_format == "json":
            sess_string = "\n" + json.dumps(session_entry)
        else:
            raise ValueError("history_format 须为 'nl' 或 'json'")
        history_string += "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
            i + 1, chunk_date or "", sess_string
        )
    return history_string.strip()

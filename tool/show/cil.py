import os
import sys


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_ORANGE = "\033[38;5;214m"
ANSI_DIM = "\033[2m"
ANSI_GRAY = "\033[38;5;245m"


def _use_color() -> bool:
    # Respect NO_COLOR and avoid escape sequences in non-tty logs.
    return sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def format_age(age_ms):
    if age_ms is None:
        return "unknown"
    age_s = age_ms / 1000
    if age_s < 60:
        return f"{int(age_s)}s ago"
    age_m = age_s / 60
    if age_m < 60:
        return f"{int(age_m)}m ago"
    age_h = age_m / 60
    if age_h < 24:
        return f"{int(age_h)}h ago"
    age_d = age_h / 24
    return f"{int(age_d)}d ago"

def format_tokens(total, context):
    if total is None:
        total_str = "unknown"
        pct_str = "?"
    else:
        total_str = f"{total/1000:.0f}k" if total >= 1000 else str(total)
        pct_str = f"{int(total / context * 100)}" if context else "?"
        
    ctx_str = f"{context/1000:.0f}k" if context and context >= 1000 else str(context)
    return f"{total_str}/{ctx_str} ({pct_str}%)"

def truncate_key(key, max_len=26):
    if len(key) <= max_len:
        return key
    return key[:15] + "..." + key[-8:]

def show_session_list(data):
    print(_c("[🦉 MemCoT] Use 🦞OpenClaw [openclaw sessions --json] to retrieve session list", ANSI_ORANGE + ANSI_BOLD))
    print()
    print(f"{_c('Session store:', ANSI_ORANGE)} {_c(data.get('path', 'unknown'), ANSI_DIM)}")
    if data.get("session_file"):
        print(f"{_c('Session file:', ANSI_ORANGE)} {_c(data.get('session_file'), ANSI_DIM)}")
    sessions = data.get("sessions", [])
    print(f"{_c('Sessions listed:', ANSI_ORANGE)} {_c(str(len(sessions)), ANSI_ORANGE)}")
    
    header = f"{'Idx':<4} {'Kind':<6} {'Key':<26} {'Age':<9} {'Model':<14} {'Tokens (ctx %)':<20} {'Flags':<46} {'Rag Status'}"
    print(_c(header, ANSI_ORANGE + ANSI_BOLD))
    
    for i, s in enumerate(sessions):
        idx = s.get("index", i)
        kind = s.get("kind", "")
        key = truncate_key(s.get("key", ""), 26)
        age = format_age(s.get("ageMs"))
        model = s.get("model", "")
        tokens = format_tokens(s.get("totalTokens"), s.get("contextTokens"))
        
        flags = []
        if s.get("systemSent"):
            flags.append("system")
        if s.get("sessionId"):
            flags.append(f"id:{s.get('sessionId')}")
        flags_str = " ".join(flags)
        
        rag_status = s.get("rag_status", "unknown")
        
        idx_col = f"{str(idx):<4}"
        mid_cols = f"{kind:<6} {key:<26} {age:<9} {model:<14} {tokens:<20} {flags_str:<46}"
        rag_col = f"{rag_status}"
        print(f"{_c(idx_col, ANSI_ORANGE)} {_c(mid_cols, ANSI_GRAY)} {_c(rag_col, ANSI_ORANGE)}")

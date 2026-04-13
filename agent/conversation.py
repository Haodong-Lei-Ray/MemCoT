from pathlib import Path
import importlib.util
import json

try:
    from .. import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

from benchmark.locomo.task_eval.gpt_utils import CONV_START_PROMPT

_conversation_cache: dict = {}
LONGMEMEVAL_DATASET_DIR = PROJECT_ROOT / "benchmark" / "LongMemEval" / "dataset"
LONGMEMEVAL_FORMAT_HISTORY = (
    PROJECT_ROOT / "module_unit" / "LongMemEval" / "src" / "generation" / "format_history.py"
)
_format_history_for_prompt_fn = None

# LongMemEval
def _get_haystack_session_ids(conv_id: str) -> list[str]:
    """从 LongMemEval 单题 JSON 载入 haystack_session_ids。"""
    cache_key = conv_id
    if cache_key not in _conversation_cache:
        conv_file = LONGMEMEVAL_DATASET_DIR / f"{conv_id}.json"
        with open(conv_file, "r", encoding="utf-8") as f:
            _conversation_cache[cache_key] = json.load(f)
    return _conversation_cache[cache_key].get("haystack_session_ids", [])

def _get_format_history_for_prompt():
    """延迟加载 format_history.format_history_for_prompt（run_generation 风格，con=False）。"""
    global _format_history_for_prompt_fn
    if _format_history_for_prompt_fn is None:
        spec = importlib.util.spec_from_file_location(
            "format_history", LONGMEMEVAL_FORMAT_HISTORY
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _format_history_for_prompt_fn = mod.format_history_for_prompt
    return _format_history_for_prompt_fn

def _build_full_conv_context_longmemeval(conv_id: str) -> str:
    """载入 LongMemEval 单题 haystack 为全文上下文。"""
    cache_key = conv_id
    if cache_key not in _conversation_cache:
        conv_file = LONGMEMEVAL_DATASET_DIR / f"{conv_id}.json"
        with open(conv_file, "r", encoding="utf-8") as f:
            _conversation_cache[cache_key] = json.load(f)

    item = _conversation_cache[cache_key]
    fmt_fn = _get_format_history_for_prompt()
    return fmt_fn(item, max_sessions=None)

# LOCOMO
def _build_full_conv_context_locomo(conv_id: str, file_path: str | Path | None = None) -> str:
    """将 locomo 完整对话格式化为上下文文本。"""
    base_path = Path(file_path)
    if conv_id not in _conversation_cache:
        conv_file = base_path / f"{conv_id}.json"
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                _conversation_cache[conv_id] = json.load(f)
        else:
            raise FileNotFoundError(f"Conversation file not found: {conv_file}")

    conv = _conversation_cache[conv_id]
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    start_prompt = CONV_START_PROMPT.format(speaker_a, speaker_b)

    session_nums = sorted(
        int(k.split("_")[1]) for k in conv.keys()
        if k.startswith("session_") and "date_time" not in k
    )
    session_nums = [n for n in session_nums if conv.get(f"session_{n}", [])]
    blocks = []
    for n in session_nums:
        date_time = conv.get(f"session_{n}_date_time", "")
        dialogs = conv.get(f"session_{n}", [])
        if not dialogs:
            continue
        lines = []
        for d in dialogs:
            turn = d["speaker"] + ' said, "' + d["text"] + '"'
            if "blip_caption" in d:
                turn += " and shared %s." % d["blip_caption"]
            lines.append(turn)
        block = "DATE: " + date_time + "\nCONVERSATION:\n" + "\n".join(lines)
        blocks.append(block)

    return start_prompt + "\n\n".join(blocks)


def _build_full_conv_context_openclaw(conv_id: str) -> str:
    """OpenClaw 预留的对话上下文构建函数。"""
    # 这里暂时返回空字符串，未来可以根据 OpenClaw 的对话数据格式进行实现。
    return ""

#OpenClaw
class Conversation:
    def __init__(self, conv_id: str, benchmark: str, conversation_base: str | Path | None = None):
        self.conv_id = conv_id
        self.benchmark = benchmark
        self.haystack_session_ids = None
        if self.benchmark == "longmemeval":
            self.full_conv = _build_full_conv_context_longmemeval(conv_id)
            self.haystack_session_ids = (
                _get_haystack_session_ids(self.conv_id)
                if self.benchmark == "longmemeval"
                else None
            )
        elif self.benchmark == "locomo":
            self.full_conv = _build_full_conv_context_locomo(conv_id, conversation_base)
        elif self.benchmark == "openclaw":
            #openclaw预留的，你别动
            self.full_conv = _build_full_conv_context_openclaw(conv_id)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
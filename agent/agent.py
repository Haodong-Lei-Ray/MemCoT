from pathlib import Path
import importlib.util
import sys

# PROJECT_ROOT 与 version2 包一致，便于复用
try:
    from .. import PROJECT_ROOT
except ImportError:
    # Local standalone layout: DMSMem/agent/agent.py -> PROJECT_ROOT=DMSMem
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

from global_methods import run_chatgpt
# Local layout fallback: DMSMem/benchmark/locomo/task_eval
from benchmark.locomo.task_eval.gpt_utils import QA_PROMPT, QA_PROMPT_CAT_5, CONV_START_PROMPT
import json

import re
from datetime import datetime
from .format_tool import fix_escape_chars
from .prompt import rag_view_agent_prompt, \
    middle_view_agent_prompt, visual_ocr_agent_prompt, \
    observation_agent_prompt, answer_agent_prompt, conv_answer_agent_prompt
from .conversation import Conversation
# cache for conversation data
_conversation_cache: dict = {}
DATA_PATH = PROJECT_ROOT / "benchmark" / "locomo" / "data" / "locomo10.json"
CONV_DIR = PROJECT_ROOT / "benchmark" / "locomo" / "data" / "con"
IMG_PDF_BASE = PROJECT_ROOT / "data" / "img_pdf_con"
IMG_PDF_BASE = PROJECT_ROOT / "data" / "img_pdf_minor"

MULTIMODAL_KEYWORDS = ["4o", "4-turbo", "vision", "VL", "vl"]
DEFAULT_MULTIMODAL_MODEL = "gpt-4o-mini"

# LongMemEval 时间格式："2023/05/20 (Sat) 02:21"（参考 LOCOMO_CONVERSION_FEASIBILITY.md）
_LONGMEMEVAL_DATE_FMT = "%Y/%m/%d (%a) %H:%M"

def _parse_longmemeval_date(s: str):
    """解析 LongMemEval 时间字符串为可排序的 datetime，解析失败返回 datetime.min。"""
    if not s or not isinstance(s, str):
        return datetime.min
    try:
        return datetime.strptime(s.strip(), _LONGMEMEVAL_DATE_FMT)
    except ValueError:
        return datetime.min


def _parse_longmemeval_dia_id(dia_id: str, haystack_session_ids: list[str] | None) -> tuple[int, int, int]:
    """
    解析 LongMemEval dia_id 为 (sess_idx, turn_id, chunk_id)。
    dia_id 格式:
      - {haystack_session_id}_{turn_id}          （turn 内容短，无 chunk）
      - {haystack_session_id}_{turn_id}_c{chunk_id}  （turn 内容长，被分成多个 chunk）
      - {sess_idx}_{turn_id} 或 {sess_idx}_{turn_id}_c{chunk_id}  （当 haystack_session_ids 有重复时 buildrag 使用）
    返回 (在 haystack_session_ids 中的位置, turn_id, chunk_id)。chunk_id 不存在则为 0。
    """
    if not dia_id:
        raise ValueError(f"Failed to dia_id. {dia_id}")
    # 格式 {si}_{turn}[_c{chunk}]：当 sess_id 重复时 buildrag 生成
    if "_c" in dia_id:
        parts = dia_id.rsplit("_c", 1)
        if len(parts) == 2 and parts[1].isdigit():
            head = parts[0]
            chunk_id = int(parts[1])
            if head.count("_") == 1:
                a, b = head.split("_", 1)
                if a.isdigit() and b.isdigit():
                    return (int(a), int(b), chunk_id)
    else:
        if dia_id.count("_") == 1:
            a, b = dia_id.split("_", 1)
            if a.isdigit() and b.isdigit():
                return (int(a), int(b), 0)
    if not haystack_session_ids:
        raise ValueError(f"Failed to dia_id. {haystack_session_ids}")
    # 格式 {sess_id}_{turn}[_c{chunk}]
    for i, sid in enumerate(haystack_session_ids):
        if dia_id == sid:
            return (i, 0, 0)
        prefix = sid + "_"
        if dia_id.startswith(prefix):
            suffix = dia_id[len(prefix):]
            if "_c" in suffix:
                parts = suffix.rsplit("_c", 1)
                turn_id = int(parts[0]) if parts[0].isdigit() else 0
                chunk_id = int(parts[1]) if parts[1].isdigit() else 0
                return (i, turn_id, chunk_id)
            if suffix.isdigit():
                return (i, int(suffix), 0)
            return (i, 0, 0)
    raise ValueError(f"Failed to dia_id. {dia_id}")


def get_sort_key(
    item,
    benchmark: str = "locomo",
    haystack_session_ids: list[str] | None = None,
):
    """
    排序关键函数。
    locomo: 按 dia_id Dx:y 的 (x,y) 排序。
    longmemeval: 按 (date_time, session在haystack中的序, turn, chunk_id) 排序。
    """
    if benchmark == "locomo":
        dia_id = item["dia_id"]
        x_str, y_str = dia_id.lstrip("D").split(":")
        return (int(x_str), int(y_str))
    elif benchmark == "longmemeval":
        dt = _parse_longmemeval_date(item.get("date_time", ""))
        dia_id = item.get("dia_id", "")
        sess_idx, turn_id, chunk_id = _parse_longmemeval_dia_id(dia_id, haystack_session_ids)
        return (dt, sess_idx, turn_id, chunk_id)
    elif benchmark == "openclaw":
        from datetime import datetime
        dt_str = item.get("date_time", "")
        try:
            dt = datetime.strptime(dt_str, "%a %Y-%m-%d %H:%M GMT+8")
            return dt.timestamp()
        except Exception:
            return 0
    else:
        raise ValueError(f"get_sort_key Invalid benchmark: {benchmark}")

def _format_short_rag_for_prompt(results: list[dict]) -> str:    
    result = [f"[{i}] ({r.get('date_time','')}) {r['context']}"
        for i, r in enumerate(results)]
    dia_id_list = [r['dia_id'] for i, r in enumerate(results)]
    return result, dia_id_list

def _format_short_memory_for_prompt(
    results: list[dict],
    benchmark: str = "locomo",
    haystack_session_ids: list[str] | None = None,
) -> tuple:
    # 执行排序：locomo 按 Dx:y；longmemeval 按 date_time + session 序 + turn + chunk
    results = sorted(
        results,
        key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
    )
    result = [f"{r['context']}"
        for i, r in enumerate(results)]
    dia_id_list = [f"({r.get('date_time','')}) {r['dia_id']}" for i, r in enumerate(results)]
    return result, dia_id_list

def _format_context_for_qa_prompt(rag_results: list[dict]) -> str:
    """Format RAG results as task_eval RAG 格式：date_time: context，每行一条。"""
    lines = []
    for r in rag_results:
        dt = r.get("date_time", "") or ""
        ctx = r.get("context", "") or ""
        lines.append(f"{dt}: {ctx}".strip() if dt else ctx)
    return "\n".join(lines)

def _get_session_turns(conv_id: str, session_id: str) -> list[dict]:
    """加载单个 session 的对话轮次，返回 [{dia_id, date_time, speaker, text}]。"""
    if conv_id not in _conversation_cache:
        conv_file = CONV_DIR / f"{conv_id}.json"
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                _conversation_cache[conv_id] = json.load(f)
        if conv_id not in _conversation_cache:
            return []

    conv = _conversation_cache[conv_id]
    num = session_id.replace("D", "")
    # try:
    n = int(num)
    # except ValueError:
    #     return []
    session_key = f"session_{n}"
    date_time = conv.get(f"{session_key}_date_time", "")
    dialogs = conv.get(session_key, [])
    return [
        {
            "dia_id": d.get("dia_id", ""),
            # "date_time": date_time,
            "speaker": d.get("speaker", "Unknown"),
            "text": d.get("text", d.get("clean_text", "")),
        }
        for d in dialogs
    ], date_time

def _parse_json_from_llm(text: str) -> dict | None:
    # 先修复转义字符
    text = fix_escape_chars(text)
    # 匹配 ```json
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # 有些 agent 不写 json 两个字，直接 ```
        json_str = text.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(json_str)
        raise ValueError(f"Failed to parse JSON from LLM output.")


# Agent
class ZoomInFocalRetrieve:
    """RAG View：`model` / `temperature` 在构造时绑定，`run()` 不再接收这两项。"""

    def __init__(self, model: str, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def run(
        self,
        root_query: str,
        query_queue: list[str],
        rag_results: list[dict],
        last_thought_information: str,
        known_info_rag: list[dict],
        conversation,
    ) -> dict:
        """
        RAG View agent: 观察 rag_results，选出 useful_dia_ids，并写 report。
        report 说明：(1) 为什么选用这些 useful_dia_ids；(2) 还缺什么信息；(3) 建议的新 query。
        返回 {useful_evidence: [...], report: "..."}
        """
        benchmark = conversation.benchmark
        haystack_session_ids = conversation.haystack_session_ids
        # 执行排序：locomo 按 Dx:y；longmemeval 按 date_time + session 序 + turn + chunk
        rag_results_sort = sorted(
            rag_results,
            key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
        )
        rag_results_list, dia_id_list = _format_short_rag_for_prompt(rag_results_sort)
        rag_results_text = "\n".join(rag_results_list)
        search_query = query_queue if query_queue else root_query
        query_information = f'root_query: {root_query}' if root_query == search_query else f'root_query: {root_query}\nsearch_query: {search_query}' 
        
        known_info_rag_sort = sorted(
            known_info_rag,
            key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
        )
        known_info_rag_text, _ = _format_short_memory_for_prompt(
            known_info_rag_sort, benchmark, haystack_session_ids=haystack_session_ids
        )
        known_information = f'Known information: {known_info_rag_text}\n{last_thought_information}' if known_info_rag_text or last_thought_information else ''

        prompt = rag_view_agent_prompt(query_information, known_information,rag_results_text,benchmark)

        try_step = 3
        while True:
            try_step -= 1
            try:
                resp = run_chatgpt(
                    prompt,
                    model=self.model,
                    num_tokens_request=1024,
                    temperature=self.temperature,
                )
                out = _parse_json_from_llm(resp)
                break
            except ValueError:
                print("Failed to parse JSON, retrying with cleaned text...")
                continue
        num_ids = out.get("useful_ids", [])
        num_ids = [int(i) if isinstance(i, str) else i for i in num_ids]
        # num_ids = [i for i in num_ids if 0 <= i < len(rag_results_sort)]
        ids = [rag_results_sort[i] for i in num_ids] if len(num_ids) != 0 else []
        thinking = out.get("thinking")
        missing_information = out.get("missing_information")
        
        return {
            "useful_evidence": ids,
            "thinking": thinking,
            "missing_information": missing_information,
        }

class ZoomOutContextExpansion:
    def __init__(self, model: str, temperature: float = 0.0, middle_scale: int = 4):
        self.model = model
        self.temperature = temperature
        self.W=middle_scale

    def get_context_window4locomo(
        self, temp_useful_rag: list[dict], conv_id: str
    ) -> tuple[list[str], list[dict]]:
        """LoCoMo: 对每个 dia_id (Dx:y) 取所在 session 上下 self.W 条对话，返回 (context_blocks, rag_results)。"""
        sessions: dict[str, dict] = {}

        for item in temp_useful_rag:
            dia_id = item.get("dia_id", "")
            sess_prefix = dia_id.split(":")[0]
            turn_num = int(dia_id.split(":")[1])

            turns, date_time = _get_session_turns(conv_id, sess_prefix)

            start = max(1, turn_num - self.W)
            end = min(len(turns), turn_num + self.W)
            window = turns[start - 1:end]

            if sess_prefix not in sessions:
                sessions[sess_prefix] = {"date_time": date_time, "turns": {}}
            turn_map = sessions[sess_prefix]["turns"]
            for t in window:
                tid = t.get("dia_id", "")
                if tid and tid not in turn_map:
                    turn_map[tid] = t

        context_blocks = []
        all_window_turns = []
        dia_id_list = []
        idx_i = 0
        rag_results = []
        for sess_prefix, info in sorted(sessions.items()):
            sorted_turns = sorted(info["turns"].values(),
                                key=lambda t: int(t["dia_id"].split(":")[1]) if ":" in t.get("dia_id", "") else 0)
            all_window_turns.extend(sorted_turns)
            date_time = info['date_time']
            lines = [f"Session {sess_prefix} (date_time: {date_time}):"]
            for t in sorted_turns:
                lines.append(f'{idx_i}: {t.get("speaker", "Unknown")}: "{t.get("text", "")}')
                dia_id_list.append(t["dia_id"])
                rag_item = {
                    "dia_id": t["dia_id"],
                    "date_time": date_time,
                    "context": f'{t.get("speaker", "Unknown")} said, {t.get("text", "")}'
                }
                rag_results.append(rag_item)
                idx_i += 1
            context_blocks.append("\n".join(lines))

        return context_blocks, rag_results


    def get_context_window4longmemeval(
        self,
        temp_useful_rag: list[dict],
        conv_id: str,
        haystack_session_ids: list[str],
        full_conv: dict,
    ) -> tuple[list[str], list[dict]]:
        """LongMemEval: 对每个 dia_id ({sess_id}_{turn}[_c{chunk}]) 取所在 session 上下 self.W 条对话。
        以 turn 为单位，在同一 session 内取命中 turn 前后各 self.W 个 turn 作为上下文窗口。
        返回 (context_blocks, rag_results)。
        """
        entry = full_conv
        all_sessions = entry.get("haystack_sessions", [])
        all_dates = entry.get("haystack_dates", [])

        # sess_idx -> {"date_time", "sess_id", "turns": {turn_0based: turn_dict}}
        sessions: dict[int, dict] = {}

        for item in temp_useful_rag:
            dia_id = item.get("dia_id", "")
            sess_idx, turn_id, _chunk_id = _parse_longmemeval_dia_id(dia_id, haystack_session_ids)
            if sess_idx < 0 or sess_idx >= len(all_sessions):
                continue
            session = all_sessions[sess_idx]
            if not isinstance(session, list) or len(session) == 0:
                continue
            date_time = all_dates[sess_idx] if sess_idx < len(all_dates) else ""

            # dia_id 中 turn_id 约定为 1-based；窗口计算需转为 0-based
            if turn_id <= 0:
                continue
            center_idx = min(max(turn_id - 1, 0), len(session) - 1)
            start_t = max(0, center_idx - self.W)
            end_t = min(len(session), center_idx + self.W + 1)

            if sess_idx not in sessions:
                sess_id = (
                    haystack_session_ids[sess_idx]
                    if sess_idx < len(haystack_session_ids)
                    else str(sess_idx)
                )
                sessions[sess_idx] = {
                    "date_time": date_time,
                    "sess_id": sess_id,
                    "turns": {},
                }
            turn_map = sessions[sess_idx]["turns"]
            for ti in range(start_t, end_t):
                if ti not in turn_map:
                    turn_map[ti] = session[ti]

        context_blocks = []
        idx_i = 0
        rag_results = []
        for sess_idx in sorted(sessions.keys()):
            info = sessions[sess_idx]
            date_time = info["date_time"]
            sess_id = info["sess_id"]
            lines = [f"Session {sess_id} (date_time: {date_time}):"]
            for ti in sorted(info["turns"].keys()):
                t = info["turns"][ti]
                role = t.get("role", "user")
                content = str(t.get("content", "")).strip()
                turn_dia_id = f"{sess_id}_{ti + 1}"  # 1-based，与 buildrag 一致
                lines.append(f'{idx_i}: {role}: "{content}"')
                rag_results.append({
                    "dia_id": turn_dia_id,
                    "date_time": date_time,
                    "context": f'{role} said, "{content}"',
                })
                idx_i += 1
            context_blocks.append("\n".join(lines))

        return context_blocks, rag_results

    def get_context_window4openclaw(
        self, temp_useful_rag: list[dict], full_conv: list[dict], dia_id_to_idx: dict
    ) -> tuple[list[str], list[dict]]:
        """
        OpenClaw: 整个 session 在 full_conv 中。
        针对每个检索到的证据（temp_useful_rag），在 full_conv 中取前后 self.W 条对话。
        合并所有窗口，去重，并按照指定格式输出。
        """
        # ==========================================
        # 1. 获取所有相关的上下文 evidence (E_all) 并去重
        # ==========================================
        e_all_indices = set()
        for mem in temp_useful_rag:
            dia_id = mem.get("dia_id")
            if dia_id in dia_id_to_idx:
                idx = dia_id_to_idx[dia_id]
                # 扩展窗口：前后各取 self.W 条
                start_idx = max(0, idx - self.W)
                end_idx = min(len(full_conv), idx + self.W + 1)
                
                for i in range(start_idx, end_idx):
                    e_all_indices.add(i)
                    
        # ==========================================
        # 2. 将去重后的 evidence 转换为所需的输出格式
        # ==========================================
        context_blocks = []
        rag_results = []
        
        # 按原始对话顺序排序
        sorted_indices = sorted(list(e_all_indices))
        
        lines = []
        for idx_i, orig_idx in enumerate(sorted_indices):
            msg = full_conv[orig_idx]
            role = msg.get("role", "Unknown")
            content = msg.get("context", "")
            date_time = msg.get("date_time", "")
            dia_id = msg.get("dia_id", "")
            
            # 格式化为: idx_i: role: "content"
            lines.append(f'{idx_i}: {role}: "{content}"')
            
            # 构建 rag_results 项
            rag_item = {
                "dia_id": dia_id,
                "date_time": date_time,
                "context": f'{role} said, "{content}"'
            }
            rag_results.append(rag_item)
            
        if lines:
            # 将所有行合并为一个 context_block
            context_blocks.append("\n".join(lines))
            
        return context_blocks, rag_results

    def run(
        self,
        root_query: str,
        query_queue: list[str],
        temp_short_memory: list[dict],
        known_info_rag: list[dict],
        conversation,
    ) -> dict:
        """
        Middle View：`model` / `temperature` / 窗口 `self.W` 由 `__init__` 绑定；本方法只接逐步上下文。
        返回 {"useful_evidence": [...], "thinking": "...", "missing_information": "..."}
        """
        conv_id = conversation.conv_id
        benchmark = conversation.benchmark
        haystack_session_ids = conversation.haystack_session_ids
        full_conv = conversation.full_conv

        if benchmark == "locomo":
            context_blocks, rag_results = self.get_context_window4locomo(
                temp_short_memory, conv_id
            )
        elif benchmark == "longmemeval":
            context_blocks, rag_results = self.get_context_window4longmemeval(
                temp_short_memory,
                conv_id,
                haystack_session_ids or [],
                full_conv=full_conv,
            )
        elif benchmark == "openclaw":
            context_blocks, rag_results = self.get_context_window4openclaw(
                temp_short_memory, full_conv, conversation.dia_id_to_idx
            )
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}")
        middle_context_text = "\n\n".join(context_blocks)
        search_query = query_queue if query_queue else root_query
        query_information = f'root_query: {root_query}' if root_query == search_query else f'root_query: {root_query}\nsearch_query: {search_query}'

        known_info_rag_sort = sorted(
            known_info_rag,
            key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
        )
        known_info_text, _ = _format_short_memory_for_prompt(
            known_info_rag_sort, benchmark, haystack_session_ids=haystack_session_ids
        )
        known_information = f'Known information: {known_info_text}' if known_info_text else ''

        prompt = middle_view_agent_prompt(query_information, known_information, middle_context_text, benchmark)
        
        resp = run_chatgpt(
            prompt,
            model=self.model,
            num_tokens_request=1024,
            temperature=self.temperature,
        )
        out = _parse_json_from_llm(resp)
        num_ids = out.get("useful_ids", [])
        num_ids = [int(i) if isinstance(i, str) else i for i in num_ids]
        # num_ids = [i for i in num_ids if 0 <= i < len(rag_results)]
        ids = [rag_results[i] for i in num_ids] if len(num_ids) != 0 else []
        thinking = out.get("thinking")
        
        missing_information = out.get("missing_information")
        
        return {
            "useful_evidence": ids,
            "thinking": thinking,
            "missing_information": missing_information,
        }

# LongMemEval benchmark: dataset per question, conv_id = "{idx:04d}_{question_id}" e.g. "0000_e47becba"
LONGMEMEVAL_DATASET_DIR = PROJECT_ROOT / "benchmark" / "LongMemEval" / "dataset"
LONGMEMEVAL_FORMAT_HISTORY = (
    PROJECT_ROOT / "module_unit" / "LongMemEval" / "src" / "generation" / "format_history.py"
)
_format_history_for_prompt_fn = None

RELATIONSHIP_DIR = PROJECT_ROOT / "module_version" / "version2" / "tool" / "related3"

_relationship_cache: dict = {}

def _get_total_sessions(conv_id: str) -> int:
    """获取 conv 的总 session 数。"""
    if conv_id not in _conversation_cache:
        conv_file = CONV_DIR / f"{conv_id}.json"
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                _conversation_cache[conv_id] = json.load(f)
    conv = _conversation_cache.get(conv_id, {})
    n = 0
    while f"session_{n + 1}" in conv:
        n += 1
    return n

def _extract_session_nums(records: list) -> set:
    """从 records 的 dia_id 提取 session 编号集合。D1:5 -> 1"""
    nums = set()
    for r in records:
        dia_id = r.get("dia_id", "")
        if ":" in dia_id:
            prefix = dia_id.split(":")[0]
            try:
                nums.add(int(prefix.lstrip("D")))
            except ValueError:
                pass
    return nums

def _select_extra_sessions(d_prime: set, seen_session: set, total_sessions: int, budget: int) -> list:
    """
    选取 D+ sessions：距 D' 最近且不在 D'/seen_session 中，按 (距离, 编号) 排序。
    示例: D'={1,9,14}, seen={3,8}, D=5 -> budget=2 -> D+=[2,10]
    """
    if budget <= 0:
        return []
    excluded = d_prime | seen_session
    candidates = [s for s in range(1, total_sessions + 1) if s not in excluded]
    if not d_prime:
        return sorted(candidates)[:budget]
    candidates.sort(key=lambda s: (min(abs(s - d) for d in d_prime), s))
    return candidates[:budget]

def _load_session_images(conv_id: str, session_nums: list) -> list:
    """加载指定 sessions 的所有 PDF 页面图片路径。"""
    image_paths = []
    for sn in sorted(session_nums):
        session_dir = IMG_PDF_BASE / conv_id / f"D{sn}"
        if not session_dir.exists():
            raise ValueError("not session_dir.exists()")
        for img_file in sorted(session_dir.iterdir()):
            if img_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                image_paths.append(str(img_file))
    return image_paths

def _is_multimodal_model(model: str) -> bool:
    return any(kw in model for kw in MULTIMODAL_KEYWORDS)

def panoramic_visual_grounding(
    root_query: str,
    query_queue: list,
    temp_short_memory: list,
    known_info_rag: list,
    model: str,
    conv_id: str,
    seen_session: list,
    max_view_sessions: int = 6,
    temperature: float = 0.0,
    benchmark: str = "locomo",
    haystack_session_ids: list[str] | None = None,
) -> dict:
    """
    Visual OCR Agent: 阅读 PDF 图片发现对话中的有用信息。

    算法：
    1. 从 temp_useful_rag + known_info_rag 提取涉及的 session 集合 D'
    2. 若 |D'| >= max_view_sessions -> 直接查看 D'（不补充额外 session）
    3. 否则按距 D' 最近、不在 seen_session/D' 中、编号从小到大选取 D+
    4. 将 D' | D+ 的 PDF 图片送多模态模型阅读
    返回: {useful_evidence, thinking, missing_information, viewed_sessions}
    """
    from global_methods import run_chatgpt_multimodal

    mm_model = model if _is_multimodal_model(model) else DEFAULT_MULTIMODAL_MODEL

    total_sessions = _get_total_sessions(conv_id)

    # D': 从已知信息中提取涉及的 session 编号
    all_known = (temp_short_memory or []) + (known_info_rag or [])
    d_prime = _extract_session_nums(all_known)

    # 计算 D+
    budget = max_view_sessions - len(d_prime)
    seen_set = set(seen_session)
    d_plus = _select_extra_sessions(d_prime, seen_set, total_sessions, budget)
    sessions_to_view = sorted(d_prime | set(d_plus))

    print(f"[visual_ocr] D'={sorted(d_prime)}, D+={d_plus}, viewing={sessions_to_view}, seen={seen_session}")

    print(f"full see is {sessions_to_view}")
    image_paths = _load_session_images(conv_id, sessions_to_view)
    
    add_view_sessions = d_plus

    # 加载 session 对话数据（用于 dia_id -> 完整记录 映射）
    all_turns = {}
    for sn in sessions_to_view:
        result = _get_session_turns(conv_id, f"D{sn}")
        if result:
            turns, date_time = result
            for t in turns:
                tid = t["dia_id"]
                all_turns[tid] = {
                    "dia_id": tid,
                    "date_time": date_time,
                    "context": f'{t.get("speaker", "Unknown")} said, {t.get("text", "")}',
                }

    # 构建 prompt
    search_query = query_queue if query_queue else [root_query]
    query_information = (
        f"root_query: {root_query}"
        if len(search_query) == 1 and search_query[0] == root_query
        else f"root_query: {root_query}\nsearch_query: {search_query}"
    )

    known_info_text = ""
    if known_info_rag:
        known_info_rag_sort = sorted(
            known_info_rag,
            key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
        )
        known_info_text, _ = _format_short_memory_for_prompt(
            known_info_rag_sort, benchmark, haystack_session_ids=haystack_session_ids
        )
    known_information = f"Known information:\n{known_info_text}" if known_info_text else ""

    rag_info_text = ""
    if temp_short_memory:
        rag_sort = sorted(
            temp_short_memory,
            key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
        )
        rag_info_list, rag_dia_id_list = _format_short_rag_for_prompt(rag_sort)
        rag_info_list = [rag_dia_id_list[i]+rag_info_list[i] for i in range(len(rag_dia_id_list))]
        rag_info_text = "\n".join(rag_info_list)
    rag_information = f"Current RAG findings:\n{rag_info_text}" if rag_info_text else ""

    session_list_str = ", ".join(f"D{s}" for s in sessions_to_view)

    prompt = visual_ocr_agent_prompt(query_information, known_information,rag_information,session_list_str,benchmark)
    batch_size = 7
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _call_batch(batch_idx, batch_paths):
        print(f"[visual_ocr] batch {batch_idx + 1}/{len(batches)}, images: {len(batch_paths)}")
        resp = run_chatgpt_multimodal(
            text_prompt=prompt,
            image_paths=batch_paths,
            model=mm_model,
            num_tokens_request=512,
            temperature=temperature,
        )
        out = _parse_json_from_llm(resp)
        return batch_idx, out

    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(_call_batch, idx, bp): idx
            for idx, bp in enumerate(batches)
        }
        for future in as_completed(futures):
            idx, out = future.result()
            results[idx] = out

    thinking_list = []
    final_useful_evidence = []
    seen_dia_ids = set()
    for out in results:
        thinking_list.append(out.get("thinking", ""))
        for did in out.get("useful_dia_ids", []):
            if did in all_turns and did not in seen_dia_ids:
                final_useful_evidence.append(all_turns[did])
                seen_dia_ids.add(did)

    return {
        "useful_evidence": final_useful_evidence,
        "thinking": thinking_list,
        "viewed_sessions": sessions_to_view,
        "add_view_sessions": add_view_sessions,
    }

# function agent
WRONG_LIST = ['No information available', 'unknown']

class JudgeAgent:
    """Observation：比对 query 与 RAG/短期记忆；`model` / 初始 `temperature` 在构造时绑定。"""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def run(
        self,
        query: str,
        conversation: Conversation,
        temp_useful_rag: list[dict],
        short_memory: list[dict],
        fail_queue_trajectory: list,
        conv_memory: list[str] | None = None,
    ) -> dict:
        """
        返回: useful_evidence, can_answer, action, new_queries, thinking
        """
        benchmark = conversation.benchmark
        haystack_session_ids = conversation.haystack_session_ids

        queries_num = 1
        rag_results_sort = sorted(
            temp_useful_rag + short_memory,
            key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
        )
        temp_useful, dia_id_list = _format_short_rag_for_prompt(rag_results_sort)
        fail_queue_information = "Fail query:\n" + "\n".join(
            f"No.{i} last_action: {x['last_action']} Query_queue: {str(x['query_queue'])} Rag think: {str(x['rag_view_report'])} \n"
            for i, x in enumerate(fail_queue_trajectory)
        ) if fail_queue_trajectory else ""
        fail_queue = []
        for data in fail_queue_trajectory:
            fail_queue.extend(data["query_queue"])

        short_memory_text = "Short Memory:" + "\n" + "\n".join(temp_useful)
        if not temp_useful_rag:
            fail_queue_information += (
                "\nNo useful information provided. Please strongly update your query. "
                "Do not let new_queries as same as and Fail query or Query!!"
            )
        if conv_memory:
            conv_memory_text = "Full Conversation Memory:" + "\n" + "\n".join(conv_memory)
            thinking = (
                "thinking: Check the Short Memory. Think about each item connection and relevant. "
                "If you cannot answer, please refer to the Full Conversation Memory, then use this "
                "information to construct new_queries to locate the correct information."
            )
        else:
            conv_memory_text = ""
            thinking = (
                "thinking: Check the Short Memory. Think about each item connection and relevant."
            )
        prompt = observation_agent_prompt(
            fail_queue_information,
            query,
            short_memory_text,
            conv_memory_text,
            thinking,
            queries_num,
            benchmark,
        )

        repeat_flag = True
        step = 5
        t = self.temperature
        num_tokens_request=1024
        while repeat_flag:
            step -= 1
            resp = run_chatgpt(
                prompt,
                model=self.model,
                num_tokens_request=num_tokens_request,
                temperature=t,
            )
            out = _parse_json_from_llm(resp)
            new_queries = out.get("new_queries")
            can_answer = bool(out.get("can_answer", False))
            if can_answer and new_queries is None:
                new_queries = "None"
            if new_queries is not None and isinstance(new_queries, list):
                new_queries = new_queries[:2]
            if isinstance(new_queries, str):
                if new_queries not in fail_queue:
                    repeat_flag = False
            elif not set(new_queries).issubset(set(fail_queue)):
                repeat_flag = False
            elif can_answer:
                repeat_flag = False
            else:
                print(f"Repeat!!!!! {new_queries}")
                prompt += (
                    "\nnIMPORTANT: You can just break query as the only one **keywords or some phases**. Like red, Shanghai..."
                )
            if step in [2, 1]:
                prompt += (
                    "\nIMPORTANT: Change this new_queries. You can just break query as the only one "
                    '**words or a phase**. Like old query=(What/When...)+keyword1+keyword2. '
                    'new_queries=["keyword1"]. The query could be declarative sentence.'
                )
            t += 0.5
            num_tokens_request = int(num_tokens_request/2)
            t = max(t, 1.0)
            if step <= 0:
                raise ValueError("Bug: new_queries == []")
        num_ids = out.get("useful_id", [])
        num_ids = [int(i) if isinstance(i, str) else i for i in num_ids]

        thinking = out.get("thinking")
        if len(num_ids) > 0 and max(num_ids) >= len(dia_id_list):
            useful_evidence = dia_id_list
        else:
            useful_evidence = [dia_id_list[i] for i in num_ids] if len(num_ids) != 0 else []
        return {
            "thinking": thinking,
            "useful_evidence": useful_evidence,
            "can_answer": can_answer,
            "action": out.get("action"),
            "new_queries": new_queries,
        }

def answer_agent(prompt: str, model: str, benchmark: str = "locomo") -> dict:
    """
    Responder: 使用已经拼接好的 prompt 调用 LLM。
    返回: {thinking: str, answer: str}
    """
    resp = run_chatgpt(prompt, model=model, num_tokens_request=1024, temperature=0.0)
    if benchmark == 'locomo' or benchmark == 'openclaw':
        out = _parse_json_from_llm(resp)
        answer = out.get("answer",'')
        thinking = out.get("thinking",'')
        if benchmark == 'longmemeval':
            answer = thinking+" Answer is: "+ answer
        assert thinking != '',f"resp = {resp}"
        return {
            "thinking": thinking,
            "answer": answer,
        }
    else:
        return {
            "thinking": resp,
            "answer": resp,
        }

def conv_answer_agent(
    root_query: str,
    full_conv: str,
    model: str,
    temperature: float = 0.0,
    benchmark: str = "locomo",
) -> dict:
    """
    Full-context agent: 将完整对话作为上下文，直接回答问题。
    与 gpt_utils.py batch_size=1 的处理方式一致：context + QA_PROMPT。
    category=5 时使用 QA_PROMPT_CAT_5。
    返回: {"answer": str}
    """
    category = 0
    if category == 2:
        query = query + " Use DATE of CONVERSATION to answer with an approximate date."
    query = root_query
    prompt = conv_answer_agent_prompt(full_conv, query, benchmark)

    resp = run_chatgpt(prompt, model=model, num_tokens_request=64, temperature=temperature)
    return {
        "thinking": '',
        "answer": resp,
    }

def return_answer_agent_prompt(
    query: str,
    short_memory: list[dict],
    obs_report: str,
    conversation,
    additional_information: str = "",
) -> str:
    """仅构造 answer_agent 所用 prompt（不调用 LLM）。返回 prompt 字符串。"""
    benchmark = conversation.benchmark
    haystack_session_ids = conversation.haystack_session_ids

    rag_results_sort = sorted(
        short_memory,
        key=lambda x: get_sort_key(x, benchmark, haystack_session_ids=haystack_session_ids),
    )
    short_memory, _ = _format_short_rag_for_prompt(rag_results_sort)
    short_memory_text = "Short Memory:\n" + "\n".join(short_memory) if short_memory else ""
    # additional_information_text = "3.IF you can not answer by Short Memory, just follow this thinking and answer: "+additional_information if additional_information else ''
    additional_information_text = ""
    prompt = answer_agent_prompt(additional_information_text, short_memory_text, query, benchmark)
    return prompt


def return_conv_answer_agent_prompt(
    root_query: str,
    conversation,
) -> str:
    """仅构造 conv_answer_agent 所用 prompt（不调用 LLM）。"""
    full_conv = conversation.full_conv
    benchmark = conversation.benchmark
    return conv_answer_agent_prompt(full_conv, root_query, benchmark)
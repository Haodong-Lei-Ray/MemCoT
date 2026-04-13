#!/usr/bin/env python3
"""
ReAct + RAG — LoCoMo 应用场景。支持 NaiveRAG 与 LightRAG 两种检索，通过超参数 --rag-type 选择。

输入 query -> 入 query queue
Loop (max step=10):
  Act: 对 query queue 中所有 query 执行 RAG 检索（naive 或 lightrag），得到 RAG 结果
  Observation: 用 query 比对 RAG 结果
    1. 可直接回答则停止
    2. 否则保留可能有效的 RAG 结果
  Thought:
    1. 当前 RAG 缺失什么信息？
    2. 为缺失信息生成新的 RAG query（可多个），入 query queue
  下一轮 Act...

轨迹保存为 Act-Observation-Thought 格式，便于调试。

Usage:
  python react_naiverag.py -c conv-26 "When did Caroline go to the LGBTQ support group?"
  python react_naiverag.py -c conv-26 --rag-type naive -o output_dir "query"   # NaiveRAG (pkl@naiverag)
  python react_naiverag.py -c conv-26 --rag-type lightrag -o output_dir "query"  # LightRAG (default)
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from tool.longmemevaltool import rewrite_first_person_to_user as _rewrite_first_person_to_user
from tool.rag import (
    DEFAULT_IMG_INDEX_BASE,
    RAG_TYPE_CHOICES,
    RAG_TYPE_LIGHTRAG,
    RAG_TYPE_NAIVE,
    create_img_retriever,
    create_lightrag,
    finalize_lightrag,
    get_rag_event_loop,
    rag_retrieve_multi,
)

LIGHTRAG_PATH = PROJECT_ROOT / "mem" / "LightRAG"
sys.path.insert(0, str(LIGHTRAG_PATH))

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Agent 模块：rag_view, full_view, answer, thought, observation
from agent.agent import (
    ZoomInFocalRetrieve,
    zoom_out_context_expansion_agent,
    answer_agent,
    guess_answer_agent,
    judge_agent,
    panoramic_visual_grounding,
    return_answer_agent_prompt,
    return_conv_answer_agent_prompt,
    _build_full_conv_context,
    _build_full_conv_context_longmemeval,
    _get_haystack_session_ids,
    _parse_json_from_llm,
    WRONG_LIST,
)
from global_methods import run_chatgpt

DEFAULT_LIGHTRAG_WORKING_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage"
NAIVERAG_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/naiverag"
DATA_PATH = PROJECT_ROOT / "benchmark" / "locomo" / "data" / "locomo10.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output"
MAX_LOOP_STEP = 10
FULL_VIEW_SESS_NUM = 1  # Full View agent 每次选择的 session 数量
DEFAULT_AGENT_FLAG = "11000"


def _conv_id_to_workspace(conv_id: str) -> str:
    return conv_id.replace("-", "")

def add_category_information(category: int) -> str:
    """根据 QA category 返回额外提示信息，用于引导各 Agent 的推理方向。"""
    if category == 2:
        return "Use DATE of CONVERSATION to answer with an approximate date."
    if category == 5:
        return 'If the answer is not mentioned in the conversation, answer "Not mentioned in the conversation".'
    return ""

@dataclass
class MemCoTExit:
    """MemCoT 主循环结束时的状态；`prompt` 已在前序拼好，供 finalize 里 run_chatgpt 使用。"""

    kind: Literal["evidence_ok", "fallback"]
    output_dir: str
    trajectory: list
    prompt: Optional[str] = None
    model: Optional[str] = None
    benchmark: Optional[str] = None
    # evidence_ok（循环内 try_responder 的通过结果，仅作记录）
    answer: Optional[str] = None
    answer_thinking: Optional[str] = None
    steps: Optional[int] = None
    final_evidence: Optional[list[str]] = None
    final_query_queue: Optional[list[str]] = None
    # fallback
    full_conv: Optional[str] = None
    max_step: Optional[int] = None

def try_responder_answer_from_evidence(
    query: str,
    short_memory: list[dict],
    model: str,
    obs_report: str,
    additional_information: str,
    benchmark: str,
    haystack_session_ids: Optional[list[str]] = None,
) -> Optional[tuple[str, str]]:
    """
    Responder：在固定证据上调用 answer_agent；答案为空或在 WRONG_LIST 中则返回 None，
    由 MemCoT 主循环继续检索。
    """
    ans = answer_agent(
        query,
        short_memory,
        model=model,
        obs_report=obs_report,
        additional_information=additional_information,
        benchmark=benchmark,
        haystack_session_ids=haystack_session_ids,
    )
    answer = ans.get("answer", "")
    thinking = ans.get("thinking", "")
    if answer == "" or answer in WRONG_LIST:
        return None
    return answer, thinking

def _save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def finalize_memcot_exit(exit_state: MemCoTExit) -> dict:
    """
    根据 MemCoT 退出状态完成 Responder 收尾（使用 exit_state.prompt 与 model/benchmark）。
    evidence_ok：run_chatgpt + 与 answer_agent 一致的解析。
    fallback：run_chatgpt，等价于 conv_answer_agent。
    """
    prompt = exit_state.prompt
    if prompt is None:
        raise ValueError("MemCoTExit.prompt 必须在 finalize 前设置")
    model = exit_state.model or DEFAULT_MODEL
    benchmark = exit_state.benchmark or "locomo"
    if exit_state.kind == "evidence_ok":
        resp = run_chatgpt(
            prompt, model=model, num_tokens_request=1024, temperature=0.0
        )
        if benchmark == "locomo":
            out = _parse_json_from_llm(resp) or {}
            out_answer = out.get("answer", "")
            out_thinking = out.get("thinking", "")
            if benchmark == "longmemeval":
                out_answer = out_thinking + " Answer is: " + out_answer
            assert out_thinking != "", f"resp = {resp}"
        else:
            out_thinking = resp
            out_answer = resp
        result = {
            "answer": out_answer,
            "answer_thinking": out_thinking,
            "stopped_reason": "can_answer",
            "steps": exit_state.steps,
            "final_evidence": exit_state.final_evidence,
            "final_query_queue": exit_state.final_query_queue,
            "trajectory": exit_state.trajectory,
        }
        _save_json(result, os.path.join(exit_state.output_dir, "result.json"))
        return result

    assert exit_state.full_conv is not None
    temperature = 0.0
    resp = run_chatgpt(
        prompt, model=model, num_tokens_request=64, temperature=temperature
    )
    conv_view_result = {"thinking": "", "answer": resp}
    answer = conv_view_result["answer"]
    answer_thinking = conv_view_result["thinking"]
    result = {
        "answer": answer,
        "answer_thinking": answer_thinking,
        "stopped_reason": "max_step",
        "steps": exit_state.max_step,
        "final_evidence": exit_state.final_evidence,
        "final_query_queue": exit_state.final_query_queue,
        "trajectory": exit_state.trajectory,
    }
    _save_json(result, os.path.join(exit_state.output_dir, "result.json"))
    print(f"Answer: {answer}")
    return result

class MemCoT:
    """
    绑定「同一对话 + RAG 基础设施」；每条 QA 调用 run() 传入 query / model / output_dir / agent_flag_str 等。
    RAG View（ZoomInFocalRetrieve）的 LLM 在构造时用 `model`（及固定 temperature=1.0）绑定，run() 里不再传。
    其它 agent 仍使用 run(..., model=...)。
    """

    def __init__(
        self,
        *,
        conv_id: str,
        model: str = DEFAULT_MODEL,
        category: int = 0,
        working_dir: Optional[str] = None,
        rag_top_k: int = 10,
        rag_type: str = RAG_TYPE_LIGHTRAG,
        rag=None,
        full_conv: Optional[str] = None,
        img_retriever=None,
        middle_scale: int = 3,
        max_step: int = MAX_LOOP_STEP,
        benchmark: str = "locomo",
    ):
        # 初始化RAG基础设施
        self.working_dir = working_dir
        self.rag = rag
        self.rag_type = rag_type
        #通用设置
        self.full_conv = full_conv
        self.max_step = max_step
        self.rag_top_k = rag_top_k
        self.zoom_in_focal_retrieve = ZoomInFocalRetrieve(model, temperature=1.0)
        # zoom_out_context_expansion
        self.middle_scale = middle_scale
        # panoramic_visual_grounding
        self.img_retriever = img_retriever

        # 为Memory Benchmark 搞的eval
        self.benchmark = benchmark
        self.conv_id = conv_id
        self.category = category

    def run(
        self,
        query: str,
        *,
        model: str = DEFAULT_MODEL,
        output_dir: Optional[str] = None,
        agent_flag_str: str = DEFAULT_AGENT_FLAG,
        benchmark: Optional[str] = None,
    ) -> MemCoTExit:
        """
        单次 QA：在已绑定的 conv / RAG 上检索并回答。
        返回 MemCoTExit；请用 finalize_memcot_exit(exit_state) 落盘。
        """
        bm = self.benchmark if benchmark is None else benchmark
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        haystack_session_ids = None
        if bm == "longmemeval":
            query = _rewrite_first_person_to_user(query)
            haystack_session_ids = _get_haystack_session_ids(self.conv_id)
        root_query = query
        query_queue = [query]
        short_memory: list[dict] = [] # 存对query的短期记忆
        trajectory_memory: list[dict] = [] # 存对query的短期记忆
        trajectory: list[dict] = [] 
        fail_queue_trajectory: list[dict] = []
        agent_flag = [c == "1" for c in agent_flag_str]
        _ = get_rag_event_loop()
        additional_information = add_category_information(self.category)
        conv_memory = []
        # for agent 3
        interrupt_step = 3
        visual_seen_session = []

        # ====MemCoT核心循环
        for step in range(1, self.max_step + 1):
            rag_view_missing_information = []
            temp_useful_rag: list[dict] = []
            short_memory_dia_ids = [m.get("dia_id", "") for m in short_memory]
            print(f"Query_queue: {query_queue}")
            assert len(query_queue) != 0
            # ─── zoom_in_focal_retrieve1 ───
            rag_results, rag_result_list = rag_retrieve_multi(
                query_queue,
                self.conv_id,
                top_k=self.rag_top_k,
                rag_type=self.rag_type,
                working_dir=self.working_dir,
                rag=self.rag,
            )
            assert len(rag_results) != 0
            if trajectory:
                last_observation = trajectory[-1].get("Observation") or {}
                last_thinking = trajectory[-1].get("Thought") or {}
                last_action = last_observation.get("action", "")
                last_query_queue = last_observation.get("query_queue", "")
                last_queries = f"Last queries: {', '.join(last_query_queue)}"
            else:
                last_queries = ''
                last_action=''
            rag_record = []
            # ─── zoom_in_focal_retrieve2 ───
            if agent_flag[0]:
                rag_view_result = self.zoom_in_focal_retrieve.run(
                    root_query=root_query,
                    query_queue=query_queue,
                    rag_results=rag_results,
                    last_thought_information=last_queries,
                    known_info_rag=short_memory,
                    benchmark=bm,
                    haystack_session_ids=haystack_session_ids,
                )
                rag_dia = rag_view_result.get("useful_evidence", [])
                rag_view_thinking = rag_view_result.get("thinking", [])
                rag_view_missing_information = rag_view_result.get("missing_information", [])
                rag_dia_ids = [r.get("dia_id", "") for r in rag_dia]
                temp_useful_rag.extend(rag_dia)
                rag_view_record = {
                    "rag_view_useful_dia_ids": rag_dia_ids,
                    "rag_view_thinking": rag_view_thinking,
                    "rag_view_missing_information": rag_view_missing_information,
                }
                rag_record.append(rag_view_record)

            # ─── zoom_out_context_expansion ───
            if agent_flag[1] and len(temp_useful_rag) > 0:
                K = self.middle_scale
                middle_view_result = zoom_out_context_expansion_agent(
                    root_query=root_query,
                    query_queue=query_queue,
                    temp_useful_rag=temp_useful_rag,
                    known_info_rag=short_memory,
                    model=model,
                    conv_id=self.conv_id,
                    K=K,
                    temperature=0.0,
                    benchmark=bm,
                    haystack_session_ids=haystack_session_ids,
                )
                middle_dia = middle_view_result.get("useful_evidence", [])
                middle_view_thinking = middle_view_result.get("thinking", "")
                middle_view_missing = middle_view_result.get("missing_information", "")
                existing_ids = {r.get("dia_id", "") for r in temp_useful_rag}
                for r in middle_dia:
                    if r.get("dia_id", "") not in existing_ids:
                        temp_useful_rag.append(r)
                        existing_ids.add(r.get("dia_id", ""))
                middle_view_record = {
                    "middle_view_useful_dia_ids": [r.get("dia_id", "") for r in middle_dia],
                    "middle_view_thinking": middle_view_thinking,
                    "middle_view_missing_information": middle_view_missing,
                }
                rag_record.append(middle_view_record)

            # ─── panoramic_visual_grounding ───
            if agent_flag[2] and bm != "longmemeval":
                visual_ocr_result = panoramic_visual_grounding(
                    root_query=root_query,
                    query_queue=query_queue,
                    temp_useful_rag=temp_useful_rag,
                    known_info_rag=short_memory,
                    model=model,
                    conv_id=self.conv_id,
                    seen_session=visual_seen_session,
                    max_view_sessions=6,
                    temperature=0.0,
                    benchmark=bm,
                    haystack_session_ids=haystack_session_ids,
                )
                visual_ocr_dia = visual_ocr_result.get("useful_evidence", [])
                visual_ocr_thinking = visual_ocr_result.get("thinking", "")
                visual_ocr_missing = visual_ocr_result.get("missing_information", "")
                viewed = visual_ocr_result.get("add_view_sessions", [])
                sessions_to_view = visual_ocr_result.get("sessions_to_view", [])
                # visual_seen_session.extend(s for s in viewed if s not in visual_seen_session)
                existing_ids = {r.get("dia_id", "") for r in temp_useful_rag}
                for r in visual_ocr_dia:
                    if r.get("dia_id", "") not in existing_ids:
                        temp_useful_rag.append(r)
                        existing_ids.add(r.get("dia_id", ""))
                visual_ocr_record = {
                    "visual_ocr_useful_dia_ids": [r.get("dia_id", "") for r in visual_ocr_dia],
                    "visual_ocr_thinking": visual_ocr_thinking,
                    "visual_ocr_missing_information": visual_ocr_missing,
                    "add_view_sessions": viewed,
                    "visual_ocr_viewed_sessions": sessions_to_view,
                }
                rag_record.append(visual_ocr_record)

            # ─── 更新 known_info_rag ───For thought agent
            temp_short_memory_dia_ids = [r.get("dia_id", "") for r in temp_useful_rag]
            act_record = {
                "step": step,
                "query_queue": list(query_queue),
                "rag_record": rag_record,
                "temp_short_memory_dia_ids": temp_short_memory_dia_ids,
            }

            # ─── Observation ───给出是否能回答的结果
            obs = judge_agent(query, temp_useful_rag,
                short_memory, model, fail_queue_trajectory=fail_queue_trajectory, temperature=0.0,
                conv_memory=conv_memory, benchmark=bm, haystack_session_ids=haystack_session_ids)
            useful_evidence = obs.get("useful_evidence", [])#TODO:这里可以做个知识更新
            new_queries = obs.get("new_queries", [])
            action = obs.get("action", [])
            fail_query_flag = True
            if len(useful_evidence) > 0:#增加new short memory
                for i in temp_useful_rag:# 遍历临时记忆
                    if i['dia_id'] in useful_evidence and i['dia_id'] not in short_memory_dia_ids:#如果临时记忆有的，但是短期记忆没有的
                        short_memory.append(i)
                        fail_query_flag = False
            
            print(f"No.{step}:")
            if fail_query_flag:
                fail_queue_trajectory.append({"last_action":last_action,"query_queue":query_queue,"rag_view_report":rag_view_missing_information})#TODO:判断真不能靠这个判断obs agent来判断
                fail_queue_trajectory=fail_queue_trajectory[:3]
            obs_record = {
                "can_answer": obs["can_answer"],
                "useful_evidence": useful_evidence,
                "short_memory": [i['dia_id'] for i in short_memory],
                "action": action,
                "new_queries": new_queries,
            }

            obs_thinking = obs["thinking"]
            trajectory.append({
                "Act": act_record,
                "Observation": obs_record,
                "Thought": obs_thinking,
            })
            # 轨迹记忆
            trajectory_memory_dia_ids = [r.get("dia_id", "") for r in trajectory_memory]
            for i in short_memory:# 遍历临时记忆
                if i['dia_id'] not in trajectory_memory_dia_ids:#如果临时记忆有的，但是短期记忆没有的
                    trajectory_memory.append(i)
            query_queue_last = query_queue
            query_queue = new_queries
            # ─── 机械判断 ───
            if obs["can_answer"] or step == self.max_step - 1:
                short_memory_dia_ids_out = [m.get("dia_id", "") for m in short_memory]
                accepted = try_responder_answer_from_evidence(
                    query=query,
                    short_memory=short_memory,
                    model=model,
                    obs_report=obs_thinking,
                    additional_information=additional_information,
                    benchmark=bm,
                    haystack_session_ids=haystack_session_ids,
                )
                if accepted is None:
                    continue
                answer, answer_thinking = accepted
                prompt = return_answer_agent_prompt(
                    query,
                    short_memory,
                    obs_report=obs_thinking or "",
                    additional_information=additional_information,
                    benchmark=bm,
                    haystack_session_ids=haystack_session_ids,
                )
                return MemCoTExit(
                    kind="evidence_ok",
                    output_dir=output_dir,
                    trajectory=trajectory,
                    prompt=prompt,
                    model=model,
                    benchmark=bm,
                    answer=answer,
                    answer_thinking=answer_thinking,
                    steps=step,
                    final_evidence=short_memory_dia_ids_out,
                    final_query_queue=list(query_queue_last),
                )
        short_memory_dia_ids_out = [m.get("dia_id", "") for m in short_memory]
        prompt = return_conv_answer_agent_prompt(
            root_query=root_query,
            full_conv=self.full_conv,
            benchmark=bm,
        )
        return MemCoTExit(
            kind="fallback",
            output_dir=output_dir,
            trajectory=trajectory,
            prompt=prompt,
            model=model,
            benchmark=bm,
            full_conv=self.full_conv,
            max_step=self.max_step,
            final_evidence=short_memory_dia_ids_out,
            final_query_queue=list(query_queue),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct + RAG (naive/lightrag) on LoCoMo")
    parser.add_argument("query", nargs="*", help="Query")
    parser.add_argument("-c", "--conv", default="conv-26", help="Conversation ID")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--max-step", type=int, default=MAX_LOOP_STEP)
    parser.add_argument("-k", "--rag-topk", type=int, default=10, help=f"RAG top-k (default: {10})")
    parser.add_argument(
        "--rag-type",
        choices=RAG_TYPE_CHOICES,
        default=RAG_TYPE_LIGHTRAG,
        help=f"RAG 类型: naive=向量检索(pkl@naiverag), lightrag=LightRAG 知识图谱 (default: {RAG_TYPE_LIGHTRAG})",
    )
    parser.add_argument(
        "--rag-base",
        default=DEFAULT_LIGHTRAG_WORKING_BASE,
        help=f"LightRAG 索引根目录 (default: {DEFAULT_LIGHTRAG_WORKING_BASE})",
    )
    parser.add_argument(
        "--agent-flag",
        default=DEFAULT_AGENT_FLAG,
        help=f"5 位 0/1 串控制 agent 开关: rag_view/middle_view/full_view/agentic_graph/visual_search (default: {DEFAULT_AGENT_FLAG})",
    )
    parser.add_argument(
        "--middle-scale",
        type=int,
        default=3,
        help=f"Middle view agent scale K (default: 3)",
    )
    parser.add_argument(
        "--img-index-base",
        default=DEFAULT_IMG_INDEX_BASE,
        help=f"图片索引根目录 (default: {DEFAULT_IMG_INDEX_BASE})",
    )
    # benchmark
    parser.add_argument(
        "--benchmark",
        default="locomo",
        type=str,
        help="Benchmark name for prompt selection (default: locomo)",
    )
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else "When did Caroline go to the LGBTQ support group?"

    # 构建 working_dir
    conv_id = args.conv
    workspace = _conv_id_to_workspace(conv_id)
    working_dir = os.path.join(args.rag_base, workspace)
    print(f"working_dir: {working_dir}")

    # 载入 LightRAG（仅初始化一次，后续检索复用同一实例）
    rag = None
    if args.rag_type == RAG_TYPE_LIGHTRAG:
        rag = create_lightrag(working_dir)
    category = 0
    agent_flag_str = args.agent_flag
    if args.benchmark == "longmemeval":
        full_conv = _build_full_conv_context_longmemeval(conv_id)
    else:
        full_conv = _build_full_conv_context(conv_id)
    img_retriever = None
    if len(agent_flag_str) > 4 and agent_flag_str[4] == "1":
        print("初始化视觉搜索...")
        img_retriever = create_img_retriever(conv_id, img_index_base=args.img_index_base)
    memcot = MemCoT(
        conv_id=conv_id,
        model=args.model,
        category=category,
        benchmark=args.benchmark,
        working_dir=working_dir,
        rag_top_k=args.rag_topk,
        rag_type=args.rag_type,
        rag=rag,
        full_conv=full_conv,
        img_retriever=img_retriever,
        middle_scale=args.middle_scale,
        max_step=args.max_step,
    )
    try:
        exit_state = memcot.run(
            query=query,
            model=args.model,
            output_dir=args.output_dir,
            agent_flag_str=agent_flag_str,
            benchmark=args.benchmark,
        )
        finalize_memcot_exit(exit_state)
    finally:
        finalize_lightrag(rag)

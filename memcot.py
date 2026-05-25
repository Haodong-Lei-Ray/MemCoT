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
from tool.longmemevaltool import rewrite_first_person_to_user
from tool.rag import (
    DEFAULT_IMG_INDEX_BASE,
    RAG_TYPE_CHOICES,
    RAG_TYPE_LIGHTRAG,
    RAG_TYPE_NAIVE,
    load_rag_retrieve,
    create_img_retriever,
    finalize_lightrag,
    get_rag_event_loop,
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
    ZoomOutContextExpansion,
    JudgeAgent,
    answer_agent,
    panoramic_visual_grounding,
    return_answer_agent_prompt,
    return_conv_answer_agent_prompt,
    _parse_json_from_llm,
    WRONG_LIST,
)
from global_methods import run_chatgpt
from tool.show.cil import grey_print, morandi_print, morandi_blue_print

RAG_CONFIG_PATH = str(PROJECT_ROOT / "config" / "rag" / "locomolightrag.json")
MEMCOT_CONFIG_PATH = str(PROJECT_ROOT / "config" / "memcot.json")

def _load_memcot_config(path: str | Path) -> dict:
    path = Path(path)
    defaults = {
        "agent_flag": "110",
        "max_step": 10,
        "middle_scale": 4,
        "img_index_base": DEFAULT_IMG_INDEX_BASE,
    }
    if not path.exists():
        raise FileNotFoundError(f"MemCoT config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    defaults.update(cfg)
    return defaults

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

def _save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def answer_memcot_exit(exit_state: MemCoTExit) -> dict:
    """
    根据 MemCoT 退出状态完成 Responder 收尾（使用 exit_state.prompt 与 model/benchmark）。
    evidence_ok：run_chatgpt + 与 answer_agent 一致的解析。
    fallback：run_chatgpt，等价于 conv_answer_agent。
    """
    prompt = exit_state.prompt
    if prompt is None:
        raise ValueError("MemCoTExit.prompt 必须在 finalize 前设置")
    model = exit_state.model
    benchmark = exit_state.benchmark or "locomo"
    if exit_state.kind == "evidence_ok":
        resp = run_chatgpt(
            prompt, model=model, num_tokens_request=1024, temperature=0.0
        )
        if benchmark == "locomo" or benchmark == "openclaw":
            out = _parse_json_from_llm(resp)
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
        print(f"Answer: {out_answer}")
        return result

    assert exit_state.full_conv is not None

    resp = run_chatgpt(
        prompt, model=model, num_tokens_request=64, temperature=0.0
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
    绑定「同一对话 + RAG 基础设施」：`model` / `benchmark` 等在 `__init__` 绑定。
    `run(query, output_dir=..., agent_flag_str=...)` 只接本次 QA 的可变项。
    """

    def __init__(
        self,
        model: str = None,
        memcot_file_path: str = MEMCOT_CONFIG_PATH,
        rag_file_path: str = RAG_CONFIG_PATH,
        conv_id: str | None = None,
        rag_top_k = None,
        verbose: bool = True,
    ):
        # 初始化RAG基础设施
        self.ragretriever, self.benchmark, self.conversation_base, self.rag_base, self.rag_top_k = load_rag_retrieve(
            rag_file_path=rag_file_path,
            top_k=rag_top_k,
        )
        self.conv_id = None
        self.conversation = None
        self.img_retriever = None
        self.verbose = verbose

        memcot_cfg = _load_memcot_config(memcot_file_path)
        agent_flag = str(memcot_cfg["agent_flag"])
        max_step = int(memcot_cfg["max_step"])
        middle_scale = int(memcot_cfg["middle_scale"])
        self.img_index_base = str(memcot_cfg["img_index_base"])

        #通用设置
        self.agent_flag = agent_flag
        self.max_step = max_step
        if model is None:
            # 读取memcot_cfg里agent_config的model_name
            agent_config = memcot_cfg.get("agent_config", {})
            model = agent_config.get("model_name", None)
            #警告一下，如果modelw为None的话
            if model is None:
                raise ValueError(f"model_name 为空，请检查 {memcot_file_path}文件")
        self.model = model

        # zoom_in_focal_retrieve
        self.zoom_in_focal_retrieve = ZoomInFocalRetrieve(self.model, temperature=1.0)
        # zoom_out_context_expansion
        if agent_flag[1] == "1":
            self.zoom_out_context_expansion = ZoomOutContextExpansion(
                self.model, temperature=0.0, middle_scale=middle_scale
            )
        # judge_agent
        self.judge_agent = JudgeAgent(self.model, temperature=0.0)

        if conv_id:
            self.switch_session(conv_id=conv_id)


    def try_responder_answer_from_evidence(
        self,
        prompt: str,
    ) -> Optional[tuple[str, str]]:
        """
        Responder：在固定证据上调用 answer_agent；使用实例上的 model / benchmark。
        答案为空或在 WRONG_LIST 中则返回 None。
        """
        ans = answer_agent(
            prompt,
            model=self.model,
            benchmark=self.conversation.benchmark,
        )
        answer = ans.get("answer", "")
        thinking = ans.get("thinking", "")
        return answer, thinking

    def add_category_information(self, category: int) -> str:
        """根据 QA category 返回额外提示信息，用于引导各 Agent 的推理方向。"""
        if category == 2:
            return "Use DATE of CONVERSATION to answer with an approximate date."
        if category == 5:
            return 'If the answer is not mentioned in the conversation, answer "Not mentioned in the conversation".'
        return ""
    
    def switch_session(self, idx: int = None, conv_id: str = None):
        if idx is not None:
            if hasattr(self.ragretriever, "get_session_list"):
                data = self.ragretriever.get_session_list()
                sessions = data.get("sessions", [])
                target = next((s for s in sessions if s.get("index") == idx), None)
                if not target:
                    raise ValueError(f"Session with index {idx} not found.")
                conv_id = target.get("sessionId")
            else:
                raise ValueError("Current retriever does not support switching by index.")
                
        if not conv_id:
            raise ValueError("Must provide either idx or conv_id to switch_session.")
            
        self.conv_id = conv_id
        from agent.conversation import Conversation
        self.conversation = Conversation(conv_id, self.benchmark, self.conversation_base, self.rag_base)
        if hasattr(self.ragretriever, "load_rag"):
            try:
                self.ragretriever.load_rag(conv_id, self.rag_top_k)
            except FileNotFoundError as e:
                print(f"[🦉 MemCoT] Warning: {e}. RAG database not found. Please run 'add --idx <idx>' to build it.")
        
        # panoramic_visual_grounding
        if self.agent_flag[2] == "1":
            if self.verbose:
                print("初始化视觉搜索...")
            self.img_retriever = create_img_retriever(self.conv_id, img_index_base=self.img_index_base)

    def run(
        self,
        query: str,
        output_dir: Optional[str] = None,
        category: int = 0
    ) -> MemCoTExit:
        """
        单次 QA：在已绑定的 conv / RAG 上检索并回答。
        返回 MemCoTExit；请用 answer_memcot_exit(exit_state) 落盘。
        """
        assert output_dir is not None
        os.makedirs(output_dir, exist_ok=True)

        if self.conversation.benchmark == "longmemeval":
            query = rewrite_first_person_to_user(query)
        root_query = query
        query_queue = [query]
        agent_flag = [c == "1" for c in self.agent_flag]
        # Short memory
        short_semantic_memory: list[dict] = [] # 存对query的短期记忆
        fail_episondic_queue_trajectory: list[dict] = []
        trajectory_memory: list[dict] = [] 
        trajectory: list[dict] = [] 
        additional_information = self.add_category_information(category)
        conv_memory = []
        visual_seen_session = []
        get_rag_event_loop()

        # ====MemCoT核心循环
        for step in range(1, self.max_step + 1):
            rag_view_missing_information = []
            temp_short_memory: list[dict] = []
            short_memory_dia_ids = [m.get("dia_id", "") for m in short_semantic_memory]
            if self.verbose:
                grey_print(f"[🦉 MemCoT] No.{step}: Query_queue: {query_queue}")
            assert len(query_queue) != 0
            # ─── zoom_in_focal_retrieve1 ───
            rag_results, rag_result_list = self.ragretriever.retrieve_multi(query_queue)
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
            zoomin_record = []
            # ─── zoom_in_focal_retrieve2 ───
            if agent_flag[0]:
                if self.verbose:
                    morandi_blue_print(f"[🦉 MemCoT Zoom_in_focal_retrieve]")
                rag_view_result = self.zoom_in_focal_retrieve.run(
                    root_query=root_query,
                    query_queue=query_queue,
                    rag_results=rag_results,
                    last_thought_information=last_queries,
                    known_info_rag=short_semantic_memory,
                    conversation=self.conversation,
                )
                rag_dia = rag_view_result.get("useful_evidence", [])
                rag_view_thinking = rag_view_result.get("thinking", [])
                rag_view_missing_information = rag_view_result.get("missing_information", [])
                rag_dia_ids = [r.get("dia_id", "") for r in rag_dia]
                temp_short_memory.extend(rag_dia)
                if self.verbose:
                    morandi_blue_print(f"[🦉 MemCoT] thinking: {rag_view_thinking}")
                rag_view_record = {
                    "rag_view_useful_dia_ids": rag_dia_ids,
                    "rag_view_thinking": rag_view_thinking,
                    "rag_view_missing_information": rag_view_missing_information,
                }
                zoomin_record.append(rag_view_record)

            # ─── zoom_out_context_expansion ───
            if agent_flag[1] and len(temp_short_memory) > 0:
                if self.verbose:
                    morandi_print(f"[🦉 MemCoT Zoom_out_context_expansion]")
                middle_view_result = self.zoom_out_context_expansion.run(
                    root_query=root_query,
                    query_queue=query_queue,
                    temp_short_memory=temp_short_memory,
                    known_info_rag=short_semantic_memory,
                    conversation=self.conversation,
                )
                middle_dia = middle_view_result.get("useful_evidence", [])
                middle_view_thinking = middle_view_result.get("thinking", "")
                middle_view_missing = middle_view_result.get("missing_information", "")
                existing_ids = {r.get("dia_id", "") for r in temp_short_memory}
                for r in middle_dia:
                    if r.get("dia_id", "") not in existing_ids:
                        temp_short_memory.append(r)
                        existing_ids.add(r.get("dia_id", ""))
                if self.verbose:
                    morandi_print(f"[🦉 MemCoT] thinking: {middle_view_thinking}")
                middle_view_record = {
                    "middle_view_useful_dia_ids": [r.get("dia_id", "") for r in middle_dia],
                    "middle_view_thinking": middle_view_thinking,
                    "middle_view_missing_information": middle_view_missing,
                }
                zoomin_record.append(middle_view_record)

            # ─── panoramic_visual_grounding ───
            if agent_flag[2] and self.conversation.benchmark != "longmemeval":
                visual_ocr_result = panoramic_visual_grounding(
                    root_query=root_query,
                    query_queue=query_queue,
                    temp_short_memory=temp_short_memory,
                    known_info_rag=short_semantic_memory,
                    model=self.model,
                    seen_session=visual_seen_session,
                    max_view_sessions=6,
                    temperature=0.0,
                    conv_id=self.conversation.conv_id,
                    benchmark=self.conversation.benchmark,
                    haystack_session_ids=self.conversation.haystack_session_ids,
                )
                visual_ocr_dia = visual_ocr_result.get("useful_evidence", [])
                visual_ocr_thinking = visual_ocr_result.get("thinking", "")
                visual_ocr_missing = visual_ocr_result.get("missing_information", "")
                viewed = visual_ocr_result.get("add_view_sessions", [])
                sessions_to_view = visual_ocr_result.get("sessions_to_view", [])
                existing_ids = {r.get("dia_id", "") for r in temp_short_memory}
                for r in visual_ocr_dia:
                    if r.get("dia_id", "") not in existing_ids:
                        temp_short_memory.append(r)
                        existing_ids.add(r.get("dia_id", ""))
                visual_ocr_record = {
                    "visual_ocr_useful_dia_ids": [r.get("dia_id", "") for r in visual_ocr_dia],
                    "visual_ocr_thinking": visual_ocr_thinking,
                    "visual_ocr_missing_information": visual_ocr_missing,
                    "add_view_sessions": viewed,
                    "visual_ocr_viewed_sessions": sessions_to_view,
                }
                zoomin_record.append(visual_ocr_record)

            # ─── 更新 known_info_rag ───For thought agent
            temp_short_memory_dia_ids = [r.get("dia_id", "") for r in temp_short_memory]
            act_record = {
                "step": step,
                "query_queue": list(query_queue),
                "zoomin_record": zoomin_record,
                "temp_short_memory_dia_ids": temp_short_memory_dia_ids,
            }

            # ─── judge_agent ───给出是否能回答的结果
            obs = self.judge_agent.run(
                query=query,
                conversation=self.conversation,
                temp_useful_rag=temp_short_memory,
                short_memory=short_semantic_memory,
                fail_queue_trajectory=fail_episondic_queue_trajectory,
                conv_memory=conv_memory,
            )

            # Short Memory Evolution
            useful_evidence = obs.get("useful_evidence", [])
            new_queries = obs.get("new_queries", [])
            action = obs.get("action", [])
            fail_query_flag = True
            if len(useful_evidence) > 0:#增加new short memory
                for i in temp_short_memory:# 遍历临时记忆
                    if i['dia_id'] in useful_evidence and i['dia_id'] not in short_memory_dia_ids:#如果临时记忆有的，但是短期记忆没有的
                        short_semantic_memory.append(i)
                        fail_query_flag = False
            if fail_query_flag:
                fail_episondic_queue_trajectory.append({"last_action":last_action,"query_queue":query_queue,"rag_view_report":rag_view_missing_information})#TODO:判断真不能靠这个判断obs agent来判断
                fail_episondic_queue_trajectory=fail_episondic_queue_trajectory[:3]
            obs_record = {
                "can_answer": obs["can_answer"],
                "useful_evidence": useful_evidence,
                "short_memory": [i['dia_id'] for i in short_semantic_memory],
                "action": action,
                "new_queries": new_queries,
            }

            obs_thinking = obs["thinking"]
            if self.verbose:
                grey_print(f"[🦉 MemCoT] Observation: {obs_thinking}")
                grey_print(f"[🦉 MemCoT] can_answer: {obs['can_answer']}")
            trajectory.append({
                "Act": act_record,
                "Observation": obs_record,
                "Thought": obs_thinking,
            })
            # 轨迹记忆
            trajectory_memory_dia_ids = [r.get("dia_id", "") for r in trajectory_memory]
            for i in short_semantic_memory:# 遍历临时记忆
                if i['dia_id'] not in trajectory_memory_dia_ids:#如果临时记忆有的，但是短期记忆没有的
                    trajectory_memory.append(i)
            query_queue_last = query_queue
            query_queue = new_queries
            # ─── 机械判断 ───
            if obs["can_answer"] or step == self.max_step - 1:
                short_memory_dia_ids_out = [m.get("dia_id", "") for m in short_semantic_memory]
                prompt = return_answer_agent_prompt(
                    query,
                    short_semantic_memory,
                    obs_report=obs_thinking,
                    additional_information=additional_information,
                    conversation=self.conversation,
                )
                ans = self.try_responder_answer_from_evidence(
                    prompt=prompt,
                )
                answer, answer_thinking = ans
                if answer == '' or answer in WRONG_LIST:
                    continue
                return MemCoTExit(
                    prompt=prompt,
                    #次要
                    kind="evidence_ok",
                    output_dir=output_dir,
                    trajectory=trajectory,
                    model=self.model,
                    benchmark=self.conversation.benchmark,
                    answer=answer,
                    answer_thinking=answer_thinking,
                    steps=step,
                    final_evidence=short_memory_dia_ids_out,
                    final_query_queue=list(query_queue_last),
                )
        short_memory_dia_ids_out = [m.get("dia_id", "") for m in short_semantic_memory]
        prompt = return_conv_answer_agent_prompt(
            root_query=root_query,
            conversation=self.conversation,
        )
        return MemCoTExit(
            prompt=prompt,
            #次要
            kind="fallback",
            output_dir=output_dir,
            trajectory=trajectory,
            model=self.model,
            benchmark=self.conversation.benchmark,
            full_conv=self.conversation.full_conv,
            max_step=self.max_step,
            final_evidence=short_memory_dia_ids_out,
            final_query_queue=list(query_queue),
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ReAct + RAG (naive/lightrag) on LoCoMo")
    parser.add_argument("query", nargs="*", help="Query")
    parser.add_argument("-m", "--model", default="gpt-4o-mini")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument(
        "--rag-config",
        default=RAG_CONFIG_PATH,
        help=f"Path to rag config json (default: {RAG_CONFIG_PATH})",
    )
    parser.add_argument(
        "--memcot-config",
        default=MEMCOT_CONFIG_PATH,
        help=f"Path to memcot config json (default: {MEMCOT_CONFIG_PATH})",
    )
    # benchmark
    parser.add_argument("-c", "--conv-id", default="conv-26", help="Conversation ID")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress debug prints")
    args = parser.parse_args()
    query = " ".join(args.query)
    memcot = MemCoT(
        model=args.model,
        memcot_file_path=args.memcot_config,
        rag_file_path=args.rag_config,
        rag_top_k=10,
        conv_id=None,
        verbose=not args.quiet,
    )
    memcot.switch_session(conv_id=args.conv_id)
    try:
        exit_state = memcot.run(
            query=query,
            output_dir=args.output_dir
        )
        answer_memcot_exit(exit_state)
    finally:
        if getattr(memcot.ragretriever, "rag_type", None) == RAG_TYPE_LIGHTRAG:
            finalize_lightrag(memcot.ragretriever.light_rag)

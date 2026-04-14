#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import signal
import requests
from pathlib import Path

# 确保能正确导入项目内的模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from tool.show.cil import slight_green_print, grey_print, red_print

# --- Configuration ---
PORT = 8088
BASE_URL = f"http://127.0.0.1:{PORT}"
PID_FILE = os.path.join(PROJECT_ROOT, ".memcot.pid")
LOG_FILE = os.path.join(PROJECT_ROOT, "memcot_daemon.log")

# --- FastAPI App (Daemon) ---
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    # Global state
    memcot_instance = None
    
    class AddRequest(BaseModel):
        idx: int
        
    class SearchRequest(BaseModel):
        query: str
        output_dir: str

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global memcot_instance
        from tool.rag.rag import build_rag_retrieve
        from memcot import MemCoT
        
        rag_cfg = os.path.join(PROJECT_ROOT, "config", "rag", "openclawnaiverag.json")
        memcot_cfg = os.path.join(PROJECT_ROOT, "config", "memcot.json")
        grey_print(f"Loading RAG config from {rag_cfg}")
        
        # 临时获取 session 列表以确定 conv_id
        temp_retriever, _ = build_rag_retrieve(rag_file_path=rag_cfg)
        data = temp_retriever.get_session_list()
        sessions = data.get("sessions", [])
        if not sessions:
            red_print("Warning: No sessions found. MemCoT initialized without a valid conv_id.")
            conv_id = "default"
        else:
            conv_id = sessions[0].get("sessionId")
            
        grey_print(f"Initializing MemCoT with conv_id: {conv_id}")
        memcot_instance = MemCoT(
            memcot_file_path=memcot_cfg,
            rag_file_path=rag_cfg,
            conv_id=conv_id,
        )
        slight_green_print("MemCoT Daemon initialized successfully.")
        yield
        slight_green_print("MemCoT Daemon shutting down.")

    app = FastAPI(lifespan=lifespan)

    @app.get("/status")
    def status():
        return {"status": "running"}

    @app.get("/session")
    def get_session():
        if not memcot_instance:
            raise HTTPException(status_code=500, detail="MemCoT not initialized")
        # 后台执行获取列表
        data = memcot_instance.ragretriever.get_session_list()
        return data

    @app.post("/add")
    def add_session(req: AddRequest):
        if not memcot_instance:
            raise HTTPException(status_code=500, detail="MemCoT not initialized")
        try:
            # 后台执行 embedding
            memcot_instance.ragretriever.build_rag(idx=req.idx)
            return {"status": "success", "message": f"Session {req.idx} embedded successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search")
    def search_session(req: SearchRequest):
        if not memcot_instance:
            raise HTTPException(status_code=500, detail="MemCoT not initialized")
        from memcot import finalize_memcot_exit
        
        try:
            slight_green_print(
                f"[🦉 MemCoT]🔍 Searching for: '{req.query}'...\n"
                f"🤔OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL')}\n"
                f"🤔OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY')}\n"
            )
            # 执行 MemCoT 检索和推理
            exit_state = memcot_instance.run(
                query=req.query,
                output_dir=req.output_dir
            )
            qa_flag = False
            if not qa_flag:
                # 我这里返回exit_state.prompt就行
                result = {
                    "prompt": exit_state.prompt,
                    "kind": exit_state.kind,
                    "stopped_reason": "prompt_only",
                }
            else:
                result = finalize_memcot_exit(exit_state)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

except ImportError:
    app = None


# --- CLI Client ---

def is_running():
    if not os.path.exists(PID_FILE):
        return False
    try:
        res = requests.get(f"{BASE_URL}/status", timeout=2)
        return res.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def cmd_start(args):
    if is_running():
        print("MemCoT daemon is already running.")
        return
        
    print("Starting MemCoT daemon...")
    cmd = [sys.executable, os.path.abspath(__file__), "serve"]
    
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    
    # 每次启动时清空日志文件
    with open(LOG_FILE, "w") as f:
        pass
    
    process = subprocess.Popen(
        cmd,
        stdout=open(LOG_FILE, "a"),
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid if os.name == 'posix' else None
    )
    
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))
        
    print(f"Daemon started with PID {process.pid} in the background.")
    print(f"Logs are automatically written to {LOG_FILE}")
    print("Use 'python memcot_cil.py logshow' to monitor the logs in real-time.")

def cmd_stop(args):
    if not os.path.exists(PID_FILE):
        print("MemCoT daemon is not running (no PID file).")
        return
        
    with open(PID_FILE, "r") as f:
        pid_str = f.read().strip()
        
    if not pid_str:
        print("Invalid PID file.")
        return
        
    pid = int(pid_str)
    print(f"Stopping daemon (PID {pid})...")
    try:
        if os.name == 'posix':
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        print("Daemon stopped.")
    except ProcessLookupError:
        print("Process not found. It may have already crashed.")
    except Exception as e:
        print(f"Error stopping process: {e}")
        
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

def cmd_status(args):
    if is_running():
        print(f"MemCoT daemon is RUNNING at {BASE_URL}")
    else:
        print("MemCoT daemon is STOPPED.")

def cmd_session(args):
    if not is_running():
        print("Daemon is not running. Please run 'memcot start' first.")
        return
    try:
        res = requests.get(f"{BASE_URL}/session")
        res.raise_for_status()
        
        # 在前台 CLI 打印精美表格
        from tool.show import cil
        cil.show_session_list(res.json())
    except Exception as e:
        print(f"Error fetching sessions: {e}")

def cmd_add(args):
    if not is_running():
        print("Daemon is not running. Please run 'memcot start' first.")
        return
    try:
        print(f"Requesting to build RAG for idx {args.idx}...")
        res = requests.post(f"{BASE_URL}/add", json={"idx": args.idx})
        res.raise_for_status()
        print(res.json().get("message", "Success"))
    except Exception as e:
        print(f"Error adding session: {e}")
        if 'res' in locals() and res.status_code != 200:
            print(res.text)

def cmd_search(args):
    if not is_running():
        print("Daemon is not running. Please run 'memcot start' first.")
        return
    try:
        res = requests.post(f"{BASE_URL}/search", json={
            "query": args.query,
            "output_dir": args.output_dir
        })
        res.raise_for_status()
        data = res.json()
        
        if data.get("prompt"):
            slight_green_print("\n[🦉 MemCoT Prompt]")
            print(data.get("prompt"))
        else:
            raise ValueError("No prompt returned from MemCoT")
            
    except Exception as e:
        print(f"Error searching: {e}")
        print(f"OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL')}")
        print(f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY')}")
        if 'res' in locals() and res.status_code != 200:
            print(res.text)

def cmd_logshow(args):
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} does not exist yet.")
        return
    try:
        # 进入备用屏幕（Alternate Screen）
        sys.stdout.write("\033[?1049h")
        sys.stdout.flush()
        # 从第一行开始输出，并持续追踪
        subprocess.run(["tail", "-n", "+1", "-f", LOG_FILE])
    except KeyboardInterrupt:
        pass
    finally:
        # 退出备用屏幕，恢复终端原状
        sys.stdout.write("\033[?1049l")
        sys.stdout.flush()

def cmd_serve(args):
    """Internal command to run the uvicorn server."""
    if app is None:
        print("FastAPI is not installed. Cannot start daemon.")
        sys.exit(1)
    uvicorn.run(app, host="127.0.0.1", port=PORT)

def main():
    parser = argparse.ArgumentParser(description="MemCoT Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # start
    parser_start = subparsers.add_parser("start", help="Start the MemCoT daemon in the background")
    parser_start.set_defaults(func=cmd_start)
    
    # stop
    parser_stop = subparsers.add_parser("stop", help="Stop the MemCoT daemon")
    parser_stop.set_defaults(func=cmd_stop)
    
    # status
    parser_status = subparsers.add_parser("status", help="Check daemon status")
    parser_status.set_defaults(func=cmd_status)
    
    # session
    parser_session = subparsers.add_parser("session", help="List all sessions")
    parser_session.set_defaults(func=cmd_session)
    
    # add
    parser_add = subparsers.add_parser("add", help="Build RAG for a specific session index")
    parser_add.add_argument("--idx", type=int, required=True, help="Session index to add")
    parser_add.set_defaults(func=cmd_add)
    
    # search
    parser_search = subparsers.add_parser("search", help="Search in the loaded session")
    parser_search.add_argument("-q", "--query", type=str, required=True, help="Query string")
    parser_search.add_argument("-o", "--output-dir", type=str, default="./output", help="Output directory for results")
    parser_search.set_defaults(func=cmd_search)
    
    # logshow
    parser_logshow = subparsers.add_parser("logshow", help="Monitor the daemon logs in real-time")
    parser_logshow.set_defaults(func=cmd_logshow)
    
    # help
    parser_help = subparsers.add_parser("help", help="Show this help message")
    parser_help.set_defaults(func=lambda args: parser.print_help())
    
    # serve (internal)
    parser_serve = subparsers.add_parser("serve", help=argparse.SUPPRESS)
    parser_serve.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)

if __name__ == "__main__":
    main()

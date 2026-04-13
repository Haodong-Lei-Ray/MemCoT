"""LongMemEval 相关的轻量工具。"""

import re


def rewrite_first_person_to_user(text: str) -> str:
    """将 I/me/my 改写为 user 第三人称，与 LongMemEval RAG 索引中的 speaker 格式一致。"""
    t = text
    t = re.sub(r"\bI'm\b", "the user is", t, flags=re.IGNORECASE)
    t = re.sub(r"\bI've\b", "the user has", t, flags=re.IGNORECASE)
    t = re.sub(r"\bI'll\b", "the user will", t, flags=re.IGNORECASE)
    t = re.sub(r"\bI'd\b", "the user would", t, flags=re.IGNORECASE)
    t = re.sub(r"\bI\b", "the user", t)
    t = re.sub(r"\bme\b", "the user", t, flags=re.IGNORECASE)
    t = re.sub(r"\bmy\b", "the user's", t, flags=re.IGNORECASE)
    return t

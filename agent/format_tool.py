import re
def fix_bracket_balance(s: str) -> str:
    """
    只修括号配对：
    - 栈里剩下的开括号 → 在末尾补对应的闭合
    - 遇到多余的闭合括号 → 在它前面补对应的开括号（较激进，可注释掉）
    """
    stack = []
    result = []
    i = 0

    while i < len(s):
        c = s[i]

        if c in '([{':
            stack.append(c)
            result.append(c)
            i += 1
            continue

        if c in ')]}':
            if not stack:
                # 多余的闭合 → 可选择在前面补开括号（激进修复）
                # 如果你不想修多余闭合，可以直接 continue 或 result.append(c)
                open_map = {')': '(', ']': '[', '}': '{'}
                result.append(open_map[c])   # 补开括号
                result.append(c)
                i += 1
                continue

            expected_open = stack[-1]
            close_map = {'(': ')', '[': ']', '{': '}'}
            if close_map.get(expected_open) == c:
                stack.pop()
                result.append(c)
            else:
                # 类型不匹配，很难自动决定，这里保守处理：直接放过
                result.append(c)
            i += 1
            continue

        # 其他字符直接保留
        result.append(c)
        i += 1

    # 补齐所有没关的开括号（最常见LLM问题）
    for open_c in reversed(stack):
        close_c = { '(':')', '[':']', '{':'}' }[open_c]
        result.append(close_c)

    return ''.join(result)

def fix_bracket_balance(s: str) -> str:
    # 你之前的版本（这里用保守版，只补闭合，不补开括号）
    stack = []
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if stack and { '(':')', '[':']', '{':'}' }[stack[-1]] == c:
                stack.pop()
    
    closing = ''.join({ '(':')', '[':']', '{':'}' }[c] for c in reversed(stack))
    return s + closing

def remove_trailing_commas(s: str) -> str:
    """
    移除 JSON 中不合法的尾随逗号：在 ] 或 } 前的逗号。
    """
    # 匹配 , 后面只有空白和 ] 或 } 的情况
    s = re.sub(r',(\s*[\]}])', r'\1', s)
    return s


# 常见的 JSON 顶层 key（LLM 输出中数组未闭合时，下一个 key 通常是顶层 key）
# 仅匹配这些 key，避免误伤如 "thinking", "thinking_choice" 等正常 key-value 对
_UNCLOSED_ARRAY_NEXT_KEYS = (
    "useful_ids", "useful_evidence", "useful_id",
    "new_queries", "can_answer", "action",
    "answer",
)


def fix_unclosed_array_before_key(s: str) -> str:
    """
    修复 LLM 常见错误：数组最后一个元素后写了逗号，但忘记写 ]，直接跟了下一个顶层 key。
    例如：  "missing_information": ["...长字符串",
      "useful_ids": [0, 7]
    应改为 "missing_information": ["...长字符串"],
      "useful_ids": [0, 7]
    """
    keys_pat = "|".join(re.escape(k) for k in _UNCLOSED_ARRAY_NEXT_KEYS)
    # 匹配: " 后面 , 换行 空格 "key":
    pattern = rf'"(\s*,\s*\n\s*)"({keys_pat})"\s*:'
    
    def repl(m):
        # 将 ", 换行 "key" 改为 "], 换行 "key"
        return f'"]{m.group(1)}"{m.group(2)}":'
    
    return re.sub(pattern, repl, s)


def fix_missing_information_trailing_comma(s: str) -> str:
    """
    专门针对：
    "...长字符串",          ← 这里多余逗号
    "useful_ids": ...
    
    闭合数组并保留正确的逗号
    """
    return fix_unclosed_array_before_key(s)


def fix_escape_chars(text: str) -> str:
    """修复常见的转义字符问题"""
    replacements = [
        # ("\\'", "\'"),      # 修复转义的单引号
        # ('\\"', '\"'),      # 修复转义的双引号
        ("\\\'s", "'s"),
        ("\\\\", "\\"),    # 修复双反斜杠
        ("\\n", "\n"),     # 修复换行符
        ("\\t", "\t"),     # 修复制表符
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text
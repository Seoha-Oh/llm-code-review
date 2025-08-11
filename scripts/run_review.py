import os, re, json, subprocess, requests, pathlib, sys

REPO = os.getenv("GITHUB_REPOSITORY")
EVENT = json.load(open(os.getenv("GITHUB_EVENT_PATH")))
PR_NUM = EVENT["pull_request"]["number"]
HEAD_SHA = EVENT["pull_request"]["head"]["sha"]
BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

def sh(*cmd):
    return subprocess.check_output(list(cmd), text=True).strip()

def get_diff_unified0():
    # base 찾기 (기본 브랜치가 다르면 BASE_BRANCH 수정/입력)
    base = sh("git", "merge-base", f"origin/{BASE_BRANCH}", "HEAD")
    return subprocess.check_output(
        ["git", "diff", f"{base}...HEAD", "--unified=0"],
        text=True
    )

def parse_hunks(diff: str):
    """
    @@ -a,b +c,d @@ 포맷에서 new 파일 영역(+c,d)만 추출.
    return: [(path, start, end), ...]
    """
    sections = []
    cur_file = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:].strip()
        elif line.startswith("+++ /dev/null"):
            cur_file = None  # 삭제된 파일은 스킵
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if cur_file and m:
            start = int(m.group(1))
            length = int(m.group(2) or "1")
            end = start + length - 1
            sections.append((cur_file, start, end))
    return sections

def numbered_section(path, start, end, ctx=3):
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    lines = text.splitlines()
    s = max(1, start - ctx)
    e = min(len(lines), end + ctx)
    body = "\n".join(f"{i+1}: {lines[i]}" for i in range(s-1, e))
    return f'<SECTION file="{path}" start={start} end={end}>\n{body}\n</SECTION>'

def build_sections_payload():
    diff = get_diff_unified0()
    hunks = parse_hunks(diff)
    blocks = []
    for path, st, en in hunks:
        if not os.path.exists(path):  # rename/삭제 등
            continue
        block = numbered_section(path, st, en)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)[:180000]  # 토큰 보호

def build_messages():
    sys_prompt = pathlib.Path("scripts/prompt.md").read_text()
    sections = build_sections_payload()
    if not sections.strip():
        sections = "변경 섹션 없음"
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": sections}
    ]

def call_openai(messages):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 1200}
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type":"application/json"},
        data=json.dumps(payload),
        timeout=180
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    # 방어적 파싱
    start = content.find("{"); end = content.rfind("}")
    try:
        return json.loads(content[start:end+1])
    except Exception:
        return {"diagnosis": [], "issues"

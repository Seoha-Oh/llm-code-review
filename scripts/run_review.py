#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_review.py
- PR diff의 섹션을 "함수/메서드 경계" 기준으로 구성하고,
  너무 크면 오버랩 분할하되, 선언부(상단 변수/클래스 필드/임포트)까지 포함해
  LLM이 중간에서 끊기지 않도록 맥락을 제공합니다.
- OpenAI Chat Completions로 호출 → JSON 응답 → PR 요약/인라인 코멘트 업로드.
"""

import os, re, json, subprocess, requests, pathlib, sys
from collections import defaultdict

# ===== 환경 변수 =====
REPO = os.getenv("GITHUB_REPOSITORY")
EVENT = json.load(open(os.getenv("GITHUB_EVENT_PATH")))
PR_NUM = EVENT["pull_request"]["number"]
HEAD_SHA = EVENT["pull_request"]["head"]["sha"]

BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 섹션/맥락 파라미터(필요시 워크플로우에서 ENV로 조정)
MAX_LINES_PER_SECTION = int(os.getenv("MAX_LINES_PER_SECTION", "220"))  # 한 섹션 최대 라인 수
OVERLAP_LINES         = int(os.getenv("OVERLAP_LINES", "10"))           # 분할 시 오버랩
FUNC_CTX_BEFORE       = int(os.getenv("FUNC_CTX_BEFORE", "16"))         # 함수 시작 위 컨텍스트(선언/필드 포함)
NUM_CTX_LINES         = int(os.getenv("NUM_CTX_LINES", "6"))            # 섹션 주변 추가 컨텍스트
MAX_PAYLOAD_CHARS     = int(os.getenv("MAX_PAYLOAD_CHARS", "180000"))   # 한 번 호출 최대 페이로드
PER_FILE_CALL         = os.getenv("PER_FILE_CALL", "true").lower() == "true"  # 파일 단위로 여러 번 호출

# ===== 유틸 =====
def sh(*cmd: str) -> str:
    return subprocess.check_output(list(cmd), text=True).strip()

def get_diff_unified0() -> str:
    base = sh("git", "merge-base", f"origin/{BASE_BRANCH}", "HEAD")
    return subprocess.check_output(["git","diff",f"{base}...HEAD","--unified=0"], text=True)

def parse_hunks(diff: str):
    """
    @@ -a,b +c,d @@ 를 파싱해 새 코드 영역(+c,d)의 시작/길이를 추출.
    return: [(path, start, end), ...]
    """
    sections = []
    cur_file = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:].strip()
        elif line.startswith("+++ /dev/null"):
            cur_file = None
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if cur_file and m:
            start = int(m.group(1))
            length = int(m.group(2) or "1")
            sections.append((cur_file, start, start + length - 1))
    return sections

def load_lines(path: str):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    return text.splitlines()

# ===== ctags 기반 심볼 수집 =====
def have_ctags() -> bool:
    try:
        subprocess.check_output(["ctags", "--version"])
        return True
    except Exception:
        return False

def ctags_symbols(path: str):
    """
    universal-ctags 가정. `ctags -n -x <file>` 출력 파싱.
    라인 예: name  kind  line  file
    """
    try:
        out = subprocess.check_output(["ctags","-n","-x",path], text=True)
    except Exception:
        return []
    symbols = []
    for line in out.splitlines():
        # 느슨한 파싱: name kind line file...
        m = re.match(r"(\S+)\s+(\S+)\s+(\d+)\s+(.*)$", line)
        if not m: 
            continue
        name, kind, lno, _ = m.groups()
        try:
            lno = int(lno)
        except ValueError:
            continue
        symbols.append({"name": name, "kind": kind.lower(), "line": lno})
    symbols.sort(key=lambda x: x["line"])
    return symbols

def function_ranges_with_ctags(path: str, total_lines: int):
    """
    함수/메서드 시작 라인 목록 → (start, end) 범위 추정
    다음 심볼 시작 - 1 까지를 끝으로 본다(완벽하진 않지만 실용적).
    """
    if not have_ctags(): 
        return []
    syms = ctags_symbols(path)
    fn_starts = [s["line"] for s in syms if s["kind"] in ("function","method")]
    if not fn_starts:
        return []
    fn_starts.sort()
    ranges = []
    for i, s in enumerate(fn_starts):
        e = (fn_starts[i+1]-1) if i+1 < len(fn_starts) else total_lines
        ranges.append((s, e))
    return ranges

def class_ranges_with_ctags(path: str, total_lines: int):
    if not have_ctags():
        return []
    syms = ctags_symbols(path)
    cls_starts = [s["line"] for s in syms if s["kind"] in ("class","struct")]
    if not cls_starts:
        return []
    cls_starts.sort()
    ranges = []
    for i, s in enumerate(cls_starts):
        e = (cls_starts[i+1]-1) if i+1 < len(cls_starts) else total_lines
        ranges.append((s, e))
    return ranges

def enclosing_class_start(path: str, line: int, total_lines: int):
    """해당 라인이 속한 클래스 시작 라인(없으면 None)"""
    for cs, ce in class_ranges_with_ctags(path, total_lines):
        if cs <= line <= ce:
            return cs
    return None

# ===== 분할/확장 로직 =====
def split_with_overlap(start: int, end: int, max_lines=MAX_LINES_PER_SECTION, overlap=OVERLAP_LINES):
    """[start,end] 범위를 오버랩으로 분할"""
    n = end - start + 1
    if n <= max_lines:
        yield (start, end); return
    cur = start
    while cur <= end:
        ps = cur
        pe = min(end, cur + max_lines - 1)
        yield (ps, pe)
        if pe == end: break
        cur = max(pe - overlap + 1, pe + 1)

def expand_range_for_decls(path: str, func_start: int, func_end: int, total_lines: int):
    """
    함수 경계로부터 위쪽 선언/필드/임포트 컨텍스트를 포함하도록 시작 지점 확장.
    - 클래스 내부면 클래스 시작 또는 함수 시작-컨텍스트 중 더 아래를 사용.
    - 모듈/파일 상단 변수/임포트는 첫 함수 이전 구간에 있을 가능성이 커서,
      함수 시작 위로 FUNC_CTX_BEFORE 만큼 확장.
    """
    cls_start = enclosing_class_start(path, func_start, total_lines)
    anchor = func_start - FUNC_CTX_BEFORE
    if cls_start:
        anchor = max(cls_start, anchor)
    start = max(1, anchor)
    end = func_end
    return (start, end)

def sections_for_file(path: str, hunks_for_file: list):
    """
    주어진 파일에 대해:
    - ctags 함수 범위가 있으면 → hunk와 겹치는 함수 단위로 섹션 생성
    - 없으면 → hunk 범위를 윈도우 분할
    각 섹션은 추가 context(NUM_CTX_LINES)를 양 옆으로 포함해 넘버링.
    """
    if not os.path.exists(path):
        return []

    lines = load_lines(path)
    total = len(lines)
    func_ranges = function_ranges_with_ctags(path, total)

    # hunk들 합쳐서(신규 파일이면 사실상 1~끝) → 함수와 교집합 계산
    merged = merge_intervals(hunks_for_file)

    sections = []

    if func_ranges:
        for hs, he in merged:
            for fs, fe in func_ranges:
                if intervals_overlap(hs, he, fs, fe):
                    # 선언/필드 포함하도록 위쪽 확장
                    es, ee = expand_range_for_decls(path, fs, fe, total)
                    # 너무 크면 오버랩 분할
                    for ps, pe in split_with_overlap(es, ee):
                        sec = numbered_section(path, ps, pe, lines, ctx=NUM_CTX_LINES)
                        if sec:
                            sections.append((path, ps, pe, sec))
    else:
        # 함수 정보 없으면 hunk 자체를 안전 분할
        for hs, he in merged:
            for ps, pe in split_with_overlap(hs, he):
                sec = numbered_section(path, ps, pe, lines, ctx=NUM_CTX_LINES)
                if sec:
                    sections.append((path, ps, pe, sec))

    # 중복 제거(동일 범위)
    uniq = {}
    for p, s, e, t in sections:
        uniq[(p, s, e)] = t
    return list(uniq.items())  # [((path, start, end), section_text), ...]

def intervals_overlap(a1, a2, b1, b2):
    return not (a2 < b1 or b2 < a1)

def merge_intervals(spans):
    """[(s,e), ...] → 겹치는 구간 병합"""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le + 1:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def numbered_section(path: str, start: int, end: int, lines=None, ctx=NUM_CTX_LINES):
    """라인 번호가 포함된 섹션(컨텍스트 ±ctx 포함)"""
    try:
        if lines is None:
            lines = load_lines(path)
    except Exception:
        return None
    s = max(1, start - ctx)
    e = min(len(lines), end + ctx)
    body = "\n".join(f"{i+1}: {lines[i]}" for i in range(s-1, e))
    return f'<SECTION file="{path}" start={start} end={end}>\n{body}\n</SECTION>'

# ===== LLM 호출/리포팅 =====
def build_messages(payload_text: str):
    base_dir = pathlib.Path(__file__).resolve().parent  # ← 추가
    sys_prompt = (base_dir / "prompt.md").read_text(encoding="utf-8")  # ← 경로 수정
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": payload_text if payload_text.strip() else "변경 섹션 없음"}
    ]

def call_openai(messages):
    model = MODEL
    use_responses = model.startswith("o4")  # o4, o4-mini 계열

    if use_responses:
        url = "https://api.openai.com/v1/responses"
        payload = {
            "model": model,
            "input": messages,              # system/user 메시지 배열 그대로 OK
            "max_output_tokens": 1200,      # <-- responses 전용 이름
            # "temperature": 0.2,           # <-- 넣으면 400! (삭제)
        }
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1200,
        }

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=180,
    )

    # 에러 바디 로깅(원인 확인용)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        print("OpenAI API error:", r.status_code, json.dumps(err, ensure_ascii=False)[:4000])
        r.raise_for_status()

    data = r.json()

    # 응답 파싱
    if use_responses:
        try:
            content = data["output"][0]["content"][0]["text"]
        except Exception:
            # 안전 fallback
            content = data.get("output_text") or json.dumps(data, ensure_ascii=False)
    else:
        content = data["choices"][0]["message"]["content"]

    # 방어적 JSON 파싱
    start = content.find("{"); end = content.rfind("}")
    try:
        return json.loads(content[start:end+1])
    except Exception:
        return {"diagnosis": [], "issues": [], "overall_summary": content}

def post_summary(body: str):
    url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUM}/comments"
    requests.post(url,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                 "Accept":"application/vnd.github+json"},
        json={"body": body}
    ).raise_for_status()

def post_inline(issues: list):
    """단일/다중 라인 인라인 코멘트 + suggestion 지원"""
    url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUM}/comments"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
               "Accept":"application/vnd.github+json"}
    for it in issues[:60]:
        path = it.get("file")
        line = it.get("line")
        sline = it.get("start_line")
        eline = it.get("end_line", line)
        if not path or (not line and not sline):
            continue
        title = f"**{it.get('type','Issue')} ({it.get('severity','minor')})**"
        reason = it.get("reason","")
        suggestion = it.get("suggestion","")
        body = f"{title}\n{reason}\n\n{suggestion}"
        payload = {"body": body, "path": path, "side":"RIGHT", "commit_id": HEAD_SHA}
        if sline:
            payload["start_line"] = int(sline)
            payload["line"] = int(eline)
        else:
            payload["line"] = int(line)
        requests.post(url, headers=headers, json=payload).raise_for_status()

def summarize_diag(diag: list):
    if not diag: 
        return "발견된 요약 없음"
    out = []
    for d in diag:
        t = d.get("type","-"); c = d.get("count",0); s = d.get("summary","")
        out.append(f"- **{t}**: {c}건 — {s}")
    return "\n".join(out)

# ===== 메인: 파일/함수 경계 기반 섹션 생성 → 호출 → 업로드 =====
def build_payload_all_at_once(hunks_by_file):
    """모든 파일 섹션을 한 번에 합쳐서 전송 (소형 PR용)"""
    blocks = []
    for path, hunks in hunks_by_file.items():
        for (_, s, e), sec in sections_for_file(path, hunks):
            blocks.append(sec)
    return "\n\n".join(blocks)[:MAX_PAYLOAD_CHARS]

def per_file_calls(hunks_by_file):
    """파일별로 나눠 여러 번 호출(대형 PR/신규 큰 파일에 유리)"""
    all_issues, all_diag = [], []
    for path, hunks in hunks_by_file.items():
        secs = sections_for_file(path, hunks)
        if not secs:
            continue
        # 너무 크면 조각내어 여러 번 호출
        batch = []
        size = 0
        for (_, s, e), text in secs:
            if size + len(text) > MAX_PAYLOAD_CHARS and batch:
                res = call_openai(build_messages("\n\n".join(batch)))
                all_diag += res.get("diagnosis", [])
                all_issues += res.get("issues", [])
                batch = [text]; size = len(text)
            else:
                batch.append(text); size += len(text)
        if batch:
            res = call_openai(build_messages("\n\n".join(batch)))
            all_diag += res.get("diagnosis", [])
            all_issues += res.get("issues", [])
    return all_diag, all_issues

def main():
    diff = get_diff_unified0()
    hunks = parse_hunks(diff)
    if not hunks:
        post_summary("변경 섹션이 없어 리뷰를 생략합니다.")
        return

    # 파일별로 모으기
    hunks_by_file = defaultdict(list)
    for path, st, en in hunks:
        hunks_by_file[path].append((st, en))

    if PER_FILE_CALL:
        diag, issues = per_file_calls(hunks_by_file)
        post_summary("### 🤖 LLM Code Review 요약\n" + summarize_diag(diag))
        post_inline(issues)
    else:
        payload = build_payload_all_at_once(hunks_by_file)
        result = call_openai(build_messages(payload))
        post_summary("### 🤖 LLM Code Review 요약\n" + summarize_diag(result.get("diagnosis", [])) + 
                     "\n\n---\n" + result.get("overall_summary",""))
        post_inline(result.get("issues", []))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            post_summary(f"리뷰 작업 중 예외 발생: `{e}`")
        finally:
            raise

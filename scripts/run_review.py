#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_review.py
- PR diff를 함수/메서드 경계 기준 섹션으로 나눠 LLM에 전달
- 결과를 인라인 코멘트(변경 라인만) + PR 상단 요약(업서트)로 게시
- 요약 형식: Diagnosis(카테고리 집계/요약)만 표시
"""

import os, re, json, subprocess, requests, pathlib, sys
from collections import defaultdict, Counter

# ===== GitHub / 모델 설정 =====
REPO = os.getenv("GITHUB_REPOSITORY")
EVENT = json.load(open(os.getenv("GITHUB_EVENT_PATH")))
PR_NUM = EVENT["pull_request"]["number"]
HEAD_SHA = EVENT["pull_request"]["head"]["sha"]

BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ===== 섹션/맥락 파라미터 =====
MAX_LINES_PER_SECTION = int(os.getenv("MAX_LINES_PER_SECTION", "220"))
OVERLAP_LINES         = int(os.getenv("OVERLAP_LINES", "10"))
FUNC_CTX_BEFORE       = int(os.getenv("FUNC_CTX_BEFORE", "16"))
NUM_CTX_LINES         = int(os.getenv("NUM_CTX_LINES", "6"))
MAX_PAYLOAD_CHARS     = int(os.getenv("MAX_PAYLOAD_CHARS", "180000"))
PER_FILE_CALL         = os.getenv("PER_FILE_CALL", "true").lower() == "true"

# ===== 카테고리 집계 =====
ORDER = ["Precondition", "Runtime", "Optimization", "Security"]
SUMMARIES = {
    "Precondition": "코드 실행 전 입력값·상태·범위·동시성 조건 등을 사전에 검증하는 부분",
    "Runtime": "실행 중 NPE, 인덱스 범위 오류, 0으로 나눔, 자원 누수, 데드락·레이스 등 안정성 관련 문제",
    "Optimization": "불필요한 연산·I/O·동기화, 데이터 복사, N+1 쿼리, 부적절한 동기↔비동기 변환 등 성능 비효율",
    "Security": "시크릿·민감정보 노출, 경로 조작, SQL 인젝션, 안전하지 않은 직렬화/모듈 사용 등 보안 취약점",
}

def classify(issue_type: str, reason: str) -> str:
    """개별 이슈를 4개 카테고리 중 하나로 매핑(한/영 키워드 포함)."""
    t = (issue_type or "").lower()
    r = (reason or "").lower()
    txt = t + " " + r

    # Security
    if any(k in txt for k in [
        "secret","token","credential","path traversal","sql","injection","pii","serialize","pickle","jwt","xss","csrf",
        "비밀","시크릿","토큰","경로 조작","인젝션","민감정보","취약","권한 상승"
    ]):
        return "Security"

    # Optimization
    if any(k in txt for k in [
        "n+1","unnecessary io","copy","deepcopy","blocking","busy loop","complexity","async","synchronous","inefficient",
        "불필요","비효율","과도한 복사","동기화 남용","성능","느림"
    ]):
        return "Optimization"

    # Runtime
    if any(k in txt for k in [
        "npe","nullpointer","attributeerror","index","out_of_bounds","keyerror","zero","divide","/0","leak","deadlock","race","resource",
        "0으로","제로","분모 0","나눗셈 오류","인덱스 범위","자원 누수","데드락","레이스"
    ]):
        return "Runtime"

    # Precondition
    if any(k in txt for k in [
        "precondition","validate","validation","null check","range","bounds","thread-safety","input check","guard",
        "mutable default","가변 기본값","기본 인수","입력 검증","범위 검증","동시성 전제","사전조건"
    ]):
        return "Precondition"

    return "Precondition"

def build_aggregated_diagnosis(issues: list) -> list:
    """이슈 배열을 카테고리별로 합산하여 4개 항목만 반환."""
    counter = Counter()
    for it in issues or []:
        cat = classify(it.get("type",""), it.get("reason",""))
        counter[cat] += 1
    return [{"type": cat, "count": int(counter.get(cat, 0)), "summary": SUMMARIES[cat]} for cat in ORDER]

# --- suggestion helpers ---
SUGG_RX = re.compile(r"(?s)```suggestion\s*\n(.*?)\n```")

def _extract_suggestion_block(text: str) -> str:
    if not text:
        return ""
    m = SUGG_RX.search(text)
    return m.group(0).strip() if m else ""

def _is_multiline_sugg(block: str) -> bool:
    if not block:
        return False
    inner = SUGG_RX.search(block).group(1)
    return inner.count("\n") >= 1

# ===== 공통 유틸 =====
def sh(*cmd: str) -> str:
    return subprocess.check_output(list(cmd), text=True).strip()

def get_diff_unified0() -> str:
    base = sh("git", "merge-base", f"origin/{BASE_BRANCH}", "HEAD")
    return subprocess.check_output(["git","diff",f"{base}...HEAD","--unified=0"], text=True)

def parse_hunks(diff: str):
    """@@ -a,b +c,d @@ 블록을 파싱 → (path, start, end) 리스트"""
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

# ===== ctags 기반 심볼 =====
def have_ctags() -> bool:
    try:
        subprocess.check_output(["ctags", "--version"])
        return True
    except Exception:
        return False

def ctags_symbols(path: str):
    try:
        out = subprocess.check_output(["ctags","-n","-x",path], text=True)
    except Exception:
        return []
    symbols = []
    for line in out.splitlines():
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
    for cs, ce in class_ranges_with_ctags(path, total_lines):
        if cs <= line <= ce:
            return cs
    return None

# ===== 분할/확장 =====
def split_with_overlap(start: int, end: int, max_lines=MAX_LINES_PER_SECTION, overlap=OVERLAP_LINES):
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

def expand_range_for_decls(path: str, func_start: int, func_end: int, total_lines: int, func_ranges: list):
    """
    - 함수 시작 위로 FUNC_CTX_BEFORE줄 확장
    - 클래스 내부면 클래스 시작 이전으로 확장 금지
    - 바로 이전 함수의 끝 이전으로 확장 금지
    """
    cls_start = enclosing_class_start(path, func_start, total_lines)
    anchor = func_start - FUNC_CTX_BEFORE
    if cls_start:
        anchor = max(cls_start, anchor)
    prev_func_end = 0
    for fs, fe in func_ranges:
        if fe < func_start and fe > prev_func_end:
            prev_func_end = fe
    start = max(1, anchor, prev_func_end + 1)
    end = func_end
    return (start, end)

def intervals_overlap(a1, a2, b1, b2):
    return not (a2 < b1 or b2 < a1)

def merge_intervals(spans):
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
    try:
        if lines is None:
            lines = load_lines(path)
    except Exception:
        return None
    s = max(1, start - ctx)
    e = min(len(lines), end + ctx)
    body = "\n".join(f"{i+1}: {lines[i]}" for i in range(s-1, e))
    return f'<SECTION file="{path}" start={start} end={end}>\n{body}\n</SECTION>'

def sections_for_file(path: str, hunks_for_file: list):
    if not os.path.exists(path):
        return []
    lines = load_lines(path)
    total = len(lines)
    func_ranges = function_ranges_with_ctags(path, total)

    merged = merge_intervals(hunks_for_file)
    sections = []

    if func_ranges:
        for hs, he in merged:
            for fs, fe in func_ranges:
                if intervals_overlap(hs, he, fs, fe):
                    es, ee = expand_range_for_decls(path, fs, fe, total, func_ranges)
                    for ps, pe in split_with_overlap(es, ee):
                        sec = numbered_section(path, ps, pe, lines, ctx=NUM_CTX_LINES)
                        if sec:
                            sections.append((path, ps, pe, sec))
    else:
        for hs, he in merged:
            for ps, pe in split_with_overlap(hs, he):
                sec = numbered_section(path, ps, pe, lines, ctx=NUM_CTX_LINES)
                if sec:
                    sections.append((path, ps, pe, sec))

    # 포함/중복 제거
    sections.sort(key=lambda x: (x[0], x[1], x[2]))
    pruned = []
    for p, s, e, t in sections:
        if any(pp == p and ss == s and ee == e for (pp, ss, ee, _) in pruned):
            continue
        if any(pp == p and ss <= s and e <= ee for (pp, ss, ee, _) in pruned):
            continue
        pruned = [(pp, ss, ee, tt) for (pp, ss, ee, tt) in pruned
                  if not (pp == p and s <= ss and ee <= e)]
        pruned.append((p, s, e, t))

    uniq = {}
    for p, s, e, t in pruned:
        uniq[(p, s, e)] = t
    return list(uniq.items())

# ===== LLM 호출 =====
def build_messages(payload_text: str):
    base_dir = pathlib.Path(__file__).resolve().parent
    sys_prompt = (base_dir / "prompt.md").read_text(encoding="utf-8")
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": payload_text if payload_text.strip() else "변경 섹션 없음"}
    ]

def call_openai(messages):
    model = MODEL.strip()
    use_responses = model.startswith("o4")  # o4, o4-mini 등만 responses 사용

    if use_responses:
        url = "https://api.openai.com/v1/responses"

        def to_responses_input(ms):
            # 🔴 모든 role(system/user/developer 등)에 대해 input_text 사용
            return [
                {
                    "role": m["role"],
                    "content": [{"type": "input_text", "text": m["content"]}],
                }
                for m in ms
            ]

        payload = {
            "model": model,
            "input": to_responses_input(messages),
            "max_output_tokens": 1200,
        }
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1200,
        }

    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} — body={r.text[:2000]}")

    data = r.json()

    def extract_text_from_responses(d):
        if d.get("output_text"):
            return d["output_text"]
        try:
            for msg in d.get("output", []) or []:
                for part in msg.get("content", []) or []:
                    if part.get("text"):
                        return part["text"]
        except Exception:
            pass
        if d.get("choices"):
            return d["choices"][0]["message"]["content"]
        return json.dumps(d, ensure_ascii=False)

    content = extract_text_from_responses(data) if use_responses \
              else data["choices"][0]["message"]["content"]

    s, e = content.find("{"), content.rfind("}")
    try:
        parsed = json.loads(content[s:e+1])
    except Exception:
        parsed = {"diagnosis": [], "issues": [], "overall_summary": content}
    return parsed, content

# ===== GitHub 포스팅 =====
def post_summary(body: str):
    url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUM}/comments"
    requests.post(url,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
        json={"body": body}
    ).raise_for_status()

SUMMARY_TAG = "<!-- LLM-CODE-REVIEW-SUMMARY -->"

def upsert_summary_comment(body: str):
    """PR에 요약 코멘트를 하나만 유지(업서트). PATCH는 /issues/comments/:id 경로."""
    base = f"https://api.github.com/repos/{REPO}"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}

    list_url = f"{base}/issues/{PR_NUM}/comments"
    r = requests.get(list_url, headers=headers); r.raise_for_status()
    comments = r.json() or []

    target = next((c for c in reversed(comments)
                   if isinstance(c.get("body"), str) and SUMMARY_TAG in c["body"]), None)

    body_with_tag = body + f"\n\n{SUMMARY_TAG}"

    if target:
        edit_url = f"{base}/issues/comments/{target['id']}"
        pr = requests.patch(edit_url, headers=headers, json={"body": body_with_tag})
        if pr.status_code == 404:
            requests.post(list_url, headers=headers, json={"body": body_with_tag}).raise_for_status()
        else:
            pr.raise_for_status()
    else:
        requests.post(list_url, headers=headers, json={"body": body_with_tag}).raise_for_status()

def post_review_summary(body: str):
    upsert_summary_comment(body)

def _overlaps_hunks(path: str, it: dict, hunks_by_file: dict) -> bool:
    """이슈가 diff 범위에 포함되는지 체크(인라인 가능 여부)."""
    hunks = hunks_by_file.get(path, [])
    line = it.get("line")
    sline = it.get("start_line")
    eline = it.get("end_line") or line

    if line:
        line = int(line)
        return any(s <= line <= e for (s, e) in hunks)
    if sline and eline:
        sline = int(sline); eline = int(eline)
        return any(not (eline < s or e < sline) for (s, e) in hunks)
    return False

def post_inline(issues: list, hunks_by_file: dict):
    """변경 라인(=diff 포함)만 인라인 코멘트로 게시."""
    url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUM}/comments"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}

    # 정렬: severity → file → line
    sev_rank = {"critical":0, "major":1, "minor":2}
    issues = sorted(issues, key=lambda it: (
        sev_rank.get((it.get("severity") or "minor").lower(), 9),
        it.get("file",""),
        it.get("line") or it.get("start_line") or 10**9
    ))

    for it in issues[:200]:
        path = it.get("file")
        if not path or not _overlaps_hunks(path, it, hunks_by_file):
            continue

        title = f"**{it.get('type','Issue')} ({it.get('severity','minor')})**"
        reason = (it.get("reason","") or "").strip()

        # suggestion 정규화
        sugg_block = _extract_suggestion_block(it.get("suggestion",""))
        has_range = it.get("start_line") is not None and it.get("end_line") is not None
        has_line  = it.get("line") is not None

        # 다중 라인 제안인데 범위가 없으면 혼선 방지: 코멘트만 남김
        if sugg_block and _is_multiline_sugg(sugg_block) and not has_range:
            body = f"{title}\n{reason}"
        else:
            body = f"{title}\n{reason}\n\n{sugg_block}" if sugg_block else f"{title}\n{reason}"

        payload = {"body": body, "path": path, "side":"RIGHT", "commit_id": HEAD_SHA}
        if has_range:
            payload["start_line"] = int(it["start_line"])
            payload["line"] = int(it.get("end_line", it["start_line"]))
        elif has_line:
            payload["line"] = int(it["line"])
        else:
            continue  # 위치 정보가 없으면 업로드 불가

        requests.post(url, headers=headers, json=payload)
        # 실패해도 조용히 진행

# ===== 요약(Diagnosis만) =====
def build_summary_markdown(diag: list, inline_issues: list, out_of_diff: list) -> str:
    """Diagnosis만 출력."""
    sev_emoji = {"critical":"🛑", "major":"⚠️", "minor":"ℹ️", "info":"📝"}
    all_issues = (inline_issues or []) + (out_of_diff or [])

    # 전체 심각도 배지
    by_sev = defaultdict(int)
    for it in all_issues:
        by_sev[(it.get("severity") or "minor").lower()] += 1
    badge = " ".join(f"{sev_emoji.get(k,'•')} {k.capitalize()}: **{by_sev.get(k,0)}**"
                     for k in ("critical","major","minor","info"))

    rows = []
    for d in (diag or []):
        cat = d.get("type","-")
        summary = d.get("summary","")
        count = d.get("count",0)
        rows.append(f"- **{cat}** — {count}건\n  · {summary}")
    diagnosis_md = "\n".join(rows) if rows else "_요약 없음_"

    return (
        "## 🧭 LLM Code Review Summary\n\n"
        f"- 총 이슈: **{len(all_issues)}**  |  {badge}\n\n"
        "### Diagnosis\n"
        f"{diagnosis_md}"
    )

# ===== 섹션 카드(참고) =====
def extract_plain_code_from_section(section_text: str) -> str:
    lines = section_text.splitlines()
    try:
        start_idx = next(i for i, ln in enumerate(lines) if ln.startswith("<SECTION "))
        end_idx   = len(lines) - 1 - next(i for i, ln in enumerate(reversed(lines)) if ln.strip() == "</SECTION>")
    except StopIteration:
        body = lines
    else:
        body = lines[start_idx+1:end_idx]
    plain = []
    for ln in body:
        m = re.match(r"^\s*\d+:\s?(.*)$", ln)
        plain.append(m.group(1) if m else ln)
    return "\n".join(plain)

def format_section_card_md(path: str, s: int, e: int, section_text: str, parsed: dict) -> str:
    code_block = extract_plain_code_from_section(section_text)
    issues = []
    for it in parsed.get("issues", []) or []:
        if (it.get("file") == path) and (
            (it.get("line") and s <= int(it["line"]) <= e) or
            (it.get("start_line") and it.get("end_line") and
             not (int(it["end_line"]) < s or e < int(it["start_line"])) )
        ):
            issues.append(it)
    if issues:
        issues_md = "\n".join(
            f"- **{it.get('type','Issue')}** ({it.get('severity','minor')}) "
            f"@ L{it.get('start_line', it.get('line'))}"
            f"{'-L'+str(it['end_line']) if it.get('start_line') else ''} — {it.get('reason','')}"
            for it in issues
        )
        first_sugg = next((it.get("suggestion") for it in issues if it.get("suggestion")), "")
    else:
        issues_md = "_No issues detected in this section._"; first_sugg = ""
    md = []
    md.append(f"### `{path}` {s}–{e}\n")
    md.append("```python"); md.append(code_block); md.append("```")
    md.append("\n**Findings**"); md.append(issues_md)
    if first_sugg: md.append("\n**Suggested change**\n" + first_sugg)
    return "\n".join(md)

def per_file_calls(hunks_by_file):
    all_issues = []
    section_cards = []
    for path, hunks in hunks_by_file.items():
        secs = sections_for_file(path, hunks)
        if not secs: continue
        for (_, s, e), section_text in secs:
            parsed, _raw = call_openai(build_messages(section_text))
            section_cards.append(format_section_card_md(path, s, e, section_text, parsed))
            all_issues += parsed.get("issues", [])
    if section_cards:
        try:
            post_review_summary("## 📦 LLM Code Review (by section)\n" + "\n\n---\n\n".join(section_cards))
        except Exception:
            pass
    return [], all_issues

# ===== 메인 =====
def build_payload_all_at_once(hunks_by_file):
    blocks = []
    for path, hunks in hunks_by_file.items():
        for (_, s, e), sec in sections_for_file(path, hunks):
            blocks.append(sec)
    return "\n\n".join(blocks)[:MAX_PAYLOAD_CHARS]

def main():
    diff = get_diff_unified0()
    hunks = parse_hunks(diff)
    if not hunks:
        upsert_summary_comment("## 🧭 LLM Code Review Summary\n변경 섹션이 없어 리뷰를 생략합니다.\n\n" + SUMMARY_TAG)
        return

    # 파일별로 모으기
    hunks_by_file = defaultdict(list)
    for path, st, en in hunks:
        hunks_by_file[path].append((st, en))

    # 섹션 디버그
    if os.getenv("SECTIONS_DEBUG", "0") == "1":
        body = "### 🔎 Section Debug\n"
        for p, h in hunks_by_file.items(): body += f"- {p}: {h}\n"
        post_summary(body); return

    # 호출 방식
    if PER_FILE_CALL:
        _diag_ignored, issues = per_file_calls(hunks_by_file)
    else:
        payload = build_payload_all_at_once(hunks_by_file)
        parsed, _raw = call_openai(build_messages(payload))
        issues = parsed.get("issues", [])

    # 인라인 가능 vs 불가 분리
    inline_candidates, out_of_diff = [], []
    for it in issues:
        path = it.get("file")
        (inline_candidates if path and _overlaps_hunks(path, it, hunks_by_file) else out_of_diff).append(it)

    # 인라인 업로드
    post_inline(inline_candidates, hunks_by_file)

    # 화면에 보이는 전체 이슈 기준으로 집계
    all_issues = inline_candidates + out_of_diff
    aggregated_diag = build_aggregated_diagnosis(all_issues)

    # 상단 요약 업서트
    summary_md = build_summary_markdown(aggregated_diag, inline_candidates, out_of_diff)
    upsert_summary_comment(summary_md)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            post_summary(f"리뷰 작업 중 예외 발생: `{e}`")
        finally:
            raise

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_review.py
- PR diff를 파일/함수 경계로 쪼개 LLM에 보내고, 결과를 요약 + 인라인 코멘트로 게시
- 가시성 개선:
  1) PR 상단 요약 코멘트는 '업서트'로 단 한 개만 유지 (배지/표/체크리스트)
  2) 인라인 코멘트는 모두 게시하되 Review API로 한 번에 제출(알림 1회)
  3) diff 밖 라인은 인라인 대신 요약에 'Out-of-diff findings'로 모아 안내
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

# 섹션/맥락 파라미터
MAX_LINES_PER_SECTION = int(os.getenv("MAX_LINES_PER_SECTION", "220"))
OVERLAP_LINES         = int(os.getenv("OVERLAP_LINES", "10"))
FUNC_CTX_BEFORE       = int(os.getenv("FUNC_CTX_BEFORE", "16"))
NUM_CTX_LINES         = int(os.getenv("NUM_CTX_LINES", "6"))
MAX_PAYLOAD_CHARS     = int(os.getenv("MAX_PAYLOAD_CHARS", "180000"))
PER_FILE_CALL         = os.getenv("PER_FILE_CALL", "true").lower() == "true"

# ===== 유틸 =====
def sh(*cmd: str) -> str:
    return subprocess.check_output(list(cmd), text=True).strip()

def get_diff_unified0() -> str:
    base = sh("git", "merge-base", f"origin/{BASE_BRANCH}", "HEAD")
    return subprocess.check_output(["git","diff",f"{base}...HEAD","--unified=0"], text=True)

def parse_hunks(diff: str):
    """@@ -a,b +c,d @@ 블록을 파싱해 (path, start, end) 리스트 생성"""
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

# ===== 분할/확장 로직 =====
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
    - 함수 시작 위로 FUNC_CTX_BEFORE 줄 확장 (선언/임포트 맥락)
    - 클래스 내부면 클래스 시작 이전으로 확장 금지
    - 이전 함수의 끝 이전으로 확장 금지
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
    """라인 번호 포함 섹션(컨텍스트 ±ctx 포함)"""
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

# ===== LLM 호출/리포팅 =====
def build_messages(payload_text: str):
    base_dir = pathlib.Path(__file__).resolve().parent
    sys_prompt = (base_dir / "prompt.md").read_text(encoding="utf-8")
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": payload_text if payload_text.strip() else "변경 섹션 없음"}
    ]

def call_openai(messages):
    model = MODEL
    use_responses = model.startswith("o4")

    if use_responses:
        url = "https://api.openai.com/v1/responses"
        def to_responses_input(ms):
            return [
                {"role": m["role"],
                 "content": [{"type": "input_text", "text": m["content"]}]}
                for m in ms
            ]
        payload = {"model": model,
                   "input": to_responses_input(messages),
                   "max_output_tokens": 1200}
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"model": model, "messages": messages,
                   "temperature": 0.2, "max_tokens": 1200}

    r = requests.post(url,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type":"application/json"},
        data=json.dumps(payload), timeout=180)

    if r.status_code != 200:
        try: err = r.json()
        except Exception: err = {"text": r.text}
        print("OpenAI error:", r.status_code, json.dumps(err, ensure_ascii=False)[:4000])
        r.raise_for_status()

    data = r.json()

    def extract_text_from_responses(d):
        if d.get("output_text"): return d["output_text"]
        try:
            for msg in d.get("output", []) or []:
                for part in msg.get("content", []) or []:
                    if part.get("text"): return part["text"]
        except Exception: pass
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

def post_summary(body: str):
    url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUM}/comments"
    requests.post(url,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                 "Accept":"application/vnd.github+json"},
        json={"body": body}
    ).raise_for_status()

# === Summary Upsert ===
SUMMARY_TAG = "<!-- LLM-CODE-REVIEW-SUMMARY -->"

def upsert_summary_comment(body: str):
    """기존 요약(있으면 PATCH, 없으면 POST) — 요약 코멘트를 한 개만 유지"""
    list_url = f"https://api.github.com/repos/{REPO}/issues/{PR_NUM}/comments"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
               "Accept":"application/vnd.github+json"}

    r = requests.get(list_url, headers=headers)
    r.raise_for_status()
    comments = r.json() or []

    target = next((c for c in reversed(comments)
                   if isinstance(c.get("body"), str)
                   and SUMMARY_TAG in c["body"]), None)

    body_with_tag = body + f"\n\n{SUMMARY_TAG}"
    if target:
        edit_url = f"{list_url}/{target['id']}"
        requests.patch(edit_url, headers=headers, json={"body": body_with_tag}).raise_for_status()
    else:
        requests.post(list_url, headers=headers, json={"body": body_with_tag}).raise_for_status()

def post_review_summary(body: str):
    upsert_summary_comment(body)

# --- diff 포함 여부 체크 ---
def _overlaps_hunks(path: str, it: dict, hunks_by_file: dict) -> bool:
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

# === 요약 마크다운 (배지/표/체크리스트 + Out-of-diff 목록) ===
def build_summary_markdown(diag: list, issues: list, out_of_diff: list) -> str:
    sev_order = {"critical":0, "major":1, "minor":2, "info":3}
    sev_emoji = {"critical":"🛑", "major":"⚠️", "minor":"ℹ️", "info":"📝"}

    total = len(issues) + len(out_of_diff)
    by_sev = defaultdict(int)
    by_file = defaultdict(list)
    for it in issues + out_of_diff:
        s = (it.get("severity") or "minor").lower()
        by_sev[s] += 1
        by_file[it.get("file","?")].append(it)

    badge = " ".join(
        f"{sev_emoji.get(k,'•')} {k.capitalize()}: **{by_sev.get(k,0)}**"
        for k in ("critical","major","minor","info")
    )

    rows = []
    for f, lst in sorted(by_file.items()):
        row = {"file": f, "critical":0,"major":0,"minor":0,"info":0}
        for it in lst:
            row[(it.get("severity") or "minor").lower()] += 1
        rows.append(row)
    table = ["| File | Critical | Major | Minor | Info |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| `{r['file']}` | {r['critical']} | {r['major']} | {r['minor']} | {r['info']} |")

    def issue_label(it):
        s = (it.get("severity") or "minor").lower()
        em = sev_emoji.get(s,"•")
        loc = f"L{it.get('start_line', it.get('line','?'))}"
        return f"- [ ] {em} **{it.get('type','Issue')}** — `{it.get('file','?')}` {loc} — {it.get('reason','')}"

    issues_sorted = sorted(issues, key=lambda it:(sev_order.get((it.get("severity") or "minor").lower(), 9), it.get("file",""), it.get("line") or it.get("start_line") or 10**9))
    checklist = "\n".join(issue_label(it) for it in issues_sorted[:80])

    out_list = "\n".join(issue_label(it) for it in out_of_diff[:80]) or "_없음_"

    def summarize_diag(diag: list):
        if not diag: return "_요약 없음_"
        return "\n".join(f"- **{d.get('type','-')}**: {d.get('count',0)}건 — {d.get('summary','')}" for d in diag)

    md = []
    md.append("## 🤖 LLM Code Review 요약")
    md.append("")
    md.append(f"- 총 이슈: **{total}**  |  {badge}")
    md.append("")
    md += table
    md.append("")
    md.append("### 주요 이슈 체크리스트")
    md.append(checklist if checklist else "_표시할 이슈 없음_")
    md.append("")
    md.append("### Out-of-diff findings (인라인 불가)")
    md.append(out_list)
    md.append("")
    md.append("### 분석 메모")
    md.append(summarize_diag(diag))
    return "\n".join(md)

# === 인라인: 리뷰로 한 번에 제출 ===
def post_inline_as_review(issues: list):
    """GitHub Review API로 코멘트를 한 번에 제출 (알림 1회)"""
    if not issues:
        return
    url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUM}/reviews"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
               "Accept":"application/vnd.github+json"}

    comments = []
    for it in issues[:200]:
        path = it.get("file"); line = it.get("line"); sline = it.get("start_line"); eline = it.get("end_line") or line
        if not path or (not line and not sline): 
            continue
        title = f"**{it.get('type','Issue')} ({it.get('severity','minor')})**"
        body  = f"{title}\n{it.get('reason','')}\n\n{it.get('suggestion','')}"
        entry = {"path": path, "side":"RIGHT", "body": body}
        if sline:
            entry["start_line"] = int(sline); entry["line"] = int(eline)
        else:
            entry["line"] = int(line)
        comments.append(entry)

    payload = {"event":"COMMENT", "comments": comments}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code not in (200,201):
        # 실패하면 개별 코멘트로 폴백
        url_c = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUM}/comments"
        for c in comments:
            c["commit_id"] = HEAD_SHA
            requests.post(url_c, headers=headers, json=c)

# ===== 섹션 카드 =====
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
             not (int(it["end_line"]) < s or e < int(it["start_line"])))
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
        issues_md = "_No issues detected in this section._"
        first_sugg = ""

    md = []
    md.append(f"### `{path}` {s}–{e}")
    md.append("")
    md.append("```python")
    md.append(code_block)
    md.append("```")
    md.append("")
    md.append("**Findings**")
    md.append(issues_md)
    if first_sugg:
        md.append("\n**Suggested change**\n" + first_sugg)
    return "\n".join(md)

def per_file_calls(hunks_by_file):
    all_issues, all_diag = [], []
    section_cards = []

    for path, hunks in hunks_by_file.items():
        secs = sections_for_file(path, hunks)
        if not secs:
            continue
        for (_, s, e), section_text in secs:
            parsed, _raw = call_openai(build_messages(section_text))
            section_cards.append(
                format_section_card_md(path, s, e, section_text, parsed)
            )
            all_diag  += parsed.get("diagnosis", [])
            all_issues += parsed.get("issues", [])

    if section_cards:
        try:
            post_review_summary(
                "## 🧩 LLM Code Review (by section)\n"
                + "\n\n---\n\n".join(section_cards)
            )
        except Exception:
            pass

    return all_diag, all_issues

# ===== 메인 =====
def main():
    diff = get_diff_unified0()
    hunks = parse_hunks(diff)
    if not hunks:
        upsert_summary_comment("## 🤖 LLM Code Review 요약\n변경 섹션이 없어 리뷰를 생략합니다.\n\n" + SUMMARY_TAG)
        return

    # 파일별로 모으기
    hunks_by_file = defaultdict(list)
    for path, st, en in hunks:
        hunks_by_file[path].append((st, en))

    if PER_FILE_CALL:
        diag, issues = per_file_calls(hunks_by_file)
    else:
        payload = build_payload_all_at_once(hunks_by_file)
        parsed, _raw = call_openai(build_messages(payload))
        diag, issues = parsed.get("diagnosis", []), parsed.get("issues", [])

    # diff 포함/미포함 분리
    inline_candidates, out_of_diff = [], []
    for it in issues:
        path = it.get("file")
        if path and _overlaps_hunks(path, it, hunks_by_file):
            inline_candidates.append(it)
        else:
            out_of_diff.append(it)

    # 인라인: 리뷰 1회 제출
    post_inline_as_review(inline_candidates)

    # 요약 업서트
    summary_md = build_summary_markdown(diag, inline_candidates, out_of_diff)
    upsert_summary_comment(summary_md)

def build_payload_all_at_once(hunks_by_file):
    blocks = []
    for path, hunks in hunks_by_file.items():
        for (_, s, e), sec in sections_for_file(path, hunks):
            blocks.append(sec)
    return "\n\n".join(blocks)[:MAX_PAYLOAD_CHARS]

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            post_summary(f"리뷰 작업 중 예외 발생: `{e}`")
        finally:
            raise

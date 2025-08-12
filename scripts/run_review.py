#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_review.py
- PR diff의 섹션을 "함수/메서드 경계" 기준으로 구성하고,
  너무 크면 오버랩 분할하되, 선언부(상단 변수/클래스 필드/임포트)까지 포함해
  LLM이 중간에서 끊기지 않도록 맥락을 제공합니다.
- OpenAI 호출 → JSON 응답 → PR 요약/인라인 코멘트 업로드.
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
def summarize_sections(hunks_by_file):
    out = []
    for path, hunks in hunks_by_file.items():
        secs = sections_for_file(path, hunks)
        if not secs:
            continue
        out.append(f"## {path}")
        for (p, s, e), text in secs:
            lines = text.splitlines()
            preview_begin = lines[0] if lines else ""
            preview_end   = lines[-1] if lines else ""
            out.append(f"- section: {s}~{e}  (len={len(text)})")
            out.append(f"  - begin: `{preview_begin[:120]}`")
            out.append(f"  - end  : `{preview_end[:120]}`")
    return "\n".join(out) if out else "섹션 없음(변경이 없거나 ctags 미설치)"

def debug_sections_and_exit(hunks_by_file):
    body = "### 🔎 Section Debug\n" + summarize_sections(hunks_by_file)
    post_summary(body)

def sh(*cmd: str) -> str:
    return subprocess.check_output(list(cmd), text=True).strip()

def get_diff_unified0() -> str:
    base = sh("git", "merge-base", f"origin/{BASE_BRANCH}", "HEAD")
    return subprocess.check_output(["git","diff",f"{base}...HEAD","--unified=0"], text=True)

def parse_hunks(diff: str):
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

# --- helper: hunk 포함 여부 체크 ---
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

def expand_range_for_decls(path: str, func_start: int, func_end: int, total_lines: int, func_ranges: list):
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

def post_review_summary(body: str):
    # 섹션 카드 묶음을 PR 코멘트로 게시
    post_summary(body)

def post_inline(issues: list, hunks_by_file: dict):
    url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUM}/comments"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
               "Accept":"application/vnd.github+json"}

    posted = 0
    skipped = []
    not_inline = []

    for it in issues[:60]:
        path = it.get("file")
        if not path:
            skipped.append({"reason":"missing path", "item":it})
            continue

        if not _overlaps_hunks(path, it, hunks_by_file):
            not_inline.append(it)
            continue

        line = it.get("line")
        sline = it.get("start_line")
        eline = it.get("end_line", line)

        if not line and not sline:
            skipped.append({"reason":"missing line", "item":it})
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

        r = requests.post(url, headers=headers, json=payload)
        if r.status_code == 201:
            posted += 1
        else:
            try: err = r.json()
            except Exception: err = {"text": r.text}
            skipped.append({"reason":"github api", "status": r.status_code, "error": err, "payload": payload})

    if not_inline:
        try:
            post_summary(
                "#### Out-of-diff findings (can’t inline)\n"
                + "\n".join(
                    f"- `{it.get('file')}` L{it.get('start_line', it.get('line'))}"
                    f"{('-L'+str(it['end_line'])) if it.get('start_line') else ''} — "
                    f"**{it.get('type','Issue')}** ({it.get('severity','minor')}): {it.get('reason','')}"
                    for it in not_inline
                )
            )
        except Exception:
            pass

    try:
        post_summary(
            "#### Inline post result\n"
            f"- posted: {posted}\n"
            f"- skipped: {len(skipped)}\n"
            + (("\n```json\n" + json.dumps(skipped, ensure_ascii=False, indent=2)[:5500] + "\n```") if skipped else "")
        )
    except Exception:
        pass

def summarize_diag(diag: list):
    if not diag:
        return "발견된 요약 없음"
    out = []
    for d in diag:
        t = d.get("type","-"); c = d.get("count",0); s = d.get("summary","")
        out.append(f"- **{t}**: {c}건 — {s}")
    return "\n".join(out)

# ===== 페이로드 조립 =====
def build_payload_all_at_once(hunks_by_file):
    blocks = []
    for path, hunks in hunks_by_file.items():
        for (_, s, e), sec in sections_for_file(path, hunks):
            blocks.append(sec)
    return "\n\n".join(blocks)[:MAX_PAYLOAD_CHARS]

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
                "## 🤖 LLM Code Review (by section)\n"
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
        post_summary("변경 섹션이 없어 리뷰를 생략합니다.")
        return

    # 파일별로 모으기
    hunks_by_file = defaultdict(list)
    for path, st, en in hunks:
        hunks_by_file[path].append((st, en))

    # 섹션 디버그 모드
    if os.getenv("SECTIONS_DEBUG", "0") == "1":
        debug_sections_and_exit(hunks_by_file)
        return

    if PER_FILE_CALL:
        diag, issues = per_file_calls(hunks_by_file)
        post_summary("### 🤖 LLM Code Review 요약\n" + summarize_diag(diag))
        post_inline(issues, hunks_by_file)  # diff 포함 라인만 인라인
    else:
        payload = build_payload_all_at_once(hunks_by_file)
        parsed, _raw = call_openai(build_messages(payload))
        post_summary(
            "### 🤖 LLM Code Review 요약\n"
            + summarize_diag(parsed.get("diagnosis", []))
            + "\n\n---\n"
            + parsed.get("overall_summary","")
        )
        post_inline(parsed.get("issues", []), hunks_by_file)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            post_summary(f"리뷰 작업 중 예외 발생: `{e}`")
        finally:
            raise

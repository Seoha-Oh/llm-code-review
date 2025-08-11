import os, json, subprocess, requests, pathlib

REPO = os.getenv("GITHUB_REPOSITORY")
EVENT = json.load(open(os.getenv("GITHUB_EVENT_PATH")))
PR = EVENT["pull_request"]["number"]
BASE = os.getenv("BASE_BRANCH","main")
MODEL = os.getenv("OPENAI_MODEL","o4-mini")

def sh(*cmd): return subprocess.check_output(list(cmd), text=True).strip()

def get_diff():
    base = sh("git","merge-base",f"origin/{BASE}","HEAD")
    return subprocess.check_output(["git","diff",f"{base}...HEAD","--unified=0"], text=True)

def call_openai(messages):
    r = requests.post("https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {os.environ['OPENAI_API_KEY']}","Content-Type":"application/json"},
        data=json.dumps({"model":MODEL,"messages":messages,"temperature":0.2,"max_tokens":1200}), timeout=180)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    j = json.loads(content[content.find("{"):content.rfind("}")+1])
    return j

def comment(body):
    url = f"https://api.github.com/repos/{REPO}/issues/{PR}/comments"
    requests.post(url, headers={"Authorization":f"Bearer {os.environ['GITHUB_TOKEN']}",
                                "Accept":"application/vnd.github+json"}, json={"body":body}).raise_for_status()

def inline(issues):
    url = f"https://api.github.com/repos/{REPO}/pulls/{PR}/comments"
    sha = EVENT["pull_request"]["head"]["sha"]
    for it in issues[:30]:
        if not it.get("file") or not it.get("line"): continue
        body = f"**{it.get('type','Issue')} ({it.get('severity','minor')})**\n{it.get('reason','')}\n\n```suggestion\n{it.get('suggestion','')}\n```"
        requests.post(url, headers={"Authorization":f"Bearer {os.environ['GITHUB_TOKEN']}",
                                    "Accept":"application/vnd.github+json"},
                      json={"body":body,"path":it["file"],"line":it["line"],"side":"RIGHT","commit_id":sha})

def main():
    diff = get_diff()
    if not diff.strip(): return comment("변경 사항이 없어 리뷰를 생략합니다.")
    sys_prompt = pathlib.Path("scripts/prompt.md").read_text()
    result = call_openai([{"role":"system","content":sys_prompt},
                          {"role":"user","content":f"[DIFF]\n{diff[:180000]}"}])
    diag = "\n".join([f"- {d['type']}: {d['count']}건 — {d['summary']}" for d in result.get("diagnosis",[])])
    comment(f"### LLM Code Review 요약\n{diag}\n\n---\n{result.get('overall_summary','')}")
    inline(result.get("issues",[]))

if __name__ == "__main__":
    main()

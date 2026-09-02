"""
ドキュメント階層の自動検査（読み取り専用）

CLAUDE.md / CONVENTIONS.md / PLAYBOOK.md / .claude/skills の 4 層構造が
SSoT 原則を守れているかを機械的に検証する。

設計思想:
    行数や見出しの diff では「教訓の実測数値が静かに消えた」ことを検知できない。
    数値の grep なら 100% 検知できる（C4 がこのチェッカーの主役）。

使い方:
    uv run python -m scripts.harness.doc_audit                    # 検査
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート

# ── 4 層の定義 ──────────────────────────────────────────────
ALWAYS_LOADED = ["CLAUDE.md"] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / ".claude" / "rules").glob("*.md")
)
# C1: 常時ロードの**文字数**上限。
# 行数で測っていた頃は「2 つの箇条書きを 1 行に結合する」だけで測定値が下がり、
# 中身を減らさずに上限を通過できた（実際、ある改善作業では行数 -2 に対して
# 文字数 +1,898 で通過していた）。コンテキストの費用は改行ではなく中身の量なので、
# 文字数で測る。**上限は分割のたびに締め直す** —— 余白が広いと強制力が失われ、
# 「まだ入る」で少しずつ膨らむ。実測 + 数%の余白を上限にし、超えたら L1/L2/L3 へ出す。
#
# 出典: Claude Code 自身が CLAUDE.md に対して **15,000 字**で警告を出す
# （"Large CLAUDE.md file detected"）。テンプレートはさらに厳しく、
# 憲法として必要な最小限（規約・コマンド・指針の索引）だけを置く。
# **上限と下限の両方**を持つ。上限だけだと膨張しか防げず、削りすぎて憲法が
# 空洞化する方向を検知できない。行数も併せて検査する —— ユーザー指定の「60 行以内」が
# どのガードにも入っていなかったため、憲法化後に 84 行までじわじわ増えていた。
ALWAYS_LOADED_BUDGET = 5_000        # 文字数の上限
ALWAYS_LOADED_MIN = 3_000           # 文字数の下限（削りすぎの検知）
ALWAYS_LOADED_MAX_LINES = 60        # 行数の上限（走査性を保つ）

DOC_FILES = ["CLAUDE.md", "GUIDELINES.md", "CONVENTIONS.md", "PLAYBOOK.md"]
SKILL_GLOB = ".claude/skills/*/SKILL.md"
AGENT_GLOB = ".claude/agents/*.md"

# ── C4: 失ってはいけない実測値（Phase 0 で凍結）────────────────
# 教訓の payload。これが消えると規範が「守るべきもの」でなくなる。
CRITICAL_NUMBERS = [
    # 天井帯の事後検証
    "0.95003", "0.95048", "0.95060", "0.95045", "0.95043", "0.95034",
    "0.95013", "0.95050", "0.95101",
    # 校正不足の発見（テンプレート最大の跳躍）
    "0.46598", "0.44485", "0.43832", "0.81189", "0.81185",
    # 情報天井（12 モデルの収束帯）
    "0.97169", "0.97227", "0.88603", "0.88970",
    # スコア格子
    "0.0000289", "11,941", "0.95084", "0.95081", "0.95047",
    # E[max] / 相関 / 回帰
    "0.000021", "0.000052", "0.998", "0.853", "0.819",
    # 件数・比率
    "339", "169", "496", "35%", "28%", "21%", "74%",
    # HP 探索の発散（L-18）
    "0.633", "0.896",
]

# ── C6: キーフレーズ → SSoT ファイル ──────────────────────────
SSOT_MAP = {
    'matplotlib.use("Agg")': "CONVENTIONS.md",
    "sub_{exp_id}_{model}": "CONVENTIONS.md",
    "feat(expNNN)": "CONVENTIONS.md",
    "ExperimentTracker(": "CONVENTIONS.md",
    "本日 X/5 回目の提出": ".claude/skills/ds-kaggle-submit/SKILL.md",
    "Public LB Top-10 ∪ OOF Top-10": "GUIDELINES.md",
}

# ── C7: コンペ識別子を書いてよいのは PLAYBOOK の教訓アーカイブのみ ──
COMPETITION_TOKENS = re.compile(r"s6e[0-9]|playground-series")
COMPETITION_ALLOWED = ["PLAYBOOK.md", "docs/TODO_TEMPLATE.md", "README.md"]


def _iter_docs():
    """検査対象の md ファイル（相対パス, 本文）を列挙する。"""
    for rel in DOC_FILES:
        p = ROOT / rel
        if p.exists():
            yield rel, p.read_text()
    for p in sorted(ROOT.glob(SKILL_GLOB)):
        yield str(p.relative_to(ROOT)), p.read_text()
    for p in sorted(ROOT.glob(AGENT_GLOB)):
        yield str(p.relative_to(ROOT)), p.read_text()
    for p in sorted((ROOT / ".claude" / "rules").glob("*.md")):
        yield str(p.relative_to(ROOT)), p.read_text()


def _slug(heading: str) -> str:
    """GitHub 式のアンカー slug に変換する。"""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s぀-ヿ一-鿿-]", "", s)
    return re.sub(r"\s+", "-", s).strip("-")


def _headings(text: str) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M)}


# 各チェックが「何件を実際に検査したか」。**0 なら守っているつもりで何も見ていない。**
# 移設や書き換えで検査対象が消えても「✅ 0 件」と表示されるため、3 度見逃した
# （README の自己申告値・G-* の ID 定義・文書中のコマンド）。分母を明示して機械で捕まえる。
CHECKED: dict[str, int] = {}

# チェックの総数。C11 が README の申告値と突き合わせる。
# 以前は `len(results) + 1` で数えており、**自分より後ろに追加された検査を数え落とした**。
TOTAL_CHECKS = 15


def check(results: list[tuple[str, str, str]]) -> None:
    docs = dict(_iter_docs())
    CHECKED.clear()

    # ── C1: 常時ロードの行数予算 ──
    present = [f for f in ALWAYS_LOADED if (ROOT / f).exists()]
    total = sum(len((ROOT / f).read_text()) for f in present)
    lines = sum(len((ROOT / f).read_text().splitlines()) for f in present)
    detail = ", ".join(f"{f}={len((ROOT/f).read_text()):,}字" for f in present)
    problems = []
    if total > ALWAYS_LOADED_BUDGET:
        problems.append(f"文字数超過（{total:,} > {ALWAYS_LOADED_BUDGET:,}）")
    if total < ALWAYS_LOADED_MIN:
        problems.append(f"文字数が下限未満（{total:,} < {ALWAYS_LOADED_MIN:,}）—— 削りすぎ")
    if lines > ALWAYS_LOADED_MAX_LINES:
        problems.append(f"行数超過（{lines} > {ALWAYS_LOADED_MAX_LINES}）")
    results.append(("ERROR" if problems else "OK", "C1 常時ロードの予算",
                    f"{total:,} 字（{ALWAYS_LOADED_MIN:,}〜{ALWAYS_LOADED_BUDGET:,}）/ "
                    f"{lines} 行（≤{ALWAYS_LOADED_MAX_LINES}）"
                    + ("\n      " + "\n      ".join(problems) if problems else "")))
    # ── C2: アンカー解決 ──
    unresolved = []
    for rel, text in docs.items():
        for m in re.finditer(r"(CLAUDE|PLAYBOOK|CONVENTIONS)\.md#([^\s)`」、。]+)", text):
            target, anchor = f"{m.group(1)}.md", m.group(2)
            if target not in docs:
                unresolved.append(f"{rel}: {target} が存在しない")
            elif anchor.startswith("<") or "…" in anchor:
                continue                                    # プレースホルダーは対象外
            elif _slug(anchor) not in _headings(docs[target]):
                unresolved.append(f"{rel}: {target}#{anchor}")
    CHECKED["C2"] = sum(len(re.findall(r"(?:CLAUDE|PLAYBOOK|CONVENTIONS)\.md#", t)) for t in docs.values())
    results.append(("ERROR" if unresolved else "OK", "C2 アンカー解決",
                    f"未解決 {len(unresolved)} 件" + ("\n      " + "\n      ".join(unresolved[:8]) if unresolved else "")))

    # ── C3: 恒久 ID の解決 / 旧番号の残存 ──
    # 指針の本文（= ID の定義）は GUIDELINES.md が SSoT。CLAUDE.md には索引だけを置く。
    claude = docs.get("CLAUDE.md", "")
    guidelines = docs.get("GUIDELINES.md", "")
    defined = {m.group(1) for m in re.finditer(r"^#{1,6}.*?\b(G-[A-Z][A-Z-]+)", guidelines, re.M)}
    used, undefined = set(), []
    for rel, text in docs.items():
        for m in re.finditer(r"\b(G-[A-Z][A-Z-]+)\b", text):
            used.add(m.group(1))
            if defined and m.group(1) not in defined:
                undefined.append(f"{rel}: {m.group(1)}")
    old_refs = [f"{rel}:{text[:m.start()].count(chr(10))+1}"
                for rel, text in docs.items()
                for m in re.finditer(r"指針\s*#[0-9]", text)]
    msg = f"定義 {len(defined)} / 使用 {len(set(used))}、未定義参照 {len(set(undefined))} 件、旧番号 `指針#N` {len(old_refs)} 件"
    CHECKED["C3"] = len(defined)
    results.append(("ERROR" if (undefined or (defined and old_refs)) else "OK", "C3 ID 解決", msg))

    # ── C4: 実測値の保存（主役）──
    corpus = "\n".join(docs.values())
    missing = [n for n in CRITICAL_NUMBERS if n not in corpus]
    results.append(("ERROR" if missing else "OK", "C4 実測値の保存",
                    f"{len(CRITICAL_NUMBERS)-len(missing)}/{len(CRITICAL_NUMBERS)} 保存"
                    + (f"、消失: {missing}" if missing else "")))

    # ── C5: 重複ブロック検知（正規化後 3 行以上の一致）──
    def norm_lines(text):
        out = []
        for ln in text.splitlines():
            s = re.sub(r"[\s\-*>|#`]+", "", ln)
            out.append(s if len(s) >= 12 else "")           # 短い行はノイズなので無視
        return out

    sigs = {}
    for rel, text in docs.items():
        lines = norm_lines(text)
        for i in range(len(lines) - 2):
            blk = lines[i:i + 3]
            if all(blk):
                sigs.setdefault(hashlib.md5("".join(blk).encode()).hexdigest(), []).append((rel, i + 1))
    dups = [v for v in sigs.values() if len({f for f, _ in v}) > 1]
    results.append(("WARNING" if dups else "OK", "C5 重複ブロック",
                    f"3 行以上の重複 {len(dups)} 箇所"
                    + ("\n      " + "\n      ".join(f"{v[0][0]}:{v[0][1]} ↔ {v[1][0]}:{v[1][1]}" for v in dups[:8]) if dups else "")))

    # ── C6: SSoT 違反 ──
    viol = [f"{rel} に「{phrase[:28]}」（SSoT は {owner}）"
            for phrase, owner in SSOT_MAP.items()
            for rel, text in docs.items()
            if phrase in text and rel != owner]
    CHECKED["C6"] = sum(1 for phrase, owner in SSOT_MAP.items() if phrase in docs.get(owner, ""))
    results.append(("ERROR" if viol else "OK", "C6 SSoT 違反",
                    f"{len(viol)} 件" + ("\n      " + "\n      ".join(viol[:8]) if viol else "")))

    # ── C7: コンペ識別子の混入 ──
    # 本文だけでなく**追跡ファイルのパス名**も検査する。
    # 過去に kaggle_nb/ の Notebook と予測 .npy が 41 ファイル・106MB 混入していたが、
    # .md しか見ていなかったため素通りした（2026-09-02 の精査で発覚）。
    tracked_leaks = []
    try:
        import subprocess
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, timeout=15).stdout
        for path in out.splitlines():
            if path in COMPETITION_ALLOWED or path.startswith("docs/"):
                continue
            if COMPETITION_TOKENS.search(path):
                tracked_leaks.append(f"追跡ファイル名: {path}")
            # 予測・提出物がテンプレート本体に入っていないか
            if path.endswith((".npy", ".pkl")) or "/submissions/" in path:
                tracked_leaks.append(f"成果物が追跡されている: {path}")
    except Exception:
        pass

    leaks = [f"{rel} ({len(COMPETITION_TOKENS.findall(text))} 件)"
             for rel, text in docs.items()
             if rel not in COMPETITION_ALLOWED and COMPETITION_TOKENS.search(text)]
    all_leaks = leaks + tracked_leaks[:8]
    results.append(("ERROR" if all_leaks else "OK", "C7 コンペ識別子",
                    f"本文 {len(leaks)} ファイル / 追跡パス {len(tracked_leaks)} 件"
                    + ("\n      " + "\n      ".join(all_leaks) if all_leaks else "")))

    # ── C8: SESSION.md 上限値の同期 ──
    caps = {}
    for rel, text in docs.items():
        for m in re.finditer(r"(\d+)\s*行を超え|(\d+)\s*行以内|最大\s*(\d+)\s*件|直近\s*(\d+)\s*件", text):
            caps.setdefault(next(g for g in m.groups() if g), set()).add(rel)
    inconsistent = {k: sorted(v) for k, v in caps.items() if len(v) > 1}
    results.append(("WARNING" if len(caps) > 3 else "OK", "C8 上限値の同期",
                    f"検出された上限値 {sorted(caps)}"
                    + (f"、複数ファイルに跨る値: {inconsistent}" if inconsistent else "")))

    # ── C9: CLAUDE.md 内のコードフェンス ──
    fences = claude.count("```") // 2
    results.append(("WARNING" if fences else "OK", "C9 コードフェンス",
                    f"CLAUDE.md 内に {fences} ブロック（L0 は 0 が目標）"))

    # ── C10: 孤立節の検知（オンデマンド層への導線が存在するか）──
    # L1/L2 は自動ロードされない。どこからも参照されない節は「誰も読まない場所」であり、
    # 移設したのに導線を張り忘れた状態を意味する（v6 で実際に起きた実装漏れ）。
    ORPHAN_EXEMPT = {"目次"}   # 自ファイル内ナビゲーション。参照されなくて正常
    orphans = []
    for owner in ("CONVENTIONS.md", "PLAYBOOK.md"):
        if owner not in docs:
            continue
        others = "\n".join(t for rel, t in docs.items() if rel != owner)
        for m in re.finditer(r"^##\s+(.+)$", docs[owner], re.M):
            title, anchor = m.group(1).strip(), _slug(m.group(1))
            if title in ORPHAN_EXEMPT:
                continue
            if f"{owner}#{anchor}" in others.replace(" ", ""):
                continue
            # アンカー無しの素の言及（「CONVENTIONS.md の ExperimentTracker」等）も導線として認める
            key = re.sub(r"[（(].*?[）)]", "", title).strip()
            if key and key in others:
                continue
            orphans.append(f"{owner}#{title}")
    results.append(("WARNING" if orphans else "OK", "C10 孤立節",
                    f"どこからも参照されない節 {len(orphans)} 件"
                    + ("\n      " + "\n      ".join(orphans) if orphans else "")))

    # ── C12: 指針の索引と本文の一致 ──
    # L0 は索引だけを持ち、本文は GUIDELINES.md にある。両者がずれると
    # 「索引に載っているのに本文が無い（読めない指針）」「本文はあるのに索引に無い
    # （存在を知られない指針）」が起きる。分離した構造ではこれが最も壊れやすい。
    idx_section = re.search(r"## 判断指針の索引.*?(?=\n## |\Z)", claude, re.S)
    idx_ids = set(re.findall(r"`(G-[A-Z][A-Z-]+)`", idx_section.group(0))) if idx_section else set()
    body_ids = defined
    missing_body = sorted(idx_ids - body_ids)     # 索引にあるが本文が無い
    missing_idx = sorted(body_ids - idx_ids)      # 本文はあるが索引に無い
    detail = []
    if missing_body:
        detail.append(f"索引にあるが GUIDELINES.md に本文が無い: {', '.join(missing_body)}")
    if missing_idx:
        detail.append(f"本文はあるが CLAUDE.md の索引に無い: {', '.join(missing_idx)}")
    results.append(("ERROR" if detail else "OK", "C12 指針の索引と本文",
                    f"索引 {len(idx_ids)} / 本文 {len(body_ids)}"
                    + ("\n      " + "\n      ".join(detail) if detail else "")))

    # ── C13: エージェント定義の妥当性 ──
    # サブエージェントは `tools` を絞ることで「学習実行・commit・提出をさせない」ことを
    # **機械的に**保証している（指示ではなく道具で縛る）。ここが緩むと保証が消えるので、
    # 読み取り専用であるべきエージェントに Bash が渡っていないかを検査する。
    READONLY_AGENTS = {"fe-ideator", "experiment-reviewer"}
    # 期待するエージェント一覧の定義元。**追加したらここにも足す** —— そうしないと
    # 「消えても気づかない」状態に戻る（LOW-8 の再発防止）。
    AGENT_NAMES_EXPECTED = {"fe-ideator", "experiment-reviewer",
                            "blocker-investigator", "kaggle-researcher"}
    ALLOWED_TOOLS = {"Read", "Grep", "Glob", "Bash", "WebFetch", "WebSearch"}
    agent_issues = []
    agent_files = sorted(ROOT.glob(AGENT_GLOB))
    for ap in agent_files:
        text = ap.read_text()
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if not m:
            agent_issues.append(f"{ap.name}: frontmatter が無い")
            continue
        fm = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()
        if fm.get("name") != ap.stem:
            agent_issues.append(f"{ap.name}: name({fm.get('name')}) がファイル名と不一致")
        if not fm.get("description"):
            agent_issues.append(f"{ap.name}: description が無い（常時ロードされる要素）")
        tools = {t.strip() for t in fm.get("tools", "").split(",") if t.strip()}
        if not tools:
            agent_issues.append(f"{ap.name}: tools が無い（無制限になる）")
        if tools - ALLOWED_TOOLS:
            agent_issues.append(f"{ap.name}: 想定外の tools {sorted(tools - ALLOWED_TOOLS)}")
        if ap.stem in READONLY_AGENTS and "Bash" in tools:
            agent_issues.append(f"{ap.name}: 読み取り専用のはずが Bash を持っている")
    # **文書が名指しするエージェントが実在するか**も見る。ファイル側だけを検査していると、
    # エージェントを 1 つ消しても「残った分は全部正しい」で ✅ のまま通る
    # （テストも glob の結果を回すだけなので、入力が消えれば検査項目ごと消える）。
    # 参照する側から見れば「消えたこと」が検知できる。
    known = {ap.stem for ap in agent_files}
    referenced: set[str] = set()
    for rel, text in _iter_docs():
        for name in re.findall(r"`([a-z][a-z0-9-]+)`", text):
            if name in known or name in AGENT_NAMES_EXPECTED:
                referenced.add(name)
    for missing in sorted(AGENT_NAMES_EXPECTED - known):
        agent_issues.append(f"{missing}: 文書が参照しているのに .claude/agents/ に定義が無い")
    for orphan in sorted(known - referenced):
        agent_issues.append(f"{orphan}: どの文書からも参照されていない（存在が伝わらない）")

    CHECKED["C13"] = len(agent_files) + len(AGENT_NAMES_EXPECTED)
    results.append(("ERROR" if agent_issues else "OK", "C13 エージェント定義",
                    f"{len(agent_files)} 件"
                    + ("\n      " + "\n      ".join(agent_issues) if agent_issues else "")))

    # ── C14: 文書中のコマンドが実行可能か ──
    # `uv run python scripts/x.py` は `src` を import できず ModuleNotFoundError になる。
    # 正しくは `-m scripts.x`。README の「スクリプトの実行」節が丸ごと動かない状態で
    # 長く放置されていた（feature_report.py の docstring で同じ誤りを直しながら見落とした）。
    # 人の注意ではなく機械で捕まえる。
    cmd_issues = []
    # docs（DOC_FILES + skills）だけでは **README.md を見落とす** ——
    # 壊れたコマンド 10 箇所の本体がまさに README だった。走査対象を明示的に広げる。
    cmd_targets = dict(docs)
    for extra in ["README.md", *[str(q.relative_to(ROOT)) for q in (ROOT / "state").glob("*.md")],
                  *[str(q.relative_to(ROOT)) for q in (ROOT / "experiments").rglob("*.md")]]:
        q = ROOT / extra
        if q.exists():
            cmd_targets[extra] = q.read_text()
    for rel, text in cmd_targets.items():
        if rel.startswith("docs/") or rel == "CHANGELOG.md":
            continue                      # 履歴ファイルは当時の記述が正しい
        for m in re.finditer(r"uv run python (scripts/[\w/]+\.py)", text):
            cmd_issues.append(f"{rel}: `{m.group(1)}` 形式は src を import できない（-m 形式にする）")
        # `scripts.<名前>` のような雛形記法は対象外（末尾がドット、または直後が < ）
        for m in re.finditer(r"uv run python -m (scripts(?:\.[a-z_][\w]*)+)(?![\w.<])", text):
            mod = m.group(1)
            if not (ROOT / (mod.replace(".", "/") + ".py")).exists():
                cmd_issues.append(f"{rel}: `-m {mod}` に対応するスクリプトが無い")
    CHECKED["C14"] = sum(len(re.findall(r"uv run python ", t)) for t in cmd_targets.values())
    results.append(("ERROR" if cmd_issues else "OK", "C14 文書中のコマンド",
                    f"{len(cmd_issues)} 件"
                    + ("\n      " + "\n      ".join(sorted(set(cmd_issues))[:8]) if cmd_issues else "")))

    # ── C11: README の自己申告値 vs 実測 ──
    # README は「テンプレートが何であるか」の対外的な宣言。実態からずれると、
    # 次にこの repo を開いた人（未来の自分）が誤った前提で作業を始める。
    readme_path = ROOT / "README.md"
    drift = []
    if readme_path.exists():
        readme = readme_path.read_text()
        actual_claude = len((ROOT / "CLAUDE.md").read_text())   # C1 と同じ「文字数」で測る
        actual_skills = len(list(ROOT.glob(SKILL_GLOB)))
        n_checks = TOTAL_CHECKS       # 定数にして順序依存をなくした
        claims = [
            (r"固定\s*(\d+)\s*個の数値", len(CRITICAL_NUMBERS), "C4 の実測値の個数"),
            (r"\*\*(\d[\d,]*)\s*字（-\d+%）\*\*", actual_claude, "常時ロードの文字数"),
            (r"C1-C(\d+)", n_checks, "doc_audit のチェック数"),
        ]
        for pattern, actual, label in claims:
            m = re.search(pattern, readme)
            if m and int(m.group(1).replace(",", "")) != actual:
                drift.append(f"{label}: README は {m.group(1)} / 実測 {actual}")
        CHECKED["C11"] = sum(1 for pattern, _, _ in claims if re.search(pattern, readme))
        listed = len(re.findall(r"^\|\s*`/ds-[a-z-]+`\s*\|", readme, re.M))
        if listed and listed != actual_skills:
            drift.append(f"スキル一覧: README は {listed} 件 / 実測 {actual_skills} 件")

        # ディレクトリ構成図とリポジトリのトップレベルの過不足。
        # 新しいディレクトリを作っても構成図を直し忘れると、README を頼りに
        # 探した人が見つけられない（state/ tests/ scripts/harness/ .claude/agents/ で実際に起きた）。
        m_tree = re.search(r"## ディレクトリ構成\n\n```\n(.*?)```", readme, re.S)
        if m_tree:
            tree = m_tree.group(1)
            IGNORE = {".git", ".venv", ".pytest_cache", "__pycache__", "catboost_info",
                      "uv.lock", "pyproject.toml", ".gitignore", ".kaggleignore",
                      ".python-version", "dataset-metadata.json.template"}
            actual_top = {q.name for q in ROOT.iterdir()
                          if not q.name.startswith(".") or q.name == ".claude"} - IGNORE
            undocumented = sorted(n for n in actual_top if n not in tree)
            if undocumented:
                drift.append(f"構成図に無いトップレベル: {', '.join(undocumented)}")
    results.append(("WARNING" if drift else "OK", "C11 README の同期",
                    f"実態とのズレ {len(drift)} 件"
                    + ("\n      " + "\n      ".join(drift) if drift else "")))

    # ── C15: ガードの空洞検知 ──
    # 「問題 0 件」と「0 件しか検査していない」は別物。後者はガードが死んでいる状態で、
    # 表示上はどちらも ✅ になる。分母を持つチェックについて、それがゼロなら ERROR にする。
    EXPECTED_NONZERO = {"C2": "アンカー参照", "C3": "指針の ID 定義", "C6": "SSoT の語句",
                        "C11": "README の自己申告値", "C13": "エージェント定義",
                        "C14": "文書中のコマンド"}
    hollow = [f"{k}（{label}）の検査対象が 0 件 —— ガードが何も見ていない"
              for k, label in EXPECTED_NONZERO.items() if CHECKED.get(k, 0) == 0]
    if len(results) + 1 != TOTAL_CHECKS:      # 自分自身を足した数
        hollow.append(f"TOTAL_CHECKS={TOTAL_CHECKS} が実際の検査数 {len(results) + 1} と不一致")
    detail = " / ".join(f"{k}={CHECKED.get(k, 0)}" for k in EXPECTED_NONZERO)
    results.append(("ERROR" if hollow else "OK", "C15 ガードの空洞検知",
                    detail + ("\n      " + "\n      ".join(hollow) if hollow else "")))




def main() -> int:
    # 引数は取らない。以前あった `--baseline-write` は**誰も読まないファイルを書くだけ**で、
    # コミットされたベースラインは実態から大きくずれたまま放置されていた。
    # 「あるのに何もしていない仕組み」は、無いより悪い（あると思って安心する）。
    argparse.ArgumentParser(description=__doc__.splitlines()[1]).parse_args()

    results: list[tuple[str, str, str]] = []
    check(results)

    icon = {"OK": "✅", "WARNING": "⚠️ ", "ERROR": "❌"}
    print("=" * 72)
    print(" ドキュメント階層 検査（scripts/harness/doc_audit.py）")
    print("=" * 72)
    for level, name, msg in results:
        print(f"{icon[level]} {name}: {msg}")

    n_err = sum(1 for lvl, _, _ in results if lvl == "ERROR")
    n_warn = sum(1 for lvl, _, _ in results if lvl == "WARNING")
    print("-" * 72)
    print(f"ERROR {n_err} / WARNING {n_warn}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

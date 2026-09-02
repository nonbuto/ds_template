"""
ドキュメント階層の自動検査（読み取り専用）

CLAUDE.md / CONVENTIONS.md / PLAYBOOK.md / .claude/skills の 4 層構造が
SSoT 原則を守れているかを機械的に検証する。

設計思想:
    行数や見出しの diff では「教訓の実測数値が静かに消えた」ことを検知できない。
    数値の grep なら 100% 検知できる（C4 がこのチェッカーの主役）。

使い方:
    uv run python -m scripts.harness.doc_audit                    # 検査
    uv run python -m scripts.harness.doc_audit --baseline-write   # 現状をベースラインとして保存
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/harness/ から見たリポジトリルート
BASELINE_PATH = ROOT / "experiments" / "doc_audit_baseline.json"

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
ALWAYS_LOADED_BUDGET = 5_000

DOC_FILES = ["CLAUDE.md", "GUIDELINES.md", "CONVENTIONS.md", "PLAYBOOK.md"]
SKILL_GLOB = ".claude/skills/*/SKILL.md"

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
    for p in sorted((ROOT / ".claude" / "rules").glob("*.md")):
        yield str(p.relative_to(ROOT)), p.read_text()


def _slug(heading: str) -> str:
    """GitHub 式のアンカー slug に変換する。"""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s぀-ヿ一-鿿-]", "", s)
    return re.sub(r"\s+", "-", s).strip("-")


def _headings(text: str) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M)}


def check(results: list[tuple[str, str, str]]) -> None:
    docs = dict(_iter_docs())

    # ── C1: 常時ロードの行数予算 ──
    present = [f for f in ALWAYS_LOADED if (ROOT / f).exists()]
    total = sum(len((ROOT / f).read_text()) for f in present)
    lines = sum(len((ROOT / f).read_text().splitlines()) for f in present)
    detail = ", ".join(f"{f}={len((ROOT/f).read_text()):,}字" for f in present)
    lvl = "ERROR" if total > ALWAYS_LOADED_BUDGET else "OK"
    results.append((lvl, "C1 文字数予算",
                    f"常時ロード {total:,} 字 / 上限 {ALWAYS_LOADED_BUDGET:,}"
                    f"（{detail} · 参考 {lines} 行）"))

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
    results.append(("ERROR" if viol else "OK", "C6 SSoT 違反",
                    f"{len(viol)} 件" + ("\n      " + "\n      ".join(viol[:8]) if viol else "")))

    # ── C7: コンペ識別子の混入 ──
    leaks = [f"{rel} ({len(COMPETITION_TOKENS.findall(text))} 件)"
             for rel, text in docs.items()
             if rel not in COMPETITION_ALLOWED and COMPETITION_TOKENS.search(text)]
    results.append(("ERROR" if leaks else "OK", "C7 コンペ識別子",
                    f"{len(leaks)} ファイルに混入" + ("\n      " + "\n      ".join(leaks) if leaks else "")))

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

    # ── C11: README の自己申告値 vs 実測 ──
    # README は「テンプレートが何であるか」の対外的な宣言。実態からずれると、
    # 次にこの repo を開いた人（未来の自分）が誤った前提で作業を始める。
    readme_path = ROOT / "README.md"
    drift = []
    if readme_path.exists():
        readme = readme_path.read_text()
        actual_claude = len((ROOT / "CLAUDE.md").read_text())   # C1 と同じ「文字数」で測る
        actual_skills = len(list(ROOT.glob(SKILL_GLOB)))
        n_checks = len(results) + 1   # 自分自身を含めた総数。**新しい検査は C11 より前に置く**
                                      # （後ろに置くと C11 が数え落として README がずれる）
        claims = [
            (r"固定\s*(\d+)\s*個の数値", len(CRITICAL_NUMBERS), "C4 の実測値の個数"),
            (r"\*\*(\d[\d,]*)\s*字（-\d+%）\*\*", actual_claude, "常時ロードの文字数"),
            (r"C1-C(\d+)", n_checks, "doc_audit のチェック数"),
        ]
        for pattern, actual, label in claims:
            m = re.search(pattern, readme)
            if m and int(m.group(1).replace(",", "")) != actual:
                drift.append(f"{label}: README は {m.group(1)} / 実測 {actual}")
        listed = len(re.findall(r"^\|\s*`/ds-[a-z-]+`\s*\|", readme, re.M))
        if listed and listed != actual_skills:
            drift.append(f"スキル一覧: README は {listed} 件 / 実測 {actual_skills} 件")
    results.append(("WARNING" if drift else "OK", "C11 README の同期",
                    f"実態とのズレ {len(drift)} 件"
                    + ("\n      " + "\n      ".join(drift) if drift else "")))



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-write", action="store_true", help="現状をベースラインとして保存する")
    args = ap.parse_args()

    if args.baseline_write:
        snap = {rel: {"lines": len(text.splitlines()), "chars": len(text)} for rel, text in _iter_docs()}
        snap["_critical_numbers"] = CRITICAL_NUMBERS
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2))
        print(f"ベースラインを保存: {BASELINE_PATH.relative_to(ROOT)}（{len(snap)-1} ファイル）")
        return 0

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

#!/usr/bin/env python3
"""
Translate Vietnamese text to English, French, Spanish using GPT-4o-mini.
Reads from the original (un-normalized) CSV for better translation quality.
Outputs to a separate CSV file.

Usage:
    python scripts/visl_translate_gpt.py --api-key "sk-..." --dry-run 5
    python scripts/visl_translate_gpt.py --api-key "sk-..." --resume
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────────
ORIGINAL_CSV = Path("/workspace/khanh/SpaMo/dataset_VSL/sentence_clips_metadata_with_signers_original.csv")
OUTPUT_CSV = Path("/workspace/khanh/SpaMo/dataset_VSL/sentence_clips_metadata_translated_gpt.csv")
CHECKPOINT_PATH = Path("/workspace/khanh/SpaMo/dataset_VSL/translate_gpt_checkpoint.json")

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
BASE_DELAY = 1.0

SYSTEM_PROMPT = """You are a professional translator. Translate the given Vietnamese sentences into English, French, and Spanish.

Return a JSON object with key "results" containing an array of objects, each with "en", "fr", "es" fields.

Example input:
["Phát huy truyền thống tương thân tương ái, lá lành đùm lá rách của dân tộc."]

Example output:
{"results": [{"en": "Promoting the tradition of mutual love and care, the strong protecting the weak of the nation.", "fr": "Promouvoir la tradition d'entraide et de solidarité, les forts protégeant les faibles de la nation.", "es": "Promover la tradición de amor mutuo y cuidado, los fuertes protegiendo a los débiles de la nación."}]}

Important:
- Translate accurately, preserving the original meaning
- Keep proper nouns as-is when appropriate
- Return exactly the same number of translations as input sentences"""


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_single(client: OpenAI, text: str) -> dict:
    """Translate a single sentence (fallback)."""
    user_message = json.dumps([text], ensure_ascii=False)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096,
            )
            result = json.loads(response.choices[0].message.content)
            if "results" in result and len(result["results"]) >= 1:
                return result["results"][0]
        except Exception:
            time.sleep(BASE_DELAY * (2 ** attempt))
    return {"en": text, "fr": text, "es": text}


def translate_batch(client: OpenAI, texts: list[str]) -> list[dict]:
    """Translate a batch of texts. Returns list of {"en": ..., "fr": ..., "es": ...}."""
    user_message = json.dumps(texts, ensure_ascii=False)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096,
            )
            result = json.loads(response.choices[0].message.content)
            if "results" in result and len(result["results"]) == len(texts):
                return result["results"]
            else:
                print(f"\n⚠️  Length mismatch: expected {len(texts)}, got {len(result.get('results', []))}. Retrying...")
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON decode error: {e}. Retrying...")
        except Exception as e:
            print(f"\n⚠️  API error (attempt {attempt + 1}/3): {e}")
        time.sleep(BASE_DELAY * (2 ** attempt))

    # Fallback: single-sentence mode
    print(f"\n🔄 Batch failed, falling back to single-sentence mode for {len(texts)} sentences...")
    results = []
    for text in texts:
        results.append(translate_single(client, text))
        time.sleep(0.05)
    return results


def main():
    parser = argparse.ArgumentParser(description="Translate VSL text using GPT-4o-mini")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Process only first N rows (0 = all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Sentences per API call (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    # ── API Key ─────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found. Set OPENAI_API_KEY or pass --api-key")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # ── Load data ───────────────────────────────────────────────────────
    print(f"📂 Loading original CSV: {ORIGINAL_CSV}")
    rows = load_csv(ORIGINAL_CSV)
    total = len(rows)
    print(f"   Total rows: {total}")

    # Add translation columns
    for r in rows:
        r.setdefault('en_text', '')
        r.setdefault('fr_text', '')
        r.setdefault('es_text', '')

    # ── Determine range ─────────────────────────────────────────────────
    if args.dry_run > 0:
        process_count = min(args.dry_run, total)
        print(f"🧪 Dry run: first {process_count} rows")
    else:
        process_count = total

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = {}
    if args.resume:
        checkpoint = load_checkpoint(CHECKPOINT_PATH)
        if checkpoint:
            print(f"📌 Resuming: {len(checkpoint)} rows already translated")

    # ── Translate ───────────────────────────────────────────────────────
    batch_size = args.batch_size
    processed = 0
    skipped = 0

    print(f"\n🌐 Translating (batch_size={batch_size})...\n")
    pbar = tqdm(total=process_count, desc="Translating", unit="row")

    # Apply checkpoint
    for i in range(process_count):
        idx_str = str(i)
        if idx_str in checkpoint:
            cp = checkpoint[idx_str]
            rows[i]['en_text'] = cp['en']
            rows[i]['fr_text'] = cp['fr']
            rows[i]['es_text'] = cp['es']
            skipped += 1
            pbar.update(1)

    if skipped > 0:
        print(f"   ⏭️  Skipped {skipped} already-translated rows")

    i = 0
    while i < process_count:
        batch_indices = []
        batch_texts = []

        while len(batch_texts) < batch_size and i < process_count:
            if str(i) not in checkpoint:
                batch_indices.append(i)
                batch_texts.append(rows[i]['text'])
            i += 1

        if not batch_texts:
            continue

        # Call API
        translations = translate_batch(client, batch_texts)

        # Update
        for idx, trans in zip(batch_indices, translations):
            rows[idx]['en_text'] = trans.get('en', '')
            rows[idx]['fr_text'] = trans.get('fr', '')
            rows[idx]['es_text'] = trans.get('es', '')
            checkpoint[str(idx)] = trans
            processed += 1

        pbar.update(len(batch_indices))

        # Save checkpoint every 5 batches
        if (processed // batch_size) % 5 == 0:
            save_checkpoint(CHECKPOINT_PATH, checkpoint)

        time.sleep(0.1)

    pbar.close()
    save_checkpoint(CHECKPOINT_PATH, checkpoint)

    # ── Save output ─────────────────────────────────────────────────────
    if args.dry_run > 0:
        dry_path = OUTPUT_CSV.with_name("translated_gpt_dry_run.csv")
        save_csv(rows[:process_count], dry_path)
        print(f"\n📝 Dry run saved to: {dry_path}")

        print("\n" + "=" * 80)
        for j in range(process_count):
            print(f"\n[{j}] VI: {rows[j]['text']}")
            print(f"    EN: {rows[j]['en_text']}")
            print(f"    FR: {rows[j]['fr_text']}")
            print(f"    ES: {rows[j]['es_text']}")
    else:
        save_csv(rows, OUTPUT_CSV)
        print(f"\n📝 Output saved to: {OUTPUT_CSV}")

    print(f"\n✅ Done! Translated {processed}, skipped {skipped}")

    if not args.dry_run:
        print(f"\n🧹 Removing checkpoint")
        CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

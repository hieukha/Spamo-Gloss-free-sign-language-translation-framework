#!/usr/bin/env python3
"""
Normalize Vietnamese text in VSL dataset using GPT-4o-mini API.

GPT handles everything: spelling correction, number→word, punctuation removal, lowercase.

Usage:
    # Dry run on first 5 rows
    python normalize_vsl_text.py --dry-run 5

    # Full run
    python normalize_vsl_text.py

    # Resume from checkpoint
    python normalize_vsl_text.py --resume

    # Custom batch size
    python normalize_vsl_text.py --batch-size 20
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("openai package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = Path("/workspace/khanh/SpaMo/dataset_VSL/sentence_clips_metadata_with_signers.csv")
BACKUP_PATH = CSV_PATH.with_name(CSV_PATH.stem + "_original" + CSV_PATH.suffix)
CHECKPOINT_PATH = CSV_PATH.parent / "normalize_checkpoint.json"
OUTPUT_PATH = CSV_PATH  # overwrite in-place after backup

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds

SYSTEM_PROMPT = """Bạn là trợ lý chuyên chuẩn hóa văn bản tiếng Việt. Với mỗi câu được đưa vào, bạn cần:

1. **Sửa lỗi chính tả**: Text này được tạo bởi hệ thống nhận dạng giọng nói (ASR) nên có thể có lỗi chính tả. Hãy sửa các lỗi chính tả rõ ràng. Ví dụ: "lá sách" → "lá rách", "nguyên đáng" → "nguyên đán", "giáp thình" → "giáp thìn", "tiền lực" → "điện lực".

2. **Chuyển số thành chữ**: Chuyển tất cả chữ số thành dạng chữ viết tiếng Việt. Ví dụ:
   - 2024 → "hai nghìn không trăm hai mươi tư"
   - 1.000 → "một nghìn" (dấu chấm là dấu phân cách hàng nghìn)
   - 110kV → "một trăm mười ki lô vôn"
   - 57,33 km → "năm mươi bảy phẩy ba ba ki lô mét"
   - 45% → "bốn mươi lăm phần trăm"
   - TP.HCM → "thành phố hồ chí minh"
   - 121 → "một trăm hai mươi mốt"
   - 54 → "năm mươi tư"

3. **Bỏ dấu câu**: Loại bỏ tất cả dấu câu (dấu chấm, phẩy, chấm hỏi, chấm than, ngoặc, gạch ngang, dấu ngoặc kép, v.v.)

4. **Chữ thường**: Chuyển tất cả chữ thành chữ thường.

Trả về kết quả dưới dạng JSON với key "results" là một mảng các câu đã chuẩn hóa, theo đúng thứ tự đầu vào.

Ví dụ input:
["Phát huy truyền thống tương thân tương ái, lá lành đùm lá sách của dân tộc, nhân dịp đón Tết nguyên đáng giáp thình 2024, Tổng Công ty Điện lực Miền Nam đã trao tặng khoảng 1.000 phần quà Tết đến các hộ nghèo, cận nghèo, gia đình chính sách tại các tỉnh thành phía Nam."]

Ví dụ output:
{"results": ["phát huy truyền thống tương thân tương ái lá lành đùm lá rách của dân tộc nhân dịp đón tết nguyên đán giáp thìn hai nghìn không trăm hai mươi tư tổng công ty điện lực miền nam đã trao tặng khoảng một nghìn phần quà tết đến các hộ nghèo cận nghèo gia đình chính sách tại các tỉnh thành phía nam"]}"""


def load_csv(path: Path) -> list[dict]:
    """Load CSV file into list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path):
    """Save list of dicts to CSV file."""
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint if exists."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict):
    """Save checkpoint data."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_single(client: OpenAI, text: str) -> str:
    """Send a single text to GPT-4o-mini for normalization (fallback)."""
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

            content = response.choices[0].message.content
            result = json.loads(content)

            if "results" in result and len(result["results"]) >= 1:
                # If GPT split it into multiple parts, join them
                return " ".join(result["results"])

        except Exception as e:
            delay = BASE_DELAY * (2 ** attempt)
            time.sleep(delay)

    return text  # keep original if all retries fail


def normalize_batch(client: OpenAI, texts: list[str]) -> list[str]:
    """Send a batch of texts to GPT-4o-mini for full normalization."""
    user_message = json.dumps(texts, ensure_ascii=False)

    for attempt in range(3):  # Try batch 3 times
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

            content = response.choices[0].message.content
            result = json.loads(content)

            if "results" in result and len(result["results"]) == len(texts):
                return result["results"]
            else:
                print(f"\n⚠️  Length mismatch: expected {len(texts)}, got {len(result.get('results', []))}. Retrying...")

        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON decode error: {e}. Retrying...")
        except Exception as e:
            print(f"\n⚠️  API error (attempt {attempt + 1}/3): {e}")

        delay = BASE_DELAY * (2 ** attempt)
        time.sleep(delay)

    # Fallback: process each sentence individually
    print(f"\n🔄 Batch failed, falling back to single-sentence mode for {len(texts)} sentences...")
    results = []
    for text in texts:
        results.append(normalize_single(client, text))
        time.sleep(0.05)
    return results


def main():
    parser = argparse.ArgumentParser(description="Normalize Vietnamese text using GPT-4o-mini")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Process only first N rows for testing (0 = process all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of sentences per API call (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    # ── API Key ─────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found. Set OPENAI_API_KEY or pass --api-key")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ── Load data ───────────────────────────────────────────────────────────
    print(f"📂 Loading CSV: {CSV_PATH}")
    rows = load_csv(CSV_PATH)
    total = len(rows)
    print(f"   Total rows: {total}")

    # ── Backup ──────────────────────────────────────────────────────────────
    if not BACKUP_PATH.exists():
        print(f"💾 Creating backup: {BACKUP_PATH}")
        shutil.copy2(CSV_PATH, BACKUP_PATH)
    else:
        print(f"💾 Backup already exists: {BACKUP_PATH}")

    # ── Determine range ─────────────────────────────────────────────────────
    if args.dry_run > 0:
        process_count = min(args.dry_run, total)
        print(f"🧪 Dry run mode: processing first {process_count} rows")
    else:
        process_count = total

    # ── Load checkpoint ─────────────────────────────────────────────────────
    checkpoint = {}
    if args.resume:
        checkpoint = load_checkpoint(CHECKPOINT_PATH)
        if checkpoint:
            print(f"📌 Resuming from checkpoint: {len(checkpoint)} rows already processed")
        else:
            print("📌 No checkpoint found, starting from scratch")

    # ── Process ─────────────────────────────────────────────────────────────
    batch_size = args.batch_size
    processed = 0
    skipped = 0

    # Keep originals for comparison in dry-run
    original_texts = [rows[i]["text"] for i in range(process_count)]

    print(f"\n🚀 Starting normalization (batch_size={batch_size})...\n")

    pbar = tqdm(total=process_count, desc="Normalizing", unit="row")

    # Apply checkpoint data
    for i in range(process_count):
        idx_str = str(i)
        if idx_str in checkpoint:
            rows[i]["text"] = checkpoint[idx_str]
            skipped += 1
            pbar.update(1)

    if skipped > 0:
        print(f"   ⏭️  Skipped {skipped} already-processed rows from checkpoint")

    i = 0
    while i < process_count:
        batch_indices = []
        batch_texts = []

        while len(batch_texts) < batch_size and i < process_count:
            idx_str = str(i)
            if idx_str not in checkpoint:
                batch_indices.append(i)
                batch_texts.append(rows[i]["text"])
            i += 1

        if not batch_texts:
            continue

        # Call API
        normalized = normalize_batch(client, batch_texts)

        # Update rows and checkpoint
        for idx, norm_text in zip(batch_indices, normalized):
            rows[idx]["text"] = norm_text
            checkpoint[str(idx)] = norm_text
            processed += 1

        pbar.update(len(batch_indices))

        # Save checkpoint periodically (every 5 batches)
        if (processed // batch_size) % 5 == 0:
            save_checkpoint(CHECKPOINT_PATH, checkpoint)

        time.sleep(0.1)

    pbar.close()

    # ── Save final checkpoint ───────────────────────────────────────────────
    save_checkpoint(CHECKPOINT_PATH, checkpoint)

    # ── Save CSV ────────────────────────────────────────────────────────────
    if args.dry_run > 0:
        dry_run_path = CSV_PATH.with_name("normalized_dry_run.csv")
        print(f"\n📝 Dry run output saved to: {dry_run_path}")
        save_csv(rows[:process_count], dry_run_path)

        print("\n" + "=" * 80)
        print("📊 COMPARISON (original → normalized)")
        print("=" * 80)
        for j in range(process_count):
            print(f"\n[{j}] ORIGINAL:   {original_texts[j]}")
            print(f"    NORMALIZED: {rows[j]['text']}")
    else:
        print(f"\n📝 Saving normalized CSV to: {OUTPUT_PATH}")
        save_csv(rows, OUTPUT_PATH)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n✅ Done! Processed {processed} rows, skipped {skipped} (from checkpoint)")
    print(f"   Backup: {BACKUP_PATH}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")

    if not args.dry_run:
        print(f"\n🧹 Removing checkpoint file (full run completed)")
        CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Classify VSL videos into 11 topics using GPT-4o-mini.
Reads transcripts from JSON files and classifies each video.

Usage:
    python scripts/visl_classify_topics.py --api-key "sk-..." --dry-run 5
    python scripts/visl_classify_topics.py --api-key "sk-..." --resume
    python scripts/visl_classify_topics.py --api-key "sk-..." --batch-size 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

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
TRANSCRIPTS_DIR = Path("/workspace/khanh/SpaMo/dataset_VSL/transcripts")
OUTPUT_PATH = Path("/workspace/khanh/SpaMo/dataset_VSL/video_topics.json")
CHECKPOINT_PATH = Path("/workspace/khanh/SpaMo/dataset_VSL/classify_topics_checkpoint.json")

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
BASE_DELAY = 1.0

TOPICS = [
    "Thời sự",
    "Thế giới",
    "Chính trị",
    "Kinh tế",
    "Đời sống",
    "Văn hóa - Giải trí",
    "Công nghệ",
    "Sức khỏe",
    "Thể thao",
    "Giáo dục",
    "HTV Show",
]

SYSTEM_PROMPT = f"""Bạn là một chuyên gia phân loại nội dung tin tức tiếng Việt.

Nhiệm vụ: Đọc phụ đề của các video tin tức và phân loại mỗi video vào ĐÚNG MỘT trong 11 chủ đề sau:
{json.dumps(TOPICS, ensure_ascii=False)}

Trả về JSON object với key "results" chứa array các object, mỗi object có:
- "video": tên video (giữ nguyên từ input)
- "topic": chủ đề được chọn (phải là 1 trong 11 chủ đề trên)
- "confidence": độ tự tin từ 0.0 đến 1.0

Ví dụ output:
{{"results": [{{"video": "video-name", "topic": "Thời sự", "confidence": 0.9}}]}}

Lưu ý:
- Chỉ chọn 1 chủ đề phù hợp nhất cho mỗi video
- Phải chọn từ danh sách 11 chủ đề, KHÔNG được tự tạo chủ đề mới
- Trả về đúng số lượng kết quả bằng số video input"""


def load_transcript(json_path: Path) -> str:
    """Load transcript and concatenate all segment texts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    texts = [seg["text"].strip() for seg in segments if seg.get("text")]
    return " ".join(texts)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def classify_batch(client: OpenAI, batch: list[dict]) -> list[dict]:
    """
    Classify a batch of videos. Each item has 'name' and 'transcript'.
    Returns list of {"video": ..., "topic": ..., "confidence": ...}.
    """
    # Truncate transcripts to save tokens (first 500 chars is enough for classification)
    user_data = []
    for item in batch:
        transcript = item["transcript"][:1000]  # limit to 1000 chars
        user_data.append({
            "video": item["name"],
            "transcript": transcript,
        })

    user_message = json.dumps(user_data, ensure_ascii=False)

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

            if "results" in result and len(result["results"]) == len(batch):
                # Validate topics
                for r in result["results"]:
                    if r.get("topic") not in TOPICS:
                        r["topic"] = "Thời sự"  # fallback
                        r["confidence"] = 0.0
                return result["results"]
            else:
                print(f"\n⚠️  Length mismatch: expected {len(batch)}, got {len(result.get('results', []))}. Retrying...")
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON decode error: {e}. Retrying...")
        except Exception as e:
            print(f"\n⚠️  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
        time.sleep(BASE_DELAY * (2 ** attempt))

    # Fallback: classify one by one
    print(f"\n🔄 Batch failed, falling back to single-video mode for {len(batch)} videos...")
    results = []
    for item in batch:
        result = classify_batch(client, [item])
        results.extend(result)
        time.sleep(0.1)
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify VSL videos into topics using GPT-4o-mini")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Process only first N videos (0 = all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Videos per API call (default: 10)")
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

    # ── Load transcripts ────────────────────────────────────────────────
    print(f"📂 Loading transcripts from: {TRANSCRIPTS_DIR}")
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    print(f"   Found {len(transcript_files)} transcript files")

    videos = []
    for tf in transcript_files:
        video_name = tf.stem  # filename without .json
        transcript = load_transcript(tf)
        if transcript.strip():
            videos.append({"name": video_name, "transcript": transcript})

    print(f"   Videos with non-empty transcripts: {len(videos)}")

    # ── Determine range ─────────────────────────────────────────────────
    if args.dry_run > 0:
        videos = videos[:args.dry_run]
        print(f"🧪 Dry run: first {len(videos)} videos")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = {}
    if args.resume:
        checkpoint = load_checkpoint(CHECKPOINT_PATH)
        if checkpoint:
            print(f"📌 Resuming: {len(checkpoint)} videos already classified")

    # ── Classify ────────────────────────────────────────────────────────
    batch_size = args.batch_size
    processed = 0
    skipped = 0

    print(f"\n🏷️  Classifying into {len(TOPICS)} topics (batch_size={batch_size})...\n")
    pbar = tqdm(total=len(videos), desc="Classifying", unit="video")

    # Collect results
    all_results = {}

    # Apply checkpoint
    for v in videos:
        if v["name"] in checkpoint:
            all_results[v["name"]] = checkpoint[v["name"]]
            skipped += 1
            pbar.update(1)

    if skipped > 0:
        print(f"   ⏭️  Skipped {skipped} already-classified videos")

    # Process remaining in batches
    remaining = [v for v in videos if v["name"] not in checkpoint]

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]

        results = classify_batch(client, batch)

        for item, result in zip(batch, results):
            entry = {
                "topic": result.get("topic", "Thời sự"),
                "confidence": result.get("confidence", 0.0),
            }
            all_results[item["name"]] = entry
            checkpoint[item["name"]] = entry
            processed += 1

        pbar.update(len(batch))

        # Save checkpoint every 5 batches
        if (processed // batch_size) % 5 == 0:
            save_checkpoint(CHECKPOINT_PATH, checkpoint)

        time.sleep(0.1)

    pbar.close()
    save_checkpoint(CHECKPOINT_PATH, checkpoint)

    # ── Save output ─────────────────────────────────────────────────────
    # Group by topic
    topic_groups = {topic: [] for topic in TOPICS}
    for video_name, info in all_results.items():
        topic = info["topic"]
        if topic in topic_groups:
            topic_groups[topic].append({
                "name": video_name,
                "confidence": info["confidence"],
            })

    output = {
        "total_videos": len(all_results),
        "topics": TOPICS,
        "topic_counts": {t: len(v) for t, v in topic_groups.items()},
        "topic_groups": topic_groups,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"📊 Classification Results")
    print(f"{'=' * 60}")
    print(f"Total videos classified: {len(all_results)}")
    print(f"\nTopic distribution:")
    for topic in TOPICS:
        count = len(topic_groups[topic])
        bar = "█" * (count // 10)
        print(f"  {topic:20s}: {count:4d} videos {bar}")

    print(f"\n📝 Results saved to: {OUTPUT_PATH}")
    print(f"✅ Done! Classified {processed}, skipped {skipped}")

    if not args.dry_run:
        print(f"\n🧹 Removing checkpoint")
        CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

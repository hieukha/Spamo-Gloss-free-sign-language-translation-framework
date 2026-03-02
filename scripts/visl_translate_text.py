"""
Script to translate Vietnamese text to English, French, and Spanish using Google Translate.

Reads original (un-normalized) text from _original.csv for better translation quality,
then writes en_text, fr_text, es_text columns to the normalized CSV.

Usage:
    python scripts/visl_translate_text.py
    python scripts/visl_translate_text.py --dry-run 5

Requirements:
    pip install deep-translator
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import json

# Try to import translation libraries
try:
    from deep_translator import GoogleTranslator
    USE_DEEP_TRANSLATOR = True
except ImportError:
    try:
        from googletrans import Translator
        USE_DEEP_TRANSLATOR = False
    except ImportError:
        print("Please install a translation library:")
        print("  pip install deep-translator")
        print("  or")
        print("  pip install googletrans==4.0.0-rc1")
        exit(1)


# ── Config ──────────────────────────────────────────────────────────────────
ORIGINAL_CSV = Path("/workspace/khanh/SpaMo/dataset_VSL/sentence_clips_metadata_with_signers_original.csv")
NORMALIZED_CSV = Path("/workspace/khanh/SpaMo/dataset_VSL/sentence_clips_metadata_with_signers.csv")
CACHE_FILE = Path("/workspace/khanh/SpaMo/dataset_VSL/translation_cache.json")


class TranslationService:
    """Wrapper for translation services with rate limiting and caching."""
    
    def __init__(self, cache_file: str = None):
        self.cache = {}
        self.cache_file = cache_file
        
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached translations")
        
        if USE_DEEP_TRANSLATOR:
            self.translators = {
                'en': GoogleTranslator(source='vi', target='en'),
                'fr': GoogleTranslator(source='vi', target='fr'),
                'es': GoogleTranslator(source='vi', target='es'),
            }
        else:
            self.translator = Translator()
    
    def translate(self, text: str, target_lang: str, max_retries: int = 5) -> str:
        """Translate Vietnamese text to target language."""
        cache_key = f"{text}_{target_lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        for attempt in range(max_retries):
            try:
                if USE_DEEP_TRANSLATOR:
                    result = self.translators[target_lang].translate(text)
                else:
                    result = self.translator.translate(text, src='vi', dest=target_lang).text
                
                self.cache[cache_key] = result
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 1 * (2 ** attempt)
                    time.sleep(delay)
                else:
                    print(f"\n❌ Translation failed for '{text[:50]}...': {e}")
                    return text  # Return original text on failure
        
        return text
    
    def save_cache(self):
        """Save translation cache to file."""
        if self.cache_file:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Translate VSL texts to en/fr/es')
    parser.add_argument('--dry-run', type=int, default=0,
                        help='Process only first N rows for testing (0 = all)')
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────────────
    print(f"📂 Loading original CSV: {ORIGINAL_CSV}")
    df_original = pd.read_csv(ORIGINAL_CSV)
    print(f"   Original rows: {len(df_original)}")

    print(f"📂 Loading normalized CSV: {NORMALIZED_CSV}")
    df_normalized = pd.read_csv(NORMALIZED_CSV)
    print(f"   Normalized rows: {len(df_normalized)}")

    # Verify both CSVs have the same 'name' column
    orig_names = set(df_original['name'])
    norm_names = set(df_normalized['name'])
    if orig_names != norm_names:
        diff = orig_names.symmetric_difference(norm_names)
        print(f"⚠️  WARNING: {len(diff)} mismatched names between CSVs")

    # ── Determine range ─────────────────────────────────────────────────
    if args.dry_run > 0:
        process_count = min(args.dry_run, len(df_original))
        print(f"🧪 Dry run mode: processing first {process_count} rows")
    else:
        process_count = len(df_original)

    # ── Initialize translator ───────────────────────────────────────────
    translator = TranslationService(cache_file=str(CACHE_FILE))

    # ── Add columns to normalized CSV ───────────────────────────────────
    for lang in ['en_text', 'fr_text', 'es_text']:
        if lang not in df_normalized.columns:
            df_normalized[lang] = ''

    # ── Build name→index map for normalized CSV ─────────────────────────
    norm_name_to_idx = {name: idx for idx, name in enumerate(df_normalized['name'])}

    # ── Translate ───────────────────────────────────────────────────────
    print(f"\n🌐 Translating {process_count} sentences (vi → en, fr, es)...\n")

    skipped = 0
    translated = 0

    for i in tqdm(range(process_count), desc="Translating", unit="row"):
        row = df_original.iloc[i]
        name = row['name']
        original_text = row['text']

        # Find corresponding row in normalized CSV
        norm_idx = norm_name_to_idx.get(name)
        if norm_idx is None:
            continue

        # Skip if already translated
        existing_en = df_normalized.at[norm_idx, 'en_text']
        if pd.notna(existing_en) and str(existing_en).strip() != '':
            skipped += 1
            continue

        # Translate from original (un-normalized) text
        en = translator.translate(original_text, 'en')
        fr = translator.translate(original_text, 'fr')
        es = translator.translate(original_text, 'es')

        df_normalized.at[norm_idx, 'en_text'] = en
        df_normalized.at[norm_idx, 'fr_text'] = fr
        df_normalized.at[norm_idx, 'es_text'] = es

        translated += 1

        # Rate limiting
        time.sleep(0.05)

        # Save periodically
        if translated % 100 == 0:
            df_normalized.to_csv(NORMALIZED_CSV, index=False)
            translator.save_cache()

    # ── Save final ──────────────────────────────────────────────────────
    df_normalized.to_csv(NORMALIZED_CSV, index=False)
    translator.save_cache()

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n✅ Done! Translated {translated}, skipped {skipped} (already done)")
    print(f"   Output: {NORMALIZED_CSV}")
    print(f"   Cache: {CACHE_FILE} ({len(translator.cache)} entries)")

    # Show sample
    if translated > 0 or skipped > 0:
        sample = df_normalized.iloc[0]
        print(f"\n📊 Sample:")
        print(f"  VI (normalized): {sample['text']}")
        print(f"  EN: {sample.get('en_text', 'N/A')}")
        print(f"  FR: {sample.get('fr_text', 'N/A')}")
        print(f"  ES: {sample.get('es_text', 'N/A')}")


if __name__ == '__main__':
    main()

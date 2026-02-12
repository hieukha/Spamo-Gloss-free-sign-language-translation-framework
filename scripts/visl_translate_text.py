"""
Script to translate Vietnamese text to English, French, and Spanish using Google Translate.
Usage:
    python scripts/visl_translate_text.py --mode train val test
    
Requirements:
    pip install googletrans==4.0.0-rc1
    # or use deep-translator for more stability:
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
    
    def translate(self, text: str, target_lang: str, max_retries: int = 3) -> str:
        """
        Translate Vietnamese text to target language.
        
        Args:
            text: Vietnamese text to translate
            target_lang: Target language code (en, fr, es)
            max_retries: Maximum number of retries on failure
            
        Returns:
            Translated text
        """
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
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Translation failed for '{text[:50]}...': {e}")
                    return text  # Return original text on failure
        
        return text
    
    def save_cache(self):
        """Save translation cache to file."""
        if self.cache_file:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.cache)} translations to cache")


def process_mode(dataset_root: str, mode: str, translator: TranslationService):
    """
    Translate all texts for a given mode (train/val/test).
    
    Args:
        dataset_root: Root directory of ViSL dataset
        mode: Dataset mode (train, val, test)
        translator: Translation service instance
    """
    mode_dir = Path(dataset_root) / mode
    metadata_path = mode_dir / 'sentence_clips_metadata.csv'
    output_path = mode_dir / 'sentence_clips_metadata_translated.csv'
    
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    print(f"\n{'='*50}")
    print(f"Translating {mode} set: {len(df)} sentences")
    print(f"{'='*50}")
    
    # Add translation columns if not exist
    for lang in ['en', 'fr', 'es']:
        if f'{lang}_text' not in df.columns:
            df[f'{lang}_text'] = ''
    
    # Translate each sentence
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Translating {mode}"):
        text = row['text']
        
        # Skip if already translated
        if pd.notna(row.get('en_text')) and row.get('en_text', '') != '':
            continue
        
        # Translate to each language
        df.at[idx, 'en_text'] = translator.translate(text, 'en')
        df.at[idx, 'fr_text'] = translator.translate(text, 'fr')
        df.at[idx, 'es_text'] = translator.translate(text, 'es')
        
        # Rate limiting
        time.sleep(0.1)
        
        # Save periodically
        if (idx + 1) % 50 == 0:
            df.to_csv(output_path, index=False)
            translator.save_cache()
    
    # Save final results
    df.to_csv(output_path, index=False)
    
    print(f"Translated metadata saved to: {output_path}")
    
    # Show sample
    print("\nSample translations:")
    sample = df.iloc[0]
    print(f"  VI: {sample['text']}")
    print(f"  EN: {sample['en_text']}")
    print(f"  FR: {sample['fr_text']}")
    print(f"  ES: {sample['es_text']}")


def main():
    parser = argparse.ArgumentParser(description='Translate ViSL texts to multiple languages')
    parser.add_argument('--dataset_root', type=str, 
                        default='/workspace/khanh/SpaMo/dataset_ViSL',
                        help='Root directory of ViSL dataset')
    parser.add_argument('--mode', nargs='+', type=str, 
                        default=['train', 'val', 'test'],
                        help='Dataset modes to process')
    parser.add_argument('--cache_file', type=str,
                        default='/workspace/khanh/SpaMo/dataset_ViSL/translation_cache.json',
                        help='Path to translation cache file')
    
    args = parser.parse_args()
    
    # Initialize translator with cache
    translator = TranslationService(cache_file=args.cache_file)
    
    for mode in args.mode:
        process_mode(args.dataset_root, mode, translator)
        translator.save_cache()
    
    print("\n" + "="*50)
    print("Translation completed!")
    print("="*50)


if __name__ == '__main__':
    main()






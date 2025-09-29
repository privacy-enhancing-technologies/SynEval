import json
from pathlib import Path

def clear_ner_cache():
    """Clear the NER analysis cache to force reprocessing."""
    cache_file = Path('./cache/amazon_fashion_ner_fast_results.json')
    
    if cache_file.exists():
        # Backup the old cache
        backup_file = cache_file.with_suffix('.json.backup')
        cache_file.rename(backup_file)
        print(f"Backed up old cache to {backup_file}")
        
        # Clear the cache by removing the file
        print("Cleared NER analysis cache")
    else:
        print("No NER cache file found")

if __name__ == "__main__":
    clear_ner_cache() 
import os
import requests
import concurrent.futures

# --- CONFIG ---
LINKS_DIR = r"D:\Yoga-82\Yoga-82\yoga_dataset_links" # Update this
OUTPUT_DIR = r"D:\Yoga-82\dataset_images"             # Raw download location
MAX_WORKERS = 20

def download_image(info):
    rel_path, url = info
    save_path = os.path.join(OUTPUT_DIR, rel_path.strip())
    
    if os.path.exists(save_path): return "Skipped"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url.strip(), headers=headers, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return "Success"
    except:
        pass
    return "Failed"

def main():
    if not os.path.exists(LINKS_DIR): return print("Links dir not found")
    
    tasks = []
    for fname in os.listdir(LINKS_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(LINKS_DIR, fname), 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split('\t')
                    if len(parts) >= 2: tasks.append((parts[0], parts[1]))

    print(f"Downloading {len(tasks)} images with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(download_image, tasks))
        
    print("Download Complete.")

if __name__ == "__main__":
    main()
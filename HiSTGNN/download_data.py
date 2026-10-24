import os
import urllib.request
import sys

def download_file(name, fid, dest_dir):
    dest_path = os.path.join(dest_dir, name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        print(f"[{name}] Already exists ({os.path.getsize(dest_path)} bytes). Skipping.")
        return
    url = f'https://drive.google.com/uc?export=download&id={fid}'
    print(f"[{name}] Downloading to {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"  {name}: {downloaded / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB ({percent:.1f}%)", end='\r')
    print(f"\n[{name}] Finished successfully ({os.path.getsize(dest_path)} bytes).")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bj_dir = os.path.join(base_dir, "data", "wfd_BJ")
    os.makedirs(bj_dir, exist_ok=True)

    bj_files = {
        'test.dict': '1t-APmktiKybLTVtEnsyTOeFWMOaOBXvs',
        'val.dict': '1Vc_q-vLkSpeQgbKv1CQf0bXabgvl7iDe',
        'train.dict': '1Vz1QA71V-yzPYdrtOLnkkWR4vpQBWbsl',
    }

    print("=== Downloading WD_BJ Dataset for Table 3 ===")
    for name, fid in bj_files.items():
        download_file(name, fid, bj_dir)
    print("\nAll WD_BJ files ready in:", bj_dir)

if __name__ == '__main__':
    main()

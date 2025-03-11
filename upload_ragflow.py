import os
import glob
import json
import requests

API_KEY = 'ragflow-g3NTNlMjE2ZWU2NTExZWZhMmY0ZmU1OG'
HOST = 'http://192.168.10.51:30010'
DATASET_ID = '7e489454ed9e11ef9d56fe58eb407195'
STATUS_FILE = 'upload_status.json'

def load_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f).get('processed_count', 0)
    except FileNotFoundError:
        return 0

def save_status(count):
    with open(STATUS_FILE, 'w') as f:
        json.dump({'processed_count': count}, f)

def main():
    processed_count = load_status()
    pdf_files = sorted(glob.glob('*.pdf'))
    total = len(pdf_files)
    
    if processed_count >= total:
        print("All files processed.")
        return

    headers = {'Authorization': f'Bearer {API_KEY}'}
    upload_url = f"{HOST}/api/v1/datasets/{DATASET_ID}/documents"
    parse_url = f"{HOST}/api/v1/datasets/{DATASET_ID}/chunks"

    for i in range(processed_count, total):
        file = pdf_files[i]
        
        # Upload
        try:
            with open(file, 'rb') as f:
                resp = requests.post(upload_url, headers=headers, files={'file': f})
            
            if resp.status_code != 200 or resp.json()['code'] != 0:
                print(f"Upload failed: {resp.text}")
                continue
                
            doc_id = resp.json()['data'][0]['id']
            
            # Parse
            parse_resp = requests.post(parse_url, headers=headers, json={'document_ids': [doc_id]})
            if parse_resp.status_code != 200 or parse_resp.json()['code'] != 0:
                print(f"Parse failed: {parse_resp.text}")
                continue
                
            processed_count += 1
            save_status(processed_count)
            print(f"Processed {file} ({processed_count}/{total})")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

if __name__ == '__main__':
    main()
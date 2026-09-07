import os
import sys
import io
import re
import csv
import time
import requests
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Enforce UTF-8 output encoding for Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================================================
# Configuration
# ============================================================
KB_FOLDER = "Knowledge_Base"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3-embedding:8b"
COLLECTION_NAME = "agriculture_disease_3"
EMBEDDING_DIM = 4096
BATCH_SIZE = 3

def normalize_text(text):
    """
    Normalizes crop and disease names to eliminate formatting differences.
    Extracts English name from parentheses if present.
    """
    if not text or pd.isna(text):
        return ""
    match = re.search(r'\(([a-zA-Z\s/]+)\)', str(text))
    if match:
        text = match.group(1).split('/')[0].strip()
    else:
        text = re.sub(r'\(.*?\)', '', str(text))
    t = text.strip().lower()
    t = re.sub(r'[^a-zA-Z0-9\u0B80-\u0BFF\s]', '', t)
    return ' '.join(t.split())

def get_batch_embeddings(texts):
    """
    Get embeddings in small batches from Ollama with 12 CPU threads.
    """
    url = f"{OLLAMA_URL}/api/embed"
    payload = {
        "model": MODEL_NAME,
        "input": texts,
        "keep_alive": "30m",
        "options": {
            "num_thread": 12
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Retrying batch embed (attempt {attempt + 1}/{max_retries}): {e}...", flush=True)
            time.sleep(5)

def load_and_deduplicate_all_records():
    """
    Load all CSV and Excel datasets from Knowledge_Base and its subdirectories,
    deduplicate by (crop, category, disease), and merge best information.
    """
    if not os.path.isdir(KB_FOLDER):
        raise FileNotFoundError(f"Knowledge Base folder '{KB_FOLDER}' not found!")

    raw_records = []

    # --------------------------------------------------------
    # 1. Load disease.csv
    # --------------------------------------------------------
    csv_path = os.path.join(KB_FOLDER, "disease.csv")
    if os.path.exists(csv_path):
        print(f"Loading dataset: {csv_path}", flush=True)
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count_csv = 0
            for row in reader:
                crop = row.get("crop", "").strip()
                disease = row.get("disease", "").strip()
                cause = row.get("cause", "").strip()
                symptoms = row.get("symptoms", "").strip()
                solution = row.get("solution", "").strip()

                if not crop and not disease:
                    continue

                raw_records.append({
                    "crop": crop,
                    "category": "Disease",
                    "disease": disease,
                    "cause": cause if cause else "Pathogen infection",
                    "symptoms": symptoms if symptoms else f"Disease symptoms on {crop}",
                    "solution": solution,
                    "source": "disease.csv",
                    "image_folder": ""
                })
                count_csv += 1
        print(f"  Loaded {count_csv} raw records from disease.csv", flush=True)

    # --------------------------------------------------------
    # 2. Find and load all unique Excel files (.xlsx)
    # --------------------------------------------------------
    excel_files = set()
    for root, dirs, files in os.walk(KB_FOLDER):
        for f in files:
            if f.endswith(".xlsx") and not f.startswith("~$"):
                excel_files.add(os.path.join(root, f))

    for ef in sorted(list(excel_files)):
        rel_ef = os.path.relpath(ef, KB_FOLDER)
        print(f"Loading Excel dataset: {rel_ef}", flush=True)
        try:
            xl = pd.ExcelFile(ef)
            sheet_names = xl.sheet_names

            if "Crops" in sheet_names and ("Diseases" in sheet_names or "Pests" in sheet_names):
                df_crops = xl.parse("Crops")
                crop_map = {}
                for _, r in df_crops.iterrows():
                    cid = str(r.get("Crop_ID", "")).strip()
                    cname = str(r.get("Crop_Name", "")).strip()
                    tname = str(r.get("Tamil_Name", "")).strip()
                    if pd.isna(cname) or cname == "nan":
                        cname = ""
                    if pd.isna(tname) or tname == "nan":
                        tname = ""
                    display_name = f"{cname} ({tname})" if tname else cname
                    crop_map[cid] = display_name if display_name else cid

                # Diseases Sheet
                if "Diseases" in sheet_names:
                    df_d = xl.parse("Diseases").dropna(subset=["Disease_Name"])
                    count_d = 0
                    for _, r in df_d.iterrows():
                        cid = str(r.get("Crop_ID", "")).strip()
                        cname = crop_map.get(cid, cid)
                        dname = str(r.get("Disease_Name", "")).strip()
                        ctrl = str(r.get("Control_Message", "")).strip()
                        if pd.isna(ctrl) or ctrl == "nan":
                            ctrl = ""
                        src = str(r.get("Source", "")).strip()
                        if pd.isna(src) or src == "nan":
                            src = rel_ef
                        img_f = str(r.get("Image_Folder", "")).strip()
                        if pd.isna(img_f) or img_f == "nan":
                            img_f = ""

                        raw_records.append({
                            "crop": cname,
                            "category": "Disease",
                            "disease": dname,
                            "cause": "Pathogen infection",
                            "symptoms": f"Disease symptoms on {cname}",
                            "solution": ctrl,
                            "source": src,
                            "image_folder": img_f
                        })
                        count_d += 1
                    print(f"  Loaded {count_d} disease records from {rel_ef}", flush=True)

                # Pests Sheet
                if "Pests" in sheet_names:
                    df_p = xl.parse("Pests").dropna(subset=["Pest_Name"])
                    count_p = 0
                    for _, r in df_p.iterrows():
                        cid = str(r.get("Crop_ID", "")).strip()
                        cname = crop_map.get(cid, cid)
                        pname = str(r.get("Pest_Name", "")).strip()
                        ctrl = str(r.get("Control_Message", "")).strip()
                        if pd.isna(ctrl) or ctrl == "nan":
                            ctrl = ""
                        sym = str(r.get("Symptoms_or_Damage", "")).strip()
                        if pd.isna(sym) or sym == "nan":
                            sym = f"Pest damage observed on {cname}"
                        src = str(r.get("Source", "")).strip()
                        if pd.isna(src) or src == "nan":
                            src = rel_ef
                        img_f = str(r.get("Image_Folder", "")).strip()
                        if pd.isna(img_f) or img_f == "nan":
                            img_f = ""

                        raw_records.append({
                            "crop": cname,
                            "category": "Pest",
                            "disease": pname,
                            "cause": "Insect pest infestation",
                            "symptoms": sym,
                            "solution": ctrl,
                            "source": src,
                            "image_folder": img_f
                        })
                        count_p += 1
                    print(f"  Loaded {count_p} pest records from {rel_ef}", flush=True)

        except Exception as e:
            print(f"  Warning: Failed to load {ef}: {e}", flush=True)

    print(f"\nTotal Raw Records Gathered: {len(raw_records)}", flush=True)

    # --------------------------------------------------------
    # 3. Deduplicate & Merge
    # --------------------------------------------------------
    deduped = {}
    duplicates_count = 0

    for r in raw_records:
        norm_crop = normalize_text(r["crop"])
        norm_cat = r["category"].lower()
        norm_dis = normalize_text(r["disease"])
        key = f"{norm_crop}||{norm_cat}||{norm_dis}"

        if key in deduped:
            duplicates_count += 1
            existing = deduped[key]
            # Keep longer, more descriptive solution
            if len(r["solution"]) > len(existing["solution"]):
                existing["solution"] = r["solution"]
            # Keep more detailed symptoms
            if len(r["symptoms"]) > len(existing["symptoms"]):
                existing["symptoms"] = r["symptoms"]
            # Keep image folder reference
            if r["image_folder"] and not existing["image_folder"]:
                existing["image_folder"] = r["image_folder"]
            # Append source
            if r["source"] not in existing["source"]:
                existing["source"] += f", {r['source']}"
        else:
            deduped[key] = r

    clean_records = list(deduped.values())
    print(f"Duplicates Removed / Merged: {duplicates_count}", flush=True)
    print(f"Final Clean Unique Records: {len(clean_records)}", flush=True)

    # --------------------------------------------------------
    # 4. Prepare Embed Text for Vector Search
    # --------------------------------------------------------
    unified_records = []
    for r in clean_records:
        text_to_embed = (
            f"Crop: {r['crop']}\n"
            f"Category: {r['category']}\n"
            f"Disease/Pest: {r['disease']}\n"
            f"Cause: {r['cause']}\n"
            f"Symptoms: {r['symptoms']}\n"
            f"Solution: {r['solution']}"
        )
        unified_records.append({
            "text_to_embed": text_to_embed,
            "payload": r
        })

    return unified_records

def main():
    print("=" * 60)
    print(" Robust Multi-Dataset Ingestion & Deduplication")
    print("=" * 60)

    records = load_and_deduplicate_all_records()
    total_records = len(records)

    print(f"\nConnecting to Qdrant at {QDRANT_URL}...", flush=True)
    qdrant_client = QdrantClient(url=QDRANT_URL)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})...", flush=True)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' exists.", flush=True)

    current_count = qdrant_client.count(COLLECTION_NAME).count
    print(f"Currently in collection: {current_count}/{total_records} points", flush=True)

    num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nProcessing {total_records} records in {num_batches} batches of {BATCH_SIZE}...", flush=True)

    t_start = time.time()
    for b_idx in range(num_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, total_records)

        # Skip batch if already completely ingested
        if b_end <= current_count:
            continue

        batch_slice = records[b_start:b_end]

        batch_texts = [r["text_to_embed"] for r in batch_slice]
        batch_payloads = [r["payload"] for r in batch_slice]

        t_batch_start = time.time()
        batch_vectors = get_batch_embeddings(batch_texts)

        points = []
        for i, (vec, payload) in enumerate(zip(batch_vectors, batch_payloads)):
            point_id = b_start + i + 1
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                )
            )

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        t_batch_elapsed = time.time() - t_batch_start
        print(f" [Batch {b_idx + 1}/{num_batches}] Upserted {b_end}/{total_records} points into Qdrant ({t_batch_elapsed:.1f}s)", flush=True)

    t_total = time.time() - t_start
    print("=" * 60)
    print(f" Successfully ingested all {total_records} deduplicated records in {t_total:.1f}s!")
    print("=" * 60)

if __name__ == "__main__":
    main()

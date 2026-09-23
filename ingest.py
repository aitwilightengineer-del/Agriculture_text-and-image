import os
import sys
import io
import re
import csv
import time
import json
import base64
import requests
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Enforce UTF-8 output encoding for Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================================================
# Configuration
# ============================================================
KB_FOLDER = "Agriculture_Dataset"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3-embedding:8b"
VL_MODEL_NAME = "qwen3-vl:4b"
COLLECTION_NAME = "agriculture_disease_demo"
EMBEDDING_DIM = 4096
BATCH_SIZE = 3

CACHE_FILE = "visual_symptoms_cache.json"
VISUAL_SYMPTOMS_CACHE = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw_c = json.load(f)
            VISUAL_SYMPTOMS_CACHE = {k: tuple(v) for k, v in raw_c.items()}
    except Exception:
        VISUAL_SYMPTOMS_CACHE = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(VISUAL_SYMPTOMS_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_image_visual_symptoms(image_folder_path, crop_name="", disease_name=""):
    """
    Analyzes sample image from image_folder_path using Qwen3-VL-4B model
    and extracts concise visual symptom descriptions.
    """
    if not image_folder_path:
        return "", ""

    norm_path = image_folder_path.replace("/", "\\").strip()
    cache_key = f"{norm_path}||{crop_name}||{disease_name}"
    if cache_key in VISUAL_SYMPTOMS_CACHE:
        return VISUAL_SYMPTOMS_CACHE[cache_key]

    # Resolve path relative to current folder or KB_FOLDER
    target_dir = norm_path
    if not os.path.exists(target_dir):
        target_dir = os.path.join(KB_FOLDER, norm_path)

    if not os.path.exists(target_dir):
        VISUAL_SYMPTOMS_CACHE[cache_key] = ("", "")
        save_cache()
        return "", ""

    try:
        files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
        if not files:
            VISUAL_SYMPTOMS_CACHE[cache_key] = ("", "")
            save_cache()
            return "", ""

        sample_img_path = os.path.join(target_dir, files[0])
        with open(sample_img_path, 'rb') as f:
            b64_str = base64.b64encode(f.read()).decode('utf-8')

        v_prompt = "Describe the visual symptoms and appearance of this crop disease/pest leaf/fruit image in 1-2 concise sentences."
        if crop_name and disease_name:
            v_prompt = f"This image shows {crop_name} affected by {disease_name}. Describe its visual appearance and symptoms concisely in 1-2 sentences without mentioning other disease names."

        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": VL_MODEL_NAME,
            "prompt": v_prompt,
            "images": [b64_str],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            analysis = resp.json().get("response", "").strip()
            rel_sample_path = os.path.relpath(sample_img_path, ".").replace("\\", "/")
            res_val = (analysis, rel_sample_path)
            VISUAL_SYMPTOMS_CACHE[cache_key] = res_val
            save_cache()
            print(f"    [Vision Analysis] Analyzed sample image for {norm_path} ({disease_name})", flush=True)
            return res_val
    except Exception as e:
        print(f"    Notice: Vision analysis skipped for {norm_path}: {e}", flush=True)

    VISUAL_SYMPTOMS_CACHE[cache_key] = ("", "")
    save_cache()
    return "", ""

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
    Load all CSV and Excel datasets from current folder and Knowledge_Base,
    deduplicate by (crop, category, disease), and merge best information.
    """
    raw_records = []

    # --------------------------------------------------------
    # 1. Load disease.csv if present
    # --------------------------------------------------------
    csv_paths = [os.path.join(KB_FOLDER, "disease.csv"), "disease.csv"]
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f"Loading dataset: {csv_path}", flush=True)
            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                count_csv = 0
                for row in reader:
                    crop = row.get("crop", "").strip()
                    disease = row.get("disease", "").strip()
                    ctrl = row.get("control_message", "").strip() or row.get("solution", "").strip()

                    if not crop and not disease:
                        continue

                    raw_records.append({
                        "crop": crop,
                        "category": "Disease",
                        "disease": disease,
                        "control_message": ctrl,
                        "solution": ctrl,
                        "source": os.path.basename(csv_path),
                        "image_folder": "",
                        "sample_image_path": "",
                        "visual_symptoms": ""
                    })
                    count_csv += 1
            print(f"  Loaded {count_csv} raw records from {csv_path}", flush=True)
            break

    # --------------------------------------------------------
    # 2. Find and load all unique Excel files (.xlsx)
    # --------------------------------------------------------
    excel_files = set()
    search_dirs = [KB_FOLDER, "."]
    for s_dir in search_dirs:
        if os.path.isdir(s_dir):
            for root, dirs, files in os.walk(s_dir):
                for f in files:
                    if f.endswith(".xlsx") and not f.startswith("~$"):
                        excel_files.add(os.path.abspath(os.path.join(root, f)))

    for ef in sorted(list(excel_files)):
        rel_ef = os.path.basename(ef)
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

                # Helper to resolve crop name with fallback logic
                def get_resolved_crop_name(cid_val, img_f_val):
                    if cid_val in crop_map and crop_map[cid_val]:
                        return crop_map[cid_val]
                    # Fallback to extracting crop name from Image_Folder if available
                    if img_f_val:
                        parts = [p for p in re.split(r'[\\/]', img_f_val) if p]
                        for part in parts:
                            if part.lower() not in ["agriculture_dataset", "images", "diseases", "pests"]:
                                return part.replace("_", " ").title()
                    return cid_val

                # Diseases Sheet
                if "Diseases" in sheet_names:
                    df_d = xl.parse("Diseases").dropna(subset=["Disease_Name"])
                    count_d = 0
                    for _, r in df_d.iterrows():
                        cid = str(r.get("Crop_ID", "")).strip()
                        img_f = str(r.get("Image_Folder", "")).strip()
                        if pd.isna(img_f) or img_f == "nan":
                            img_f = ""

                        cname = get_resolved_crop_name(cid, img_f)
                        dname = str(r.get("Disease_Name", "")).strip()
                        ctrl = str(r.get("Control_Message", "")).strip()
                        if pd.isna(ctrl) or ctrl == "nan":
                            ctrl = ""
                        src = str(r.get("Source", "")).strip()
                        if pd.isna(src) or src == "nan":
                            src = rel_ef

                        vis_symptoms, sample_img_path = get_image_visual_symptoms(img_f, cname, dname)

                        raw_records.append({
                            "crop": cname,
                            "category": "Disease",
                            "disease": dname,
                            "control_message": ctrl,
                            "solution": ctrl,
                            "source": src,
                            "image_folder": img_f,
                            "sample_image_path": sample_img_path,
                            "visual_symptoms": vis_symptoms
                        })
                        count_d += 1
                    print(f"  Loaded {count_d} disease records from {rel_ef}", flush=True)

                # Pests Sheet
                if "Pests" in sheet_names:
                    df_p = xl.parse("Pests").dropna(subset=["Pest_Name"])
                    count_p = 0
                    for _, r in df_p.iterrows():
                        cid = str(r.get("Crop_ID", "")).strip()
                        img_f = str(r.get("Image_Folder", "")).strip()
                        if pd.isna(img_f) or img_f == "nan":
                            img_f = ""

                        cname = get_resolved_crop_name(cid, img_f)
                        pname = str(r.get("Pest_Name", "")).strip()
                        ctrl = str(r.get("Control_Message", "")).strip()
                        if pd.isna(ctrl) or ctrl == "nan":
                            ctrl = ""
                        src = str(r.get("Source", "")).strip()
                        if pd.isna(src) or src == "nan":
                            src = rel_ef

                        vis_symptoms, sample_img_path = get_image_visual_symptoms(img_f, cname, pname)

                        raw_records.append({
                            "crop": cname,
                            "category": "Pest",
                            "disease": pname,
                            "control_message": ctrl,
                            "solution": ctrl,
                            "source": src,
                            "image_folder": img_f,
                            "sample_image_path": sample_img_path,
                            "visual_symptoms": vis_symptoms
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
            # Keep longer, more descriptive control_message
            if len(r["control_message"]) > len(existing["control_message"]):
                existing["control_message"] = r["control_message"]
                existing["solution"] = r["control_message"]
            # Keep image folder reference and visual symptoms
            if r["image_folder"] and not existing["image_folder"]:
                existing["image_folder"] = r["image_folder"]
                existing["sample_image_path"] = r["sample_image_path"]
                existing["visual_symptoms"] = r["visual_symptoms"]
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
        )
        if r.get("visual_symptoms"):
            text_to_embed += f"Visual Appearance / Symptoms: {r['visual_symptoms']}\n"
        text_to_embed += f"Control Message: {r['control_message']}"

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

    if total_records == 0:
        print("\nERROR: No records loaded - aborting so the existing Qdrant collection is not wiped.", flush=True)
        print("Check the warnings above (e.g. missing packages like openpyxl, or wrong dataset paths).", flush=True)
        return

    print(f"\nConnecting to Qdrant at {QDRANT_URL}...", flush=True)
    qdrant_client = QdrantClient(url=QDRANT_URL)

    if qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Re-creating Qdrant collection '{COLLECTION_NAME}' for fresh schema...", flush=True)
        qdrant_client.delete_collection(COLLECTION_NAME)

    print(f"Creating Qdrant collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})...", flush=True)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

    num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nProcessing {total_records} records in {num_batches} batches of {BATCH_SIZE}...", flush=True)

    t_start = time.time()
    for b_idx in range(num_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, total_records)
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

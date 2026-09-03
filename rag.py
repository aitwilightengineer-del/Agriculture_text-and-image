import sys
import io
import time
import re
import base64
import requests
import torch
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# ============================================================
# Enforce UTF-8 output encoding for Windows consoles
# ============================================================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================================================
# Configuration
# ============================================================
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

EMBEDDING_MODEL_NAME = "qwen3-embedding:8b"
LLM_MODEL_NAME = "qwen3:8b"
VISION_MODEL_NAME = "qwen2.5vl:7b"

COLLECTION_NAME = "agriculture_disease2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Minimum relevance score threshold to filter out irrelevant/unrelated queries
RELEVANCE_THRESHOLD = 0.20

# Number of CPU threads to maximize speed
NUM_CPU_THREADS = 12

# ============================================================
# Initialize Flask App
# ============================================================
app = Flask(__name__)

# ============================================================
# Initialize Qdrant Client
# ============================================================
print(f"Connecting to Qdrant at {QDRANT_URL}...", flush=True)
qdrant_client = QdrantClient(url=QDRANT_URL)

# ============================================================
# Setup device for reranker model
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading reranker model ({RERANKER_MODEL_NAME}) on {device}...", flush=True)

reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=device,
    trust_remote_code=True
)
print("Reranker model loaded successfully.", flush=True)

# ============================================================
# Helper: Detect Language of Question
# ============================================================
def detect_language(text):
    """
    Detect if text contains Tamil characters or English.
    """
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "Tamil"
    return "English"

# ============================================================
# Generate Query Embedding (Fast & Direct)
# ============================================================
def get_query_embedding(query_text):
    """
    Generate query embedding using Ollama's qwen3-embedding:8b model.
    """
    url = f"{OLLAMA_URL}/api/embed"
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": query_text,
        "keep_alive": "30m"
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["embeddings"][0]

# ============================================================
# Retrieve and Rerank (Optimized 4-Candidate Fast Pool)
# ============================================================
def retrieve_and_rerank(query_text, top_k_retrieve=4, top_n_final=2):
    """
    Directly embed user query, search top 4 in Qdrant (<50ms),
    and fast-rerank in ~2s on CPU.
    """
    t_start = time.time()

    # 1. Direct query embedding
    query_vector = get_query_embedding(query_text)
    t_embed = time.time()

    # 2. Qdrant vector retrieval
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k_retrieve
    )
    search_results = response.points
    t_qdrant = time.time()

    if not search_results:
        return [], 0.0

    # 3. Prepare pairs for reranking
    pairs = []
    for hit in search_results:
        payload = hit.payload
        context_doc = (
            f"Crop: {payload.get('crop', '')}\n"
            f"Category: {payload.get('category', 'Disease')}\n"
            f"Disease: {payload.get('disease', '')}\n"
            f"Symptoms: {payload.get('symptoms', '')}\n"
            f"Solution: {payload.get('solution', '')}"
        )
        pairs.append((query_text, context_doc))

    # 4. Predict relevance scores with cross-encoder
    scores = reranker.predict(pairs)
    t_rerank = time.time()

    # 5. Sort and filter by relevance threshold
    scored_hits = sorted(zip(scores, search_results), key=lambda x: x[0], reverse=True)
    valid_hits = [(s, h) for s, h in scored_hits if float(s) >= RELEVANCE_THRESHOLD]

    top_score = float(scored_hits[0][0]) if scored_hits else 0.0
    print(f"[Timing] Embed: {t_embed - t_start:.2f}s | Qdrant: {(t_qdrant - t_embed)*1000:.1f}ms | Rerank: {t_rerank - t_qdrant:.2f}s | Top Score: {top_score:.4f}", flush=True)

    if not valid_hits:
        return [], top_score

    # 6. Format top N valid results
    final_results = []
    for score, hit in valid_hits[:top_n_final]:
        payload = hit.payload
        final_results.append({
            "id": hit.id,
            "score": float(score),
            "crop": payload.get("crop"),
            "category": payload.get("category", "Disease"),
            "disease": payload.get("disease"),
            "cause": payload.get("cause"),
            "symptoms": payload.get("symptoms"),
            "solution": payload.get("solution"),
            "image_folder": payload.get("image_folder", ""),
            "source": payload.get("source", "")
        })

    return final_results, top_score

# ============================================================
# Vision Model: Analyze Image with Qwen2.5-VL
# ============================================================
def analyze_image_with_vision(image_b64, custom_question=None):
    """
    Pass image to Ollama Vision model (qwen2.5vl:7b) to detect crop, symptoms, and disease/pest.
    """
    prompt = """Analyze this agricultural crop/leaf/pest image carefully.
Extract:
1. Crop name (e.g. Paddy, Cotton, Tomato, Chili, Maize, Apple, etc.)
2. Suspected Disease or Pest name (e.g. Brown spot, Stem borer, Bacterial blight, etc.)
3. Key observable symptoms (leaf spots, lesions, discoloration, pest damage).

Be concise and precise. Format:
Crop: [Crop name]
Disease/Pest: [Name]
Symptoms: [Observed signs]"""

    if custom_question:
        prompt += f"\n\nFarmer Question: {custom_question}"

    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": VISION_MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_thread": NUM_CPU_THREADS
        }
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("response", "").strip()

# ============================================================
# Query LLM (Multi-threaded for Speed)
# ============================================================
def query_llm(prompt, model_name=None):
    """
    Generate final answer via Ollama with 12 CPU threads.
    Supports model_name to reuse already-loaded models (zero RAM swapping).
    """
    target_model = model_name if model_name else LLM_MODEL_NAME
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_thread": NUM_CPU_THREADS
        }
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    raw_answer = response.json().get("response", "").strip()
    clean_answer = re.sub(r'^(Direct Answer|Answer)\s*:\s*', '', raw_answer, flags=re.IGNORECASE).strip()
    return clean_answer

# ============================================================
# Endpoint 1: Text Query (/query)
# ============================================================
@app.route("/query", methods=["POST"])
def query_endpoint():
    req_start = time.time()
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question field cannot be empty."}), 400

    try:
        target_lang = detect_language(question)

        # 1. Fast Retrieve & Rerank (~4s total)
        contexts, top_score = retrieve_and_rerank(question)

        # 2. If no relevant context passes threshold, short-circuit immediately
        if not contexts:
            elapsed = time.time() - req_start
            fallback_msg = (
                "The requested information is not available in the database."
                if target_lang == "English"
                else "கோரப்பட்ட தகவல் தரவுத்தளத்தில் கிடைக்கவில்லை."
            )
            print(f"[Request Done] Fast Fallback returned in {elapsed:.2f}s", flush=True)
            return jsonify({
                "answer": fallback_msg,
                "contexts": [],
                "time_seconds": round(elapsed, 2)
            })

        # 3. Format retrieved context
        context_blocks = []
        for p in contexts:
            block = (
                f"Crop: {p['crop']}\n"
                f"Category: {p.get('category', 'Disease')}\n"
                f"Disease/Pest: {p['disease']}\n"
                f"Symptoms: {p['symptoms']}\n"
                f"Solution: {p['solution']}"
            )
            context_blocks.append(block)

        context_str = "\n\n---\n\n".join(context_blocks)

        # 4. Strict, Language-Enforced Prompt
        prompt = f"""Context:
{context_str}

User Question: {question}

MANDATORY INSTRUCTIONS:
1. The user's question is asked in {target_lang}.
2. You MUST answer 100% in {target_lang}.
3. If the context above is in a different language (e.g., Tamil or English), TRANSLATE the solution into natural {target_lang}.
4. Provide ONLY the direct factual answer in 1-3 concise sentences based on the context above.
5. If the information is not in the context, reply with: The requested information is not available in the database.

Answer in {target_lang}:"""

        # 5. Fast Multi-threaded LLM generation
        t_llm_start = time.time()
        answer = query_llm(prompt)
        t_llm_end = time.time()

        if "not available in the database" in answer.lower():
            contexts = []

        total_elapsed = time.time() - req_start
        print(f"[Request Done] Language: {target_lang} | LLM Gen: {t_llm_end - t_llm_start:.2f}s | Total: {total_elapsed:.2f}s", flush=True)

        return jsonify({
            "answer": answer,
            "contexts": contexts,
            "time_seconds": round(total_elapsed, 2)
        })

    except Exception as e:
        print(f"Error processing request: {e}", flush=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# ============================================================
# Endpoint 2: Image Upload & Visual Diagnosis (/diagnose-image)
# ============================================================
@app.route("/diagnose-image", methods=["POST"])
def diagnose_image_endpoint():
    """
    Accepts:
      - multipart/form-data with file field 'image' and optional 'question'
      - OR JSON body with 'image' (base64 string) and optional 'question'
    """
    req_start = time.time()
    image_b64 = None
    question = ""

    # Check multipart form-data
    if 'image' in request.files:
        image_file = request.files['image']
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        question = request.form.get('question', '').strip()

    # Check JSON base64
    elif request.is_json:
        data = request.get_json()
        image_b64 = data.get('image', '').strip()
        question = data.get('question', '').strip()

    if not image_b64:
        return jsonify({"error": "No image provided. Please upload an image file or provide a base64 image string."}), 400

    target_lang = detect_language(question) if question else "English"

    try:
        # Step 1: Visual Inspection with Vision Model (Qwen2.5-VL)
        print("\n[Vision Analysis] Sending image to Qwen2.5-VL model...", flush=True)
        t_vision_start = time.time()
        visual_diagnosis = analyze_image_with_vision(image_b64, custom_question=question)
        t_vision_end = time.time()
        print(f"[Vision Analysis Done in {t_vision_end - t_vision_start:.2f}s]:\n{visual_diagnosis}\n", flush=True)

        # Step 2: Vector Search in Qdrant with the Visual Diagnosis
        search_query = f"{question} {visual_diagnosis}".strip()
        contexts, top_score = retrieve_and_rerank(search_query)

        if not contexts:
            elapsed = time.time() - req_start
            fallback_msg = "No matching treatment found in the knowledge base." if target_lang == "English" else "தரவுத்தளத்தில் பொருந்தக்கூடிய சிகிச்சை கிடைக்கவில்லை."
            return jsonify({
                "answer": fallback_msg,
                "contexts": [],
                "visual_diagnosis": visual_diagnosis,
                "treatment_solution": fallback_msg,
                "matched_records": [],
                "time_seconds": round(elapsed, 2)
            })

        # Step 3: Format Context and Generate Advisory
        context_blocks = []
        for p in contexts:
            block = (
                f"Crop: {p['crop']}\n"
                f"Category: {p.get('category', 'Disease')}\n"
                f"Disease/Pest: {p['disease']}\n"
                f"Symptoms: {p['symptoms']}\n"
                f"Solution: {p['solution']}"
            )
            context_blocks.append(block)

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""Context from Verified Agricultural Knowledge Base:
{context_str}

Visual Image Diagnosis:
{visual_diagnosis}

User Question: {question if question else 'Provide the remedy for the identified issue.'}

INSTRUCTIONS:
1. Provide a direct, actionable treatment solution for the diagnosed problem.
2. Answer 100% in {target_lang}.
3. Use ONLY the verified solutions from the context above.

Treatment Solution in {target_lang}:"""

        t_llm_start = time.time()
        treatment_solution = query_llm(prompt, model_name=VISION_MODEL_NAME)
        t_llm_end = time.time()

        total_elapsed = time.time() - req_start
        print(f"[Diagnose Done] Total Request Time: {total_elapsed:.2f}s", flush=True)

        return jsonify({
            "answer": treatment_solution,
            "contexts": contexts,
            "visual_diagnosis": visual_diagnosis,
            "treatment_solution": treatment_solution,
            "matched_records": contexts,
            "time_seconds": round(total_elapsed, 2)
        })

    except Exception as e:
        print(f"Error in image diagnosis: {e}", flush=True)
        return jsonify({"error": f"Image diagnosis failed: {str(e)}"}), 500

# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    print("Starting Multimodal Vision + Text RAG Server on port 5000...", flush=True)
    app.run(host="0.0.0.0", port=5002)

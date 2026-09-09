import sys
import io
import os
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
VL_MODEL_NAME = os.getenv("VL_MODEL_NAME", "qwen3-vl:2b")

COLLECTION_NAME = "agriculture_disease_demo"

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_IMAGE_PROMPT = (
    "Using the provided image, find the crop present in image, check if any disease is present to the crop. "
    "If present show the name of crop, disease, and solution to cure that. "
    "If disease not present, find the crop present in image and show answer for that."
)



# ============================================================
# Initialize Flask App
# ============================================================
app = Flask(__name__)


# ============================================================
# Initialize Qdrant Client
# ============================================================
print(f"Connecting to Qdrant at {QDRANT_URL}...", flush=True)

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# ============================================================
# Setup device for reranker model
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    f"Loading reranker model ({RERANKER_MODEL_NAME}) on {device}...",
    flush=True
)

reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=device,
    trust_remote_code=True
)

print("Reranker model loaded successfully.", flush=True)


# ============================================================
# Image Analysis using Qwen3-VL-4B / Vision LLM
# ============================================================
def analyze_image(image_b64, prompt_text=""):
    """
    Analyze the input image using Qwen3-VL-4B (or available vision LLM in Ollama).
    Extract crop/plant name, symptoms, and potential disease.
    """
    vision_prompt = (
        "You are an expert agricultural plant disease diagnostic assistant.\n\n"
        "Carefully analyze the provided image of a crop/plant leaf/fruit and identify:\n"
        "1. Crop / Plant species (e.g., Apple, Tomato, Potato, Rice, Wheat, Grape, etc.).\n"
        "2. Specific symptoms, lesions, spots, rot, leaf curling, discoloration, or pest damage.\n"
        "3. Specific disease or pest name (e.g., Apple Scab, Black Rot, Early Blight, Powdery Mildew, etc.) or state 'Healthy / No Disease' if no disease is present.\n\n"
    )

    if prompt_text:
        vision_prompt += f"User Prompt / Context: {prompt_text}\n\n"

    vision_prompt += (
        "Output a concise summary listing:\n"
        "- Crop Name: [Identified Crop]\n"
        "- Disease/Pest: [Identified Disease/Pest or Healthy / No Disease]\n"
        "- Symptoms: [Observed Symptoms or None]\n"
    )

    url = f"{OLLAMA_URL}/api/generate"

    # Try requested model first, with fallbacks to other locally available vision models if needed
    models_to_try = [VL_MODEL_NAME, "qwen3-vl:2b", "Qwen3-VL-2B", "qwen3-vl:4b", "Qwen3-VL-4B"]

    last_error = None
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "prompt": vision_prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        try:
            print(f"Sending image to Vision Model ({model_name})...", flush=True)
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("response", "").strip()
                if analysis:
                    print(f"Image vision analysis completed using {model_name}.", flush=True)
                    return analysis
            else:
                err_msg = response.json().get("error", response.text)
                print(f"Vision model '{model_name}' error: {err_msg}", flush=True)
                last_error = err_msg
        except Exception as e:
            print(f"Exception calling vision model '{model_name}': {e}", flush=True)
            last_error = str(e)

    raise RuntimeError(
        f"Failed to analyze image with vision model ({VL_MODEL_NAME}): {last_error}"
    )



# ============================================================
# Generate Query Embedding
# ============================================================
def get_query_embedding(query_text):
    """
    Generate query embedding using Ollama's
    qwen3-embedding:8b model.
    """

    url = f"{OLLAMA_URL}/api/embed"

    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": query_text
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# ============================================================
# Expand Query
# ============================================================
def expand_query(query_text):
    """
    Expand the user query into both English and Tamil
    to maximize cross-lingual retrieval accuracy.
    """

    prompt = f"""
You are a bilingual agricultural search assistant.

Translate and expand the following user query into both English and Tamil.

Extract:
- Crop name
- Disease name

Provide both English and Tamil terms.

Output ONLY a single line in exactly this format:

Expanded Query: [Tamil Query] / [English Query] | Crop: [Tamil Crop] ([English Crop]) | Disease: [Tamil Disease] ([English Disease])

Do not provide explanations.
Do not provide additional text.
Do not provide multiple lines.

User Query:
{query_text}
"""

    try:

        url = f"{OLLAMA_URL}/api/generate"

        payload = {
            "model": LLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }

        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        expanded = response.json()["response"].strip()

        if "Expanded Query:" in expanded:
            return expanded

        return f"Expanded Query: {query_text} | {expanded}"

    except Exception as e:

        print(
            f"Error expanding query: {e}",
            flush=True
        )

        return query_text


# ============================================================
# Retrieve and Rerank
# ============================================================
def retrieve_and_rerank(
    query_text,
    top_k_retrieve=15,
    top_n_final=3,
    skip_expansion=False
):
    """
    Retrieve candidate matches from Qdrant,
    then rerank them using the cross-encoder.
    """

    # --------------------------------------------------------
    # 1. Expand query (skip if already structured image query)
    # --------------------------------------------------------
    if skip_expansion or query_text.startswith("Image Analysis:"):
        expanded_query = query_text
    else:
        expanded_query = expand_query(query_text)

    print(
        f"Original Query: '{query_text}'",
        flush=True
    )

    print(
        f"Expanded Query: '{expanded_query}'",
        flush=True
    )

    # --------------------------------------------------------
    # 2. Generate query embedding
    # --------------------------------------------------------
    query_vector = get_query_embedding(
        expanded_query
    )

    # --------------------------------------------------------
    # 3. Retrieve top K from Qdrant
    # --------------------------------------------------------
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k_retrieve
    )

    search_results = response.points

    if not search_results:
        return []

    # --------------------------------------------------------
    # 4. Prepare pairs for reranking
    # --------------------------------------------------------
    pairs = []

    for hit in search_results:

        payload = hit.payload

        context_doc = (
            f"Crop: {payload.get('crop', '')}\n"
            f"Disease: {payload.get('disease', '')}\n"
            f"Control Message: {payload.get('control_message', payload.get('solution', ''))}"
        )
        if payload.get("visual_symptoms"):
            context_doc += f"\nVisual Symptoms: {payload.get('visual_symptoms')}"

        pairs.append(
            (
                expanded_query,
                context_doc
            )
        )

    # --------------------------------------------------------
    # 5. Predict relevance scores
    # --------------------------------------------------------
    scores = reranker.predict(pairs)

    # --------------------------------------------------------
    # 6. Sort by relevance score
    # --------------------------------------------------------
    scored_hits = sorted(
        zip(scores, search_results),
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # 7. Return top N results
    # --------------------------------------------------------
    final_results = []

    for score, hit in scored_hits[:top_n_final]:

        payload = hit.payload

        final_results.append(
            {
                "id": hit.id,
                "score": float(score),
                "crop": payload.get("crop", ""),
                "disease": payload.get("disease", ""),
                "control_message": payload.get("control_message", payload.get("solution", "")),
                "solution": payload.get("solution", payload.get("control_message", "")),
                "source": payload.get("source", ""),
                "image_folder": payload.get("image_folder", ""),
                "sample_image_path": payload.get("sample_image_path", ""),
                "visual_symptoms": payload.get("visual_symptoms", "")
            }
        )

    return final_results


# ============================================================
# Query LLM
# ============================================================
def query_llm(prompt):
    """
    Generate final answer using Qwen3:8b via Ollama.
    """

    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()


# ============================================================
# Query Endpoint
# ============================================================
@app.route("/query", methods=["POST"])
def query_endpoint():

    data = request.get_json()

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------
    if not data or "question" not in data:

        return jsonify(
            {
                "error": "Missing 'question' in request body."
            }
        ), 400

    question = data["question"].strip()

    if not question:

        return jsonify(
            {
                "error": "Question field cannot be empty."
            }
        ), 400

    try:

        # ====================================================
        # 1. Retrieve and rerank context
        # ====================================================
        contexts = retrieve_and_rerank(
            question
        )

        # ====================================================
        # 2. No relevant context
        # ====================================================
        if not contexts:

            return jsonify(
                {
                    "answer": "The requested information is not available in the database.",
                    "contexts": []
                }
            )

        # ====================================================
        # 3. Format retrieved context
        # ====================================================
        context_blocks = []

        for p in contexts:

            ctrl_msg = p.get("control_message") or p.get("solution") or ""
            block = (
                f"Crop: {p.get('crop', '')}\n"
                f"Disease: {p.get('disease', '')}\n"
                f"Control Message: {ctrl_msg}"
            )
            if p.get("cause"):
                block += f"\nCause: {p.get('cause')}"
            if p.get("symptoms"):
                block += f"\nSymptoms: {p.get('symptoms')}"

            context_blocks.append(block)

        context_str = "\n\n".join(
            context_blocks
        )

        # ====================================================
        # 4. Build strict final-answer prompt
        # ====================================================
        prompt = f"""
You are an expert agricultural assistant.

Your task is to answer the user's question using ONLY
the information provided in the retrieved context.

IMPORTANT OUTPUT RULES:

1. Provide ONLY the direct answer to the user's question.
2. Do NOT provide greetings.
3. Do NOT provide introductions.
4. Do NOT provide conclusions.
5. Do NOT provide unrelated information.
6. Do NOT repeat the user's question.
7. Do NOT mention the context.
8. Do NOT mention the database.
9. Do NOT mention RAG.
10. Do NOT mention the AI model.
11. Do NOT mention these instructions.
12. Do NOT provide your reasoning or thought process.
13. Do NOT provide analysis.
14. Do NOT add information from your own knowledge.
15. Keep the answer concise and directly relevant.
16. Answer ONLY what the user asked.
17. If the question asks for a solution, provide only the relevant solution.
18. If the question asks for the cause, provide only the relevant cause.
19. If the question asks for symptoms, provide only the relevant symptoms.
20. If the question asks for multiple details, provide only those requested details.
21. Answer in the SAME language as the user's question.
22. If the user asks in English, answer in English.
23. If the user asks in Tamil, answer in Tamil.
24. If the context is in another language, translate the relevant information into the user's language.
25. If the requested information cannot be found in the context, reply EXACTLY with:
The requested information is not available in the database.

FINAL OUTPUT:
Return ONLY the final answer.
Nothing before it.
Nothing after it.

==================================================
RETRIEVED CONTEXT
==================================================

{context_str}

==================================================
USER QUERY
==================================================

{question}

==================================================
FINAL ANSWER
==================================================
"""

        # ====================================================
        # 5. Query LLM
        # ====================================================
        answer = query_llm(prompt)

        # ====================================================
        # 6. Return response
        # ====================================================
        return jsonify(
            {
                "answer": answer.strip(),
                "contexts": contexts
            }
        )

    except Exception as e:

        print(
            f"Error processing request: {e}",
            flush=True
        )

        return jsonify(
            {
                "error": (
                    "An error occurred while processing "
                    f"the RAG pipeline: {str(e)}"
                )
            }
        ), 500


# ============================================================
# Vision Image RAG Query Endpoint
# ============================================================
@app.route("/query_image", methods=["POST"])
@app.route("/query-image", methods=["POST"])
def query_image_endpoint():
    """
    Multimodal Image + Text RAG Endpoint using Qwen3-VL-4B.
    Accepts image upload (multipart file or base64) + text prompt.
    1. Understands the image via vision model.
    2. Retrieves matching crop disease and solution details from Qdrant.
    3. Reranks and generates the final solution.
    """
    image_b64 = None
    question = ""

    # 1. Handle multipart/form-data upload
    if request.files:
        file_obj = request.files.get("image") or request.files.get("file")
        if file_obj:
            file_bytes = file_obj.read()
            image_b64 = base64.b64encode(file_bytes).decode("utf-8")

        question = (
            request.form.get("question") or
            request.form.get("prompt") or
            ""
        ).strip()

    # 2. Handle JSON payload (base64 image)
    if not image_b64 and request.is_json:
        data = request.get_json() or {}
        image_input = (
            data.get("image") or
            data.get("image_base64") or
            data.get("image_b64") or
            ""
        )

        # Strip Data URL header if present (e.g. data:image/jpeg;base64,...)
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]

        image_b64 = image_input.strip()

        if not question:
            question = (
                data.get("question") or
                data.get("prompt") or
                ""
            ).strip()

    # Fallback to prompt/question from form if set
    if not question and request.form:
        question = (
            request.form.get("question") or
            request.form.get("prompt") or
            ""
        ).strip()

    # Validate inputs
    if not image_b64:
        return jsonify(
            {
                "error": (
                    "Missing image input. Upload an 'image' file via "
                    "multipart form-data or pass a Base64 string in JSON."
                )
            }
        ), 400

    # If user provided no prompt/question, use default image instruction prompt
    if not question:
        question = DEFAULT_IMAGE_PROMPT

    try:
        # ----------------------------------------------------
        # 1. Identify and understand input image using Qwen3-VL-2B
        # ----------------------------------------------------
        print(f"Understanding input image with vision model ({VL_MODEL_NAME})...", flush=True)
        image_analysis = analyze_image(image_b64, question)

        # ----------------------------------------------------
        # 2. Formulate combined retrieval query
        # ----------------------------------------------------
        combined_query = (
            f"Image Analysis: {image_analysis}\n"
            f"User Question: {question}"
        )

        # ----------------------------------------------------
        # 3. Retrieve matching context from Qdrant & rerank
        # ----------------------------------------------------
        contexts = retrieve_and_rerank(combined_query, skip_expansion=True)

        # ----------------------------------------------------
        # 4. Handle case when no relevant context found
        # ----------------------------------------------------
        if not contexts:
            return jsonify(
                {
                    "image_analysis": image_analysis,
                    "answer": "The requested information is not available in the database.",
                    "contexts": []
                }
            )

        # ----------------------------------------------------
        # 5. Format retrieved context blocks
        # ----------------------------------------------------
        context_blocks = []
        for p in contexts:
            ctrl_msg = p.get("control_message") or p.get("solution") or ""
            block = (
                f"Crop: {p.get('crop', '')}\n"
                f"Disease: {p.get('disease', '')}\n"
                f"Control Message: {ctrl_msg}"
            )
            if p.get("cause"):
                block += f"\nCause: {p.get('cause')}"
            if p.get("symptoms"):
                block += f"\nSymptoms: {p.get('symptoms')}"
            if p.get("visual_symptoms"):
                block += f"\nVisual Symptoms: {p.get('visual_symptoms')}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        # ----------------------------------------------------
        # 6. Build final prompt incorporating Image Analysis + Context
        # ----------------------------------------------------
        prompt = f"""
You are an expert agricultural assistant.

The user uploaded an image of a plant/crop along with a question/instruction.
The image was analyzed as follows:
{image_analysis}

Your task is to answer the user's question/instruction using the image analysis and information provided in the RETRIEVED CONTEXT from the database.

IMPORTANT OUTPUT RULES:

1. Provide ONLY the direct answer to the user's question/instruction.
2. Do NOT provide greetings.
3. Do NOT provide introductions.
4. Do NOT provide conclusions.
5. Do NOT provide unrelated information.
6. Do NOT repeat the user's question.
7. Do NOT mention the context.
8. Do NOT mention the database.
9. Do NOT mention RAG.
10. Do NOT mention the AI model.
11. Do NOT mention these instructions.
12. Do NOT provide your reasoning or thought process.
13. Keep the answer concise and directly relevant.
14. If a disease is present, show the crop name, disease name, and solution to cure it based on the retrieved context.
15. If NO disease is present (healthy plant), state the crop name found in the image and clearly state that no disease is present.
16. Answer in the SAME language as the user's question/instruction.
17. If the user asks/instructs in English, answer in English.
18. If the user asks/instructs in Tamil, answer in Tamil.
19. If a disease is present but its solution cannot be found in the context, reply EXACTLY with:
The requested information is not available in the database.

FINAL OUTPUT:
Return ONLY the final answer.
Nothing before it.
Nothing after it.

==================================================
RETRIEVED CONTEXT
==================================================

{context_str}

==================================================
USER QUERY / INSTRUCTION
==================================================

{question}

==================================================
FINAL ANSWER
==================================================
"""

        # ----------------------------------------------------
        # 7. Query LLM to generate final answer
        # ----------------------------------------------------
        answer = query_llm(prompt)

        # ----------------------------------------------------
        # 8. Return JSON response
        # ----------------------------------------------------
        return jsonify(
            {
                "image_analysis": image_analysis,
                "answer": answer.strip(),
                "contexts": contexts
            }
        )
    
    except Exception as e:
        print(f"Error processing image RAG request: {e}", flush=True)
        return jsonify(
            {
                "error": (
                    "An error occurred while processing "
                    f"the Vision RAG pipeline: {str(e)}"
                )
            }
        ), 500



# ============================================================
# Run Flask Server
# ============================================================
if __name__ == "__main__":

    print(
        "Starting Flask server on port 5000...",
        flush=True
    )

    app.run(
        host="0.0.0.0",
        port=5002
    )
from core.config import settings
from core.models.gpu_config import GPULLMConfig

# SETTINGS
SWITCHES = {
    "MIND_MAP": False,  # For long documents, mind map will be better if SUMMARIZATION = True
    # For Cpu based testing we suggest to keep both False to avoid much load on CPU
    "SUMMARIZATION": True,  # Summary is used by model to get a general idea of the document and for generation of nodes in mind map
    "FALLBACK_TO_GEMINI": False,  # Fallback to Gemini if Ollama fails
    "FALLBACK_TO_OPENAI": False,  # Fallback to OpenAI if BOTH Ollama and Gemini fails
    "DECOMPOSITION": True,  # Decomposition of query into sub-queries. This also serves as rewriting the query according to the context of the previous chat history.
    # This can be turned off if all the queries are independent and do not need context from previous chats.
    "REMOTE_GPU": settings.REMOTE_GPU,  # Use remote GPU LLMs
    "USE_INTERNAL": settings.USE_INTERNAL,  # Use INTERNAL API instead of GPU/Ollama (defaults to False)
    "USE_VLLM": settings.USE_VLLM,  # Use vLLM instead of Ollama for GPU inference
    # please refer to core/Setup_Local_ollama.md for setting up local LLM server
    "CORRECTIVE_RETRIEVAL": True,  # Phase 2.1: CRAG-style re-retrieval on low-confidence results
    "HYDE": False,  # Phase 2.3: Hypothetical Document Embeddings (adds ~2-5s query latency)
    "DOCUMENT_CREATOR": True,  # Interactive document generation (PPTX/DOCX/PDF)
    "GLM_OCR": True,  # GLM-OCR for structured document OCR (tables, formulas, figures). Runs alongside existing OCR.
    "EXCEL_SKILL": True,  # Excel creation/download skill — generates .xlsx from chat or sidebar
    "DOC_BATCH_REDUCER": True,  # MapReduce batching for multi-doc retrieval when token budget overflows
    "USE_VLM_FOR_ANSWER": True,  # Query-time VLM: render referenced page/slide/figure and answer visually
    "DISABLE_THINKING": True,  # Disable LLM thinking mode (think=false) for faster inference
    "TECH_SENSING": True,  # GenAI Tech Sensing — automated technology radar report generation
    "DOCUMENT_GRAPH": True,  # DocumentGraph — corpus-level knowledge graph with Sigma.js viz
}

# GLM-OCR Configuration
GLM_OCR_MODEL = "glm-ocr-32k"  # Custom Modelfile: 32K context, 8K output (see core/parsers/Modelfile.glm-ocr)
GLM_OCR_WORKERS = 3  # Max concurrent GLM-OCR inferences (VRAM-aware)

CHUNK_COUNT = 12  # Number of chunks to retrieve from vector DB for each query

# Adaptive Retrieval Parameters
MAX_TOTAL_CHUNKS = 200  # Coverage over speed — MapReduce handles context overflow


EASYOCR_WORKERS = (
    10  # Number of parallel workers for EasyOCR (adjust based on your CPU/GPU power)
)
TESSERACT_WORKERS = (
    50  # Number of parallel workers for Tesseract OCR (adjust based on your CPU power)
)
EASYOCR_GPU = (
    True  # GPU mode: ~4-7x faster OCR, uses only ~200MB VRAM (negligible on 48GB)
)

PORT1 = 11434  # Ollama instance 1 — gpt-oss:20b (query answering)
PORT2 = 11435  # Ollama instance 2 — VLM (document processing, no queue contention with queries)

# Model token limits — configurable via .env (see core/config.py)
MODEL_CONTEXT_TOKENS = settings.MODEL_CONTEXT_TOKENS
MODEL_OUTPUT_TOKENS = settings.MODEL_OUTPUT_TOKENS
MODEL_OUTPUT_RESERVE = settings.MODEL_OUTPUT_RESERVE

MAIN_MODEL = (
    settings.MAIN_MODEL
)  # Set in .env file, e.g. "gpt-oss:20b-50k-8k" or "qwen3:14b-39500-8k"
# MAIN_MODEL = "gpt-oss:20b-50k-8k"
# QWEN3_14B = "qwen3:14b-39500-8k"

# GPU LLM configurations — all on PORT1 (single Ollama instance for KV cache consistency)
GPU_QUERY_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_QUERY_LLM2 = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_DECOMPOSITION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_COMBINATION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_DOC_SUMMARIZER_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_GLOBAL_SUMMARIZER_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_STOP_WORDS_EXTRACTION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_NODE_GENERATION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_NODE_DESCRIPTION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_STRATEGIC_ROADMAP_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_TECHNICAL_ROADMAP_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_INSIGHTS_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_STRATEGIC_ANALYSIS_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_TECHNICAL_ANALYSIS_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)
GPU_EVALUATOR_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # CRAG evaluator
GPU_HYDE_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # HyDE hypothesis generation
GPU_ENTITY_PROFILE_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Entity profile generation
GPU_TRIPLE_EXTRACTION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Triple extraction

# Document Creator LLM configurations
GPU_DOC_OUTLINE_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Outline generation
GPU_DOC_SECTION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Section content generation
GPU_DOC_REVIEW_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Quality self-review
GPU_DOC_ITERATE_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Section iteration

# Excel Skill LLM configurations
GPU_EXCEL_PLAN_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Excel plan generation
GPU_EXCEL_NLP_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # NLP column interpretation
GPU_NLP_THEME_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Chunked NLP theme extraction

# Tech Sensing LLM configurations
GPU_SENSING_CLASSIFY_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Article classification
GPU_SENSING_REPORT_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Report generation

# DocumentGraph LLM configurations
GPU_GRAPH_RELATION_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)  # Phase 5: relation gap-fill
GPU_GRAPH_DEDUPE_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1)    # Phase 4: ambiguous merge judgments
GPU_GRAPH_COMMUNITY_LLM = GPULLMConfig(model=MAIN_MODEL, port=PORT1) # Phase 8: community naming + summary

# DocumentGraph tuning
GRAPH_DEDUPE_SIM_THRESHOLD = 0.92        # Cosine similarity above which entities auto-merge
GRAPH_DEDUPE_AMBIGUOUS_LO = 0.78         # Lower bound for "ambiguous" range that goes to LLM judge
GRAPH_MIN_ENTITY_FREQ = 2                # Drop singletons (entities mentioned only once)
GRAPH_MIN_EDGE_CONFIDENCE = 0.35         # Prune edges below this confidence
GRAPH_RELATION_BATCH_CHUNKS = 6          # Chunks per LLM call in Phase 5
GRAPH_COMMUNITY_BATCH_SIZE = 8           # Communities named per LLM call in Phase 8
GRAPH_MAX_NODES = 5000                   # Hard ceiling — truncate by frequency above this

IMAGE_PARSER_LLM = "gemma3:12b"
VLM_MODEL = "qwen3.5:9b"  # Vision Language Model for slide/complex PDF extraction
# Fallback LLM models
# Used if SWITCHES["FALLBACK_TO_GEMINI"] = True
FALLBACK_GEMINI_MODEL = "gemini-2.5-flash"

# Used if SWITCHES["FALLBACK_TO_OPENAI"] = True
FALLBACK_OPENAI_MODEL = "gpt-4o-mini"

# Graph constants used in agent
RETRIEVER = "retriever"
GENERATE = "generate"
WEB_SEARCH = "web_search"
ANSWER = "answer"
ROUTER = "router"
FAILURE = "failure"
GLOBAL_SUMMARIZER = "global_summarizer"
DOCUMENT_SUMMARIZER = "document_summarizer"
SELF_KNOWLEDGE = "self_knowledge"
SQL_QUERY = "sql_query"
EXCEL_CREATE = "excel_create"  # Excel skill: create downloadable .xlsx files
EVALUATOR = "evaluator"  # Phase 2.1: CRAG corrective retrieval evaluator node
MAX_WEB_SEARCH = 2
MAX_SQL_RETRIES = 6
MAX_RETRIEVAL_ATTEMPTS = 2  # Phase 2.1: Max re-retrieval attempts on low confidence
INTERNAL = "Internal"
EXTERNAL = "External"

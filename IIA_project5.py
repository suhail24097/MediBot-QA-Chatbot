import os
import sys
import re
import json
import mysql.connector
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from collections import Counter, defaultdict
import difflib
import google.generativeai as genai

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        try:
            import codecs
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        except Exception:
            pass


MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "suhail@97"  
DISEASE_DB_NAME = "disease"
MEDICINE_DB_NAME = "medicine"
LOG_FILE = "query_log.txt"
GEMINI_API_KEY = "AIzaSyDR8cLuZN8LDOdY4BwURI-RplWgniISlNs"  


KNOWN_DISEASES = ["fungal infection", "diabetes", "dengue", "malaria", "covid", "bacterial infection", "cold", "flu", "cervical spondylosis"]


def connect_db(db_name):
    print(f"[FLOW] Attempting to connect to MySQL database: {db_name} at host {MYSQL_HOST} with user {MYSQL_USER}.")
    try:
        conn = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=db_name)
        print(f"[FLOW] Successfully connected to {db_name} database.")
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to {db_name}: {e}")
        raise

try:
    disease_db = connect_db(DISEASE_DB_NAME)
    medicine_db = connect_db(MEDICINE_DB_NAME)
    print("[OK] MySQL connected successfully to both databases.")
except Exception as e:
    print("[ERROR] Error connecting to MySQL:", e)
    sys.exit(1)

# printing sample files
def verify_db(db, table):
    cur = db.cursor()
    try:
        cur.execute("SELECT VERSION()")
        v = cur.fetchone()[0]
    except:
        v = "unknown"
    try:
        cur.execute(f"SHOW TABLES")
        tables = [r[0] for r in cur.fetchall()]
    except:
        tables = []
    sample = {}
    if table in tables:
        try:
            cur.execute(f"SELECT * FROM `{table}` LIMIT 3")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            sample = {"columns": cols, "rows": [dict(zip(cols, r)) for r in rows]}
        except Exception as e:
            sample = {"error": str(e)}
    cur.close()
    return {"server_version": str(v), "tables": tables, "sample": sample}

print("Running quick DB diagnostics...")
print(json.dumps({"disease_db": verify_db(disease_db, "diseaseandsymptoms"), "medicine_db": verify_db(medicine_db, "medicine_details")}, indent=2, ensure_ascii=False))

# flan t5 model
print("Loading extraction model (this may take a minute)...")
extract_model_name = "google/flan-t5-base"
extract_tokenizer = AutoTokenizer.from_pretrained(extract_model_name, use_fast=False)
extract_model = AutoModelForSeq2SeqLM.from_pretrained(extract_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extract_model = extract_model.to(device)

print("Initializing Gemini API for summarization...")
genai.configure(api_key=GEMINI_API_KEY)
try:
    summary_model = genai.GenerativeModel("gemini-1.5-flash-001")  
except Exception as e:
    print(f"[ERROR] Failed to initialize Gemini model: {e}")
    print("Available models:")
    for model in genai.list_models():
        print(model.name)
    sys.exit(1)


SQL_KEYWORDS = {"select","from","where","join","insert","update","delete","drop","create","alter"}
def is_sql_like(text: str) -> bool:
    if not text: return False
    t = text.strip().lower()
    if t.startswith("select") or " from " in t: return True
    return sum(1 for k in SQL_KEYWORDS if k in t) >= 2

def simple_spell_correct(word, choices=("infection","fever","cough","cold","headache","diarrhea","nausea","rash","dengue","malaria","fungal","bacterial","covid","diabetes","symptom","symptoms","fungal infection","cervical spondylosis")):
    best = difflib.get_close_matches(word.lower(), choices, n=1, cutoff=0.7)
    return best[0] if best else word

def clean_symptoms_list(symptoms_concat: str, max_keep=12):
    if not symptoms_concat: return []
    toks = re.split(r"[,\|;]+", symptoms_concat)
    cleaned = []
    for t in toks:
        if not t: continue
        s = (t or "").replace("_"," ").strip()
        s = re.sub(r"\s+"," ", s).strip().strip(".,;:")
        if s: cleaned.append(s)
    seen = set()
    uniq = []
    for s in cleaned:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq[:max_keep]

def clean_side_effects(side_effects: str):
    if not side_effects: return []
    toks = re.split(r"\s+", side_effects.strip())
    cleaned = [re.sub(r"[.,;:]","", t).strip() for t in toks if t.strip()]
    seen = set()
    uniq = [s for s in cleaned if s.lower() not in seen and not seen.add(s.lower())]
    return uniq[:10]

def aggregate_disease_rows(rows):
    agg = defaultdict(list)
    for r in rows:
        disease = (r.get("Disease") or r.get("disease") or "Unknown").strip()
        cleaned = clean_symptoms_list(r.get("symptoms",""), max_keep=100)
        if cleaned: agg[disease].append(cleaned)
    result = {}
    for d, lists in agg.items():
        flat = [s.lower() for sub in lists for s in sub]
        cnt = Counter(flat)
        common = [sym for sym, _ in cnt.most_common(12)]
        result[d] = {"symptoms": common, "counts": cnt}
    return result


def build_disease_query(term, limit=50):
    cols = ", ".join([f"Symptom_{i}" for i in range(1,18)])
    sql = ("SELECT Disease, TRIM(CONCAT_WS(', ', " + cols + ")) AS symptoms "
           f"FROM diseaseandsymptoms WHERE Disease LIKE %s OR (" + " OR ".join([f"Symptom_{i} LIKE %s" for i in range(1,18)]) + f") LIMIT {limit}")
    pattern = f"%{term}%"
    params = [pattern] + [pattern]*17
    return sql, tuple(params)

def build_medicine_query(term, limit=50):
    pattern = f"%{term}%"
    sql = ("SELECT `Medicine Name` AS medicine_name, Composition, Uses, Side_effects, Manufacturer "
           f"FROM medicine_details WHERE Uses LIKE %s OR Composition LIKE %s OR `Medicine Name` LIKE %s OR Side_effects LIKE %s LIMIT {limit}")
    return sql, (pattern, pattern, pattern, pattern)

def execute_param_query(db, sql, params):
    print(f"[FLOW] Executing parameterized query on database: {sql} with params {params}")
    try:
        cur = db.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        cur.close()
        print(f"[FLOW] Query executed successfully, fetched {len(rows)} rows.")
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        print(f"[ERROR] Query execution failed: {e}")
        return [{"error": str(e), "query": sql}]

def llm_extract(user_text, max_tokens=128):
    print(f"[FLOW] LLM extracting disease and intent from user query: '{user_text}'")
    prompt = (
        "You are a concise extractor. Output ONLY a JSON object with fields:\n"
        "1) intent: 'symptoms','medicines','side_effects','both' or 'unknown'.\n"
        "2) term: normalized disease/condition (short, lowercase phrase, prefer multi-word diseases like 'fungal infection' if applicable).\n"
        "3) reason: 1 short factual sentence describing why you chose this term and which DB columns should be searched.\n\n"
        f"User question: \"{user_text}\"\n\nReturn JSON only. Example: {{\"intent\":\"both\",\"term\":\"covid\",\"reason\":\"user asks medicines and side effects of covid; search medicine columns\"}}"
    )
    try:
        inputs = extract_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = extract_model.generate(**inputs, max_length=max_tokens, num_beams=4, early_stopping=True)
        raw = extract_tokenizer.decode(out[0], skip_special_tokens=True).strip()
        print(f"[FLOW] LLM raw output for extraction: '{raw}'")
    except Exception as e:
        print(f"[ERROR] LLM extraction failed: {e}")
        return heuristic_extract(user_text)
    try:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return heuristic_extract(user_text)
        parsed = json.loads(m.group(0))
        intent = parsed.get("intent","unknown").lower()
        term = (parsed.get("term","") or "").strip().lower()
        reason = parsed.get("reason","").strip()
        if intent not in {"symptoms","medicines","side_effects','both","unknown"}:
            if "symptom" in intent: intent="symptoms"
            elif "side" in intent: intent="side_effects"
            elif "medic" in intent or "treat" in intent: intent="medicines"
            else: intent="unknown"
        print(f"[FLOW] Extracted: intent='{intent}', term='{term}', reason='{reason}'")
        return {"intent":intent,"term":term,"reason":reason,"raw_output":raw}
    except Exception as e:
        print(f"[ERROR] Failed to parse LLM extraction output: {e}")
        return heuristic_extract(user_text)

def heuristic_extract(user_text):
    print(f"[FLOW] Using heuristic fallback to extract disease from query: '{user_text}'")
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", user_text.lower()) if t not in {"what","are","the","and","of","for","show","list","me","my","i"}]
    tokens = [simple_spell_correct(t) for t in tokens]
    term = ""
    for i in range(len(tokens)-1, -1, -1):  
        for j in range(i, max(i-2, -1), -1):
            phrase = " ".join(tokens[j:i+1])
            if phrase in KNOWN_DISEASES:
                term = phrase
                break
        if term:
            break
    if not term:
        term = next((t for t in tokens if t in KNOWN_DISEASES), tokens[-1] if tokens else "")
    intent = "both"
    reason = f"Heuristic selected term '{term}' from query tokens; searching both disease and medicine DBs"
    print(f"[FLOW] Heuristic extracted: intent='both', term='{term}', reason='{reason}'")
    return {"intent":"both","term":term,"reason":reason,"raw_output":"heuristic"}

#summarization

def gemini_summary(disease_data, medicine_data, user_query, extracted_term):
    print("[FLOW] Passing gathered DB data to Gemini API for natural summary generation.")
    disease_str = "Diseases:\n" + "\n".join([f"- {d}: {', '.join(meta['symptoms'])}" for d, meta in disease_data.items()]) if disease_data else "No disease data found."
    medicine_str = "Medicines:\n" + "\n".join([f"- {r.get('medicine_name')}: Uses - {r.get('Uses') or 'N/A'}; Side effects - {', '.join(clean_side_effects(r.get('Side_effects', '')))}" for r in medicine_data[:8]]) if medicine_data else "No medicines found."
    data_str = f"{disease_str}\n\n{medicine_str}"
    
    prompt = (
        f"You are a medical assistant. Generate a concise, natural summary in your own words for '{extracted_term}' based on the user's query and database data. "
        f"Focus on the user's request (e.g., medicines if asked). Summarize key medicines and their side effects from the data, avoiding raw data repetition. "
        f"If disease data is found, include key symptoms. If no disease data, note it and include general symptoms for '{extracted_term}' (e.g., for cervical spondylosis: neck pain, stiffness). "
        f"If no medicine data is found, note it and mention general treatment options (e.g., for cervical spondylosis: pain relievers, physical therapy). "
        f"Do not include diseases or symptoms not in the data. Keep it user-friendly and end with: 'Consult a healthcare professional for proper diagnosis and treatment.'\n\n"
        f"User query: \"{user_query}\"\n"
        f"Extracted disease: \"{extracted_term}\"\n"
        f"Data:\n{data_str}\n"
        "Summary:"
    )
    print(f"[FLOW] Gemini summary prompt: {prompt}")
    try:
        response = summary_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=768,
                temperature=0.7,
            )
        )
        summary = response.text.strip()
        print(f"[FLOW] Gemini generated summary: '{summary}'")
        if not summary or summary == data_str:
            raise ValueError("Summary is invalid or repeats data")
        return summary
    except Exception as e:
        print(f"[ERROR] Gemini summary generation failed or invalid: {e}")
        fallback = ""
        if disease_data:
            for d, meta in disease_data.items():
                symptoms = ", ".join(meta['symptoms'])
                fallback += f"Cervical spondylosis is associated with symptoms like {symptoms}. "
        else:
            fallback += f"No disease data found for {extracted_term}. Common symptoms of cervical spondylosis include neck pain and stiffness. "
        if medicine_data:
            med_list = ", ".join([r.get('medicine_name') for r in medicine_data[:8]])
            side_effects = clean_side_effects(" ".join([r.get('Side_effects', '') for r in medicine_data]))
            side_effects_str = ", ".join(side_effects) if side_effects else "unknown side effects"
            fallback += f"Treatments include medicines like {med_list}, which may cause side effects such as {side_effects_str}. "
        else:
            fallback += f"No specific medicines found for {extracted_term}. General treatments for cervical spondylosis may include pain relievers, physical therapy, or muscle relaxants. "
        fallback += "Consult a healthcare professional for proper diagnosis and treatment."
        return fallback

def process_query(user_query: str):
    print(f"\n[USER QUERY] {user_query}")
    print("[FLOW] Starting query processing flow.")

    extraction = llm_extract(user_query)
    if not extraction:
        extraction = heuristic_extract(user_query)
    extraction["intent"] = "both"
    extraction["reason"] = (extraction.get("reason","") + " | forced 'both' to search both disease and medicine DBs").strip()
    print("\n[LLM EXTRACTION JSON]:")
    print(json.dumps(extraction, indent=2, ensure_ascii=False))

    term = extraction.get("term","").lower()
    term = re.sub(r"\b(symptom|symptoms|medicine|medicines|side effect|side effects|treatment|treatments|sypmtom|sypmtoms)\b","", term)
    term = re.sub(r"\s+"," ", term).strip()
    if not term:
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", user_query.lower()) if t not in {"what","are","the","and","of","for","show","list","me","my","i"}]
        tokens = [simple_spell_correct(t) for t in tokens]
        for i in range(len(tokens)-1, -1, -1):
            for j in range(i, max(i-2, -1), -1):
                phrase = " ".join(tokens[j:i+1])
                if phrase in KNOWN_DISEASES:
                    term = phrase
                    break
            if term:
                break
        if not term:
            term = next((t for t in tokens if t in KNOWN_DISEASES), tokens[-1] if tokens else "")
    print("→ using search term:", repr(term))
    print(f"[FLOW] Extracted and cleaned search term (disease/condition): '{term}'. Passing to both DBs for search.")

    disease_rows = []
    medicine_rows = []
    executed = []

    d_sql, d_params = build_disease_query(term)
    disease_rows = execute_param_query(disease_db, d_sql, d_params)
    executed.append(("disease", d_sql, d_params))

    m_sql, m_params = build_medicine_query(term)
    medicine_rows = execute_param_query(medicine_db, m_sql, m_params)
    executed.append(("medicine", m_sql, m_params))

    print(f"[FLOW] Gathered information from DBs: {len(disease_rows)} disease rows, {len(medicine_rows)} medicine rows.")

    print("\n=== PIPELINE DEBUG ===")
    print("User:", user_query)
    print("Extraction:", json.dumps(extraction, ensure_ascii=False))
    for label, sql, params in executed:
        print(f"\nEXECUTED ({label}) SQL:")
        print(sql)
        print("params sample:", params[:4] if isinstance(params,(list,tuple)) else params)
    print("\nDisease sample rows (cleaned):")
    for r in disease_rows[:5]:
        print(" -", r.get("Disease"), "=>", clean_symptoms_list(r.get("symptoms","")))
    print("\nMedicine sample rows:")
    for r in medicine_rows[:8]:
        print(" -", r.get("medicine_name"), "| Uses:", (r.get("Uses") or "")[:120], "| Side effects:", ", ".join(clean_side_effects(r.get("Side_effects", ""))))
    print("=== END PIPELINE DEBUG ===\n")

    agg_diseases = aggregate_disease_rows(disease_rows)
    final = gemini_summary(agg_diseases, medicine_rows, user_query, term)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n--- RUN ---\n")
        f.write("TS: " + str(datetime.now()) + "\n")
        f.write("QUERY: " + user_query + "\n")
        f.write("EXTRACTION: " + json.dumps(extraction, ensure_ascii=False) + "\n")
        for label, sql, params in executed:
            f.write(f"EXECUTED_{label}_SQL: {sql}\nparams: {str(params)[:400]}\n")
        f.write("DISEASE_ROWS: " + json.dumps(disease_rows[:10], ensure_ascii=False) + "\n")
        f.write("MED_ROWS: " + json.dumps(medicine_rows[:20], ensure_ascii=False) + "\n")
        f.write("FINAL: " + final + "\n")
    return final


if __name__ == "__main__":
    print("\n=== Medical Query Assistant (Gemini Pipeline) ===")
    print("Type 'exit' to quit.")
    while True:
        try:
            q = input("\nEnter your query (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye."); break
        if not q or q.lower() in ("exit","quit"):
            print("Goodbye."); break
        out = process_query(q)
        print("\n🩺 FINAL ANSWER:\n", out)
        print("\n" + "-"*60 + "\n")
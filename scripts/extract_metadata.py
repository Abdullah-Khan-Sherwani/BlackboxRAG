import os
import json
import logging
import re
from pathlib import Path
from tqdm import tqdm
import sys

# Add the project root to the path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.llm.ollama_client import call_ollama, is_ollama_running

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "dataset-pipeline" / "data" / "extracted" / "extracted"
PROCESSED_DIR = project_root / "data" / "processed"
METADATA_FILE = PROCESSED_DIR / "report_metadata.json"

REQUIRED_KEYS = [
    "report_type", "date", "location", "aircraft_type", "operator",
    "flight_type", "fatalities", "probable_cause", "contributing_factors",
    "safety_issues", "recommendations", "key_findings"
]

# ─────────────────────────────────────────────
# REGEX PRE-EXTRACTION for structured fields
# ─────────────────────────────────────────────

def regex_extract_date(content: str) -> str | None:
    """Extract accident date using multiple regex patterns."""
    # Pattern 1: Full date like "August 6, 1997" or "July 17, 1996" or "March 3, 1991"
    months = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    
    # Check title area first (first 20 lines)
    title_area = '\n'.join(content.split('\n')[:20])
    m = re.search(rf'({months}\s+\d{{1,2}},?\s+\d{{4}})', title_area)
    if m:
        return m.group(1).strip()
    
    # Check History of Flight section opening
    hof_match = re.search(r'History of (?:the )?Flight\s*\n+(.*?)(?:\n#|\Z)', content, re.IGNORECASE | re.DOTALL)
    if hof_match:
        first_para = hof_match.group(1)[:500]
        m = re.search(rf'[Oo]n\s+({months}\s+\d{{1,2}},?\s+\d{{4}})', first_para)
        if m:
            return m.group(1).strip()
    
    # Pattern 2: ISO-style date "2000-12-24"
    m = re.search(r'(\d{4}-\d{2}-\d{2})', title_area)
    if m:
        return m.group(1)
    
    # Pattern 3: In the abstract or citation line
    citation_area = '\n'.join(content.split('\n')[:35])
    m = re.search(rf'({months}\s+\d{{1,2}},?\s+\d{{4}})', citation_area)
    if m:
        return m.group(1).strip()
    
    return None


def regex_extract_fatalities(content: str) -> int | None:
    """Extract fatalities from the Injuries to Persons table."""
    # Look for the injury table which has a "Fatal" row with a Total column
    # Pattern: | Fatal | ... | <number> | or | <number> | ... | Fatal |
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if 'Fatal' in line and '|' in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            # The Total is typically the last numeric column
            for cell in reversed(cells):
                try:
                    val = int(cell)
                    if val >= 0:
                        return val
                except ValueError:
                    continue
    
    # Fallback: search for "All X people on board were killed"
    m = re.search(r'[Aa]ll\s+(\d+)\s+(?:people|persons|occupants)\s+(?:on board\s+)?(?:were|was)\s+killed', content)
    if m:
        return int(m.group(1))
    
    # Fallback: "X passengers aboard were fatally injured"
    m = re.search(r'(\d+)\s+passengers?\s+aboard\s+were\s+fatally\s+injured', content)
    if m:
        return int(m.group(1))
    
    return None


def regex_extract_from_title(content: str) -> dict:
    """Extract aircraft type, operator, location, date from the title/header lines."""
    lines = content.split('\n')
    title_area = '\n'.join(lines[:25])
    
    result = {}
    
    # Common aircraft types
    aircraft_pattern = r'(Boeing\s+\d{3}[A-Za-z0-9-]*|Airbus\s+A\d{3}[A-Za-z0-9-]*|McDonnell\s+Douglas\s+[A-Z]+[-]?\d+[A-Za-z0-9-]*|MD-\d+[A-Za-z0-9-]*|Cessna\s+\d+[A-Za-z]*|Beech(?:craft)?\s+\w+|Embraer\s+\w+|Bombardier\s+\w+|de\s+Havilland\s+\w+|ATR[-\s]?\d+)'
    m = re.search(aircraft_pattern, title_area, re.IGNORECASE)
    if m:
        result['aircraft_type'] = m.group(1).strip()
    
    # Try to get aircraft registration (N-number)
    reg_match = re.search(r'[,\s](N\d+[A-Z]*)[,\s]', title_area)
    if reg_match and 'aircraft_type' in result:
        result['aircraft_type'] = result['aircraft_type'] + ', ' + reg_match.group(1)
    
    return result


def regex_extract_operator(content: str) -> str | None:
    """Extract airline/operator from title and early text."""
    lines = content.split('\n')
    title_area = '\n'.join(lines[:25])
    
    # Common airline patterns in NTSB titles
    # "Federal Express, Inc." "Trans World Airlines" "United Airlines" "American Airlines" 
    # "Korean Air" "Southwest Airlines" "Delta Air Lines"
    airlines = [
        r'((?:United|American|Delta|Southwest|Continental|Northwest|USAir|US\s*Airways|Eastern|TWA|Trans\s+World\s+Airlines|'
        r'Federal\s+Express|FedEx|Korean\s+Air|Alaska\s+Airlines|JetBlue|Spirit|Frontier|Allegiant|'
        r'Hawaiian|Aloha|Piedmont|Braniff|Pan\s+American|Comair|Atlantic\s+Southeast|'
        r'Air\s+Midwest|Executive\s+Airlines|Colgan\s+Air|Pinnacle|Mesa|SkyWest|'
        r'Emery\s+Worldwide|Fine\s+Air|Air\s+Sunshine|Casino\s+Express|'
        r'ValuJet|ATA|AirTran|Midwest\s+Express|Atlas\s+Air)(?:\s*[\w\s,.]*))',
    ]
    
    for pattern in airlines:
        m = re.search(pattern, title_area, re.IGNORECASE)
        if m:
            # Clean up: take just the airline name, not trailing words
            name = m.group(1).strip()
            # Trim after common suffixes
            for suffix in [', Inc.', ' Inc.', ', LLC', ' Airlines', ' Air Lines', ' Air', ' Worldwide']:
                idx = name.lower().find(suffix.lower())
                if idx > 0:
                    name = name[:idx + len(suffix)]
                    break
            return name.strip(' ,.')
    
    return None


def regex_extract_location(content: str) -> str | None:
    """Extract accident location from title and History of Flight."""
    lines = content.split('\n')
    
    # Title area typically has location after aircraft registration
    title_area = '\n'.join(lines[:25])
    
    # Pattern: "Near <City>, <State>" or just "<City>, <State/Country>"
    m = re.search(r'[Nn]ear\s+([A-Z][a-zA-Z\s]+,\s+[A-Z][a-zA-Z\s]+)', title_area)
    if m:
        return 'Near ' + m.group(1).strip()
    
    # US state patterns in title
    us_states = (
        r'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|'
        r'Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|'
        r'Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|'
        r'New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|'
        r'Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|'
        r'Washington|West\s+Virginia|Wisconsin|Wyoming|Guam|Puerto\s+Rico|Colombia'
    )
    
    # Look for "<City>, <State>" in title
    m = re.search(rf'([A-Z][a-zA-Z\s]+),\s+({us_states})', title_area)
    if m:
        return f"{m.group(1).strip()}, {m.group(2).strip()}"
    
    return None


def regex_extract_report_type(filename: str) -> str:
    """Determine report type from filename prefix."""
    stem = Path(filename).stem.upper()
    if stem.startswith('AAR'):
        return 'AAR'  # Aircraft Accident Report
    elif stem.startswith('AIR'):
        return 'AIR'  # Aircraft Incident Report / Safety Recommendation
    elif stem.startswith('ASR'):
        return 'ASR'  # Aviation Special Report
    elif stem.startswith('MAR'):
        return 'MAR'  # Marine Accident Report
    elif stem.startswith('RAR'):
        return 'RAR'  # Railroad Accident Report
    return 'Unknown'


# ─────────────────────────────────────────────
# LLM EXTRACTION for unstructured fields
# ─────────────────────────────────────────────

def extract_llm_sections(file_path: Path) -> str:
    """Extract targeted sections for LLM to analyze (analytical/narrative content only)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    header = '\n'.join(lines[:35])
    
    # Sections for the LLM — only analytical/narrative ones
    target_keywords = ['executive summary', 'synopsis', 'abstract', 'probable cause',
                       'conclusion', 'finding', 'recommendation', 'history of flight',
                       'history of the flight']
    skip_sections = [
        'airplane information', 'aircraft information', 'personnel information',
        'meteorological', 'wreckage', 'fire', 'survival', 'tests and research',
        'medical', 'communications', 'aids to navigation', 'airport information',
        'flight recorders', 'other damage', 'abbreviation', 'organizational',
        'appendix', 'figure', 'table of', 'contents', 'damage to',
    ]
    
    sections = []
    capture = False
    buf = []
    line_count = 0
    max_lines = 40
    
    for line in lines[35:]:
        clean = line.strip().lower()
        if clean.startswith('#'):
            if capture and buf:
                sections.append('\n'.join(buf))
                buf = []
                line_count = 0
                capture = False
            
            if any(kw in clean for kw in skip_sections):
                capture = False
                continue
            
            if any(kw in clean for kw in target_keywords):
                buf = [line]
                line_count = 0
                max_lines = 50 if 'history' in clean else 35
                capture = True
            else:
                capture = False
        elif capture:
            line_count += 1
            if line_count <= max_lines:
                buf.append(line)
    
    if capture and buf:
        sections.append('\n'.join(buf))
    
    combined = header + '\n\n---KEY SECTIONS---\n\n' + '\n\n'.join(sections)
    words = combined.split()
    if len(words) > 5000:
        words = words[:5000]
    return ' '.join(words)


SYSTEM_PROMPT = """You extract metadata from NTSB aviation accident/incident reports.
Output ONLY a valid JSON object with exactly these keys. Do not add extra keys.
For list fields, provide 1-5 concise items. Use null for unknown values."""


def extract_with_llm(text: str) -> dict:
    """Use LLM for the fields that need interpretation."""
    prompt = f"""Read this NTSB report and extract the following information into JSON.

REPORT TEXT:
{text}

Complete this JSON (fill in values, use null if not found, use [] for empty lists):
{{
  "flight_type": "___",
  "probable_cause": "___",
  "contributing_factors": ["___"],
  "safety_issues": ["___"],
  "recommendations": ["___"],
  "key_findings": ["___"]
}}"""
    
    response = call_ollama(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        model="qwen2.5:32b",
        temperature=0.0,
        max_tokens=2000,
        json_mode=True,
        timeout=180
    )
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        clean = response.strip()
        clean = re.sub(r'^```json?\s*', '', clean)
        clean = re.sub(r'\s*```$', '', clean)
        return json.loads(clean)


# ─────────────────────────────────────────────
# KEY NORMALIZATION (safety net for LLM output)
# ─────────────────────────────────────────────

LLM_KEY_ALIASES = {
    "flight_type": ["operation_type", "flight_operation", "type_of_flight", "operation"],
    "probable_cause": ["cause_of_accident", "cause", "probable_cause_determination", "accident_cause"],
    "contributing_factors": ["contributing_causes", "factors"],
    "safety_issues": ["safety_concerns", "issues"],
    "recommendations": ["safety_recommendations"],
    "key_findings": ["findings", "main_findings"],
}

def normalize_llm_keys(metadata: dict) -> dict:
    """Map alternative key names from LLM to expected schema."""
    normalized = {}
    for expected_key, aliases in LLM_KEY_ALIASES.items():
        if expected_key in metadata:
            normalized[expected_key] = metadata[expected_key]
        else:
            for alias in aliases:
                if alias in metadata:
                    normalized[expected_key] = metadata[alias]
                    break
    return normalized


def ensure_schema(metadata: dict) -> dict:
    """Ensure all required keys exist with appropriate defaults."""
    list_keys = {'contributing_factors', 'safety_issues', 'recommendations', 'key_findings'}
    for key in REQUIRED_KEYS:
        if key not in metadata:
            metadata[key] = [] if key in list_keys else None
    return metadata


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def process_metadata():
    if not is_ollama_running():
        logger.error("Ollama is not running. Please start Ollama before extracting metadata.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing progress
    existing_metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                for item in existing_data:
                    rid = item.get("report_id")
                    if rid:
                        existing_metadata[rid] = item

    md_files = sorted(DATA_DIR.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files.")
    
    results = list(existing_metadata.values())
    if results:
        logger.info(f"Resuming: {len(results)} reports already processed.")
    
    for md_file in tqdm(md_files, desc="Extracting metadata"):
        report_id = md_file.stem
        
        if report_id in existing_metadata:
            continue
            
        try:
            # Read full file content
            with open(md_file, 'r', encoding='utf-8') as f:
                full_content = f.read()
            
            # ── STEP 1: Regex extraction for structured fields ──
            date = regex_extract_date(full_content)
            fatalities = regex_extract_fatalities(full_content)
            title_info = regex_extract_from_title(full_content)
            operator = regex_extract_operator(full_content)
            location = regex_extract_location(full_content)
            report_type = regex_extract_report_type(md_file.name)
            
            logger.info(f"[REGEX] {report_id}: date={date}, fatalities={fatalities}, "
                        f"aircraft={title_info.get('aircraft_type')}, operator={operator}, "
                        f"location={location}")
            
            # ── STEP 2: LLM extraction for analytical fields ──
            llm_text = extract_llm_sections(md_file)
            llm_data = extract_with_llm(llm_text)
            llm_data = normalize_llm_keys(llm_data)
            
            # ── STEP 3: Merge regex + LLM results (regex takes priority) ──
            metadata = {
                "report_id": report_id,
                "report_type": report_type,
                "date": date,
                "location": location or llm_data.get("location"),
                "aircraft_type": title_info.get("aircraft_type") or llm_data.get("aircraft_type"),
                "operator": operator or llm_data.get("operator"),
                "flight_type": llm_data.get("flight_type"),
                "fatalities": fatalities,
                "probable_cause": llm_data.get("probable_cause"),
                "contributing_factors": llm_data.get("contributing_factors", []),
                "safety_issues": llm_data.get("safety_issues", []),
                "recommendations": llm_data.get("recommendations", []),
                "key_findings": llm_data.get("key_findings", []),
            }
            
            metadata = ensure_schema(metadata)
            results.append(metadata)
            existing_metadata[report_id] = metadata
            
            # Save after every report
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to process {report_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    logger.info(f"Metadata extraction complete. {len(results)} reports processed.")
    logger.info(f"Results saved to {METADATA_FILE}")

if __name__ == "__main__":
    process_metadata()

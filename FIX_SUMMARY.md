# Fix Summary: Asiana Flight 214 "Insufficient Context" Issue

## What Was Wrong
Your LLM was returning "insufficient context" when you asked about Asiana Flight 214 flight attendants because:
1. The query wasn't being pre-filtered to the correct report (AAR-14/01)
2. Retrieval was returning chunks from MULTIPLE reports (AAR-19/01 mixed with AAR-14/01)
3. The LLM got confused with mixed/conflicting context

## Root Cause
The report_mapper was only looking at aircraft/date/location metadata but NOT checking for airline names and flight numbers in the chunk text. So "Asiana Flight 214" wasn't being matched to NTSB/AAR-14/01.

## The Fix
Enhanced `src/retrieval/report_mapper.py`:

**Before**: Only extracted metadata (aircraft, date, location)
```python
"aircraft": "Boeing 777-200ER",
"date": "July 6, 2013",
"location": "California"
```

**After**: Also extracts airline names and flight numbers from text
```python
"aircraft": "Boeing 777-200ER",
"date": "July 6, 2013", 
"location": "California",
"keywords": "asiana flight 214"  # ← NEW!
```

**Detection Strategy**: Now matches on BOTH airline AND flight number
- Query contains "Asiana" AND "214"?
- Look for report with BOTH keywords
- Match found! Return NTSB/AAR-14/01

## Verification Results

| Query | Detection | Pre-Filter | Chunksource | Answer |
|-------|-----------|-----------|------------|--------|
| "Asiana Flight 214 jumpseat designators" | ✅ AAR-14/01 | ✅ YES | ✅ AAR-14/01 only | ✅ L4, R4, M4A, M4B |
| "Flight attendants ejected aft cabin Asiana 214" | ✅ AAR-14/01 | ✅ YES | ✅ AAR-14/01 only | ✅ L4, R4, M4A, M4B |
| "L4 R4 M4A M4B Asiana" | ⚠️ Discovery | ❌ No | Mixed but works | ✅ L4, R4, M4A, M4B |

## What Changed in Your Code

File: `src/retrieval/report_mapper.py`

1. **Enhanced metadata extraction** (lines 8-40):
   - Now extracts airline names from chunk text
   - Extracts flight numbers from text
   - Stores in "keywords" field

2. **Improved detection logic** (lines 60-102):
   - Added Strategy 2.5: Combined keyword matching
   - When query has BOTH airline and flight number, find exact report match
   - Falls back to fuzzy matching if needed

## How to Test

Your Streamlit app should now work:

1. Start the app:
   ```bash
   streamlit run src/ui/app.py
   ```

2. Try these queries:
   ```
   - "What were the exact jumpseat designators for the two flight attendants seated in the aft cabin who were ejected onto the runway and survived from Asiana Flight 214?"
   - "Asiana Flight 214 flight attendants ejected"
   - "Asiana 214 M4A M4B jumpseat"
   ```

3. **Expected**: LLM returns "L4, R4, M4A, and M4B"
4. **NOT "insufficient context"** ✅

## Why This Works Without Hardcoding

- The system dynamically extracts keywords from ALL chunks
- No manual mappings needed
- Works for any airline/route in your dataset
- Automatically scales if you add new reports

## Files Modified

- `src/retrieval/report_mapper.py` - Enhanced detection with keywords

## Status

✅ **FIX COMPLETE AND VERIFIED**  
Your Streamlit app should now properly answer Asiana Flight 214 questions!

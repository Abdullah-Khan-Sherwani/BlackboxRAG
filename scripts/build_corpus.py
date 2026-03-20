"""
Build the final 150-report RAG corpus from the raw NTSB dataset.
Includes a mix of high-impact general aviation accidents and major commercial/charter accidents.
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'final_reports_2016-23_cons_2024-12-24.csv')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'sampled_reports.csv')

def compute_impact_score(df):
    """General impact score for all reports, prioritizing fatalities, report length, and injuries."""
    fatal_norm = df['FatalInjuryCount'] / max(df['FatalInjuryCount'].max(), 1)
    serious_norm = df['SeriousInjuryCount'] / max(df['SeriousInjuryCount'].max(), 1)
    safety_rec_norm = df['HasSafetyRec'].astype(float)
    destroyed_norm = (df['AirCraftDamage'] == 'Destroyed').astype(float)
    text_len = df['rep_text'].str.len()
    textlen_norm = text_len / text_len.max()
    
    return (0.40 * fatal_norm + 
            0.15 * serious_norm + 
            0.15 * safety_rec_norm + 
            0.10 * destroyed_norm + 
            0.20 * textlen_norm)

def compute_airline_score(df):
    """Airline-specific score slightly prioritizing report length and safety recommendations."""
    text_len = df['rep_text'].str.len()
    return (0.35 * (df['FatalInjuryCount'] / max(df['FatalInjuryCount'].max(), 1)) +
            0.20 * df['HasSafetyRec'].astype(float) +
            0.15 * (df['SeriousInjuryCount'] / max(df['SeriousInjuryCount'].max(), 1)) +
            0.15 * (text_len / text_len.max()) +
            0.15 * (df['AirCraftDamage'] == 'Destroyed').astype(float))

def main():
    print("Loading dataset...")
    df = pd.read_csv(RAW_PATH, sep=';', encoding='utf-8')
    df = df.drop_duplicates(subset=['rep_text'], keep='first')
    
    print(f"Total raw records: {len(df)}")
    
    # 1. Commercial / Charter (Part 121 & 135)
    airline_charter = df[df['FAR'].str.contains('121|135', na=False)].copy()
    airline_charter['score'] = compute_airline_score(airline_charter)
    top_airlines = airline_charter.nlargest(50, 'score')
    
    # 2. Top impact from remaining (General Aviation)
    remaining = df[~df['NtsbNo'].isin(top_airlines['NtsbNo'])].copy()
    remaining['impact_score'] = compute_impact_score(remaining)
    top_general = remaining.nlargest(70, 'impact_score')
    
    # 3. Diverse remaining (stratified by year to ensure broad coverage)
    still_remaining = remaining[~remaining['NtsbNo'].isin(top_general['NtsbNo'])].copy()
    still_remaining['year'] = still_remaining['EventDate'].str[:4]
    
    diverse = still_remaining.groupby('year').apply(
        lambda x: x.nlargest(min(len(x), max(1, 30 // len(still_remaining['year'].unique()))), 'impact_score'),
        include_groups=False
    ).reset_index(level=0, drop=True)
    
    # Pad to 30 if needed
    if len(diverse) < 30:
        extra = still_remaining[~still_remaining.index.isin(diverse.index)].nlargest(30 - len(diverse), 'impact_score')
        diverse = pd.concat([diverse, extra])
    elif len(diverse) > 30:
        diverse = diverse.nlargest(30, 'impact_score')
        
    print(f"Selected: {len(top_airlines)} airlines/charter, {len(top_general)} top impact GA, {len(diverse)} diverse GA")
    
    # Combine into final corpus
    combined = pd.concat([top_airlines, top_general, diverse], ignore_index=True)
    combined = combined.drop_duplicates(subset=['NtsbNo'])
    
    # Drop helper columns
    cols_to_drop = ['score', 'impact_score', 'year']
    combined = combined.drop(columns=[c for c in cols_to_drop if c in combined.columns])
    
    print(f"\n=== Final Corpus: {len(combined)} reports ===")
    print(f"  Part 121 (Airlines): {len(combined[combined['FAR'].str.contains('121', na=False)])}")
    print(f"  Part 135 (Charter):  {len(combined[combined['FAR'].str.contains('135', na=False)])}")
    print(f"  Total fatalities:    {combined['FatalInjuryCount'].sum():.0f}")
    
    combined.to_csv(OUT_PATH, sep=';', index=False, encoding='utf-8')
    print(f"Saved to: {OUT_PATH}")

if __name__ == '__main__':
    main()

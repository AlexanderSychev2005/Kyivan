import json
import re

roman_to_cent = {
    'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18,
    'Х': 10, 'ХI': 11, 'ХII': 12, 'ХIII': 13, 'ХIV': 14, 'ХV': 15, 'ХVI': 16, 'ХVII': 17, 'ХVIII': 18 # Cyrillic 'Х'
}

def parse_bulgarian_date(date_str):
    if not date_str or date_str.lower() in ('nn', 'tt', 'нормализиран текст', 'miscellanea'):
        return None
        
    s = str(date_str).upper()
    
    # Check for direct year exact matches: e.g. "1263 г."
    exact = re.findall(r'\b(1[0-9]{3})\b', s)
    if exact and "В" not in s and "ВЕК" not in s:
        nums = [int(x) for x in exact]
        if len(nums) == 2:
            return f"{min(nums)}-{max(nums)}"
        elif len(nums) >= 1:
            return f"{nums[0]}-{nums[0]}"
            
    # Find centuries
    centuries = []
    # Match roman numerals (using cyrillic and latin X, V, I)
    for rom, c in roman_to_cent.items():
        if re.search(r'\b' + rom + r'\b', s):
            centuries.append(c)
            
    # Match arabic centuries "15 в."
    m_ar = re.findall(r'\b([1-9][0-9])\s*В', s)
    for m in m_ar:
        centuries.append(int(m))
        
    if not centuries:
        # Fallback to pure numbers if nothing else matches
        if exact:
            nums = [int(x) for x in exact]
            return f"{min(nums)}-{max(nums)}"
        return None
        
    c_start = min(centuries)
    c_end = max(centuries)
    
    start_year = (c_start - 1) * 100
    end_year = (c_end - 1) * 100 + 99
    
    # Modifiers
    s_low = str(date_str).lower()
    
    if "първа половина" in s_low or "първата половина" in s_low:
        end_year = start_year + 49
    elif "втора половина" in s_low or "втората половина" in s_low:
        start_year = start_year + 50
    elif "първа четвърт" in s_low or "първата четвърт" in s_low:
        end_year = start_year + 24
    elif "трета четвърт" in s_low or "третата четвърт" in s_low:
        start_year = start_year + 50
        end_year = start_year + 24
    elif "последна четвърт" in s_low or "последната четвърт" in s_low:
        start_year = start_year + 75
    elif "среда" in s_low:
        start_year = start_year + 40
        end_year = start_year + 20
    elif "край" in s_low or "края" in s_low:
        start_year = start_year + 80
    elif "начало" in s_low or "началото" in s_low:
        end_year = start_year + 20
        
    return f"{start_year}-{end_year}"

def main():
    in_file = 'data/sofia/histdict_normalized.json'
    out_file = 'data/sofia/sofia_cleaned.json'
    
    with open(in_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    cleaned = []
    
    for doc in data:
        text = doc.get('content', '')
        if not text: continue
        
        # Раскрываем скобки: убираем символы ( ) [ ]
        text = re.sub(r'[\(\)\[\]]', '', text)
        
        raw_date = doc.get('manuscript_date', '')
        if not raw_date:
            raw_date = doc.get('translation_date', '')
            
        year_interval = parse_bulgarian_date(raw_date)
        
        cleaned.append({
            'doc_id': doc.get('doc_id', 'sofia_unk'),
            'text': text,
            'year': year_interval,
            'dialect': 'CS',
            'source': 'sofia_corpus'
        })
        
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
        
    print(f"Processed {len(cleaned)} docs into {out_file}")

if __name__ == '__main__':
    main()

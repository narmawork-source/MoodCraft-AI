from typing import Dict, List


CATALOG = [
    {'title':'Neutral Area Rug','price':129,'retailer':'Walmart','url':'https://www.walmart.com/','tags':['rug','neutral','modern']},
    {'title':'Oak Console Table','price':219,'retailer':'Walmart','url':'https://www.walmart.com/','tags':['console','oak','mid-century']},
    {'title':'Fluted Table Lamp','price':74,'retailer':'Target','url':'https://www.target.com/','tags':['lamp','modern']},
    {'title':'Rattan Accent Chair','price':199,'retailer':'Wayfair','url':'https://www.wayfair.com/','tags':['chair','rattan','coastal']},
]


def search_products(design_dna: Dict, decor_types: List[str], budget_min: int, budget_max: int) -> List[Dict]:
    wanted = set([x.lower() for x in (design_dna.get('style_tags', []) + design_dna.get('materials', []) + decor_types)])
    out = []
    for p in CATALOG:
        if budget_min <= p['price'] <= budget_max:
            overlap = len([t for t in p['tags'] if t.lower() in wanted])
            q = dict(p)
            q['match_score'] = round(overlap / max(1, len(p['tags'])), 3)
            out.append(q)
    out.sort(key=lambda x: x['match_score'], reverse=True)
    return out

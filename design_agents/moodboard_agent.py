from typing import Dict, List


def compose_moodboard(design_dna: Dict, products: List[Dict]) -> Dict:
    slots = ['Center Piece','Lighting','Textiles','Accent','Storage','Wall Decor']
    items = []
    for i, p in enumerate(products[:6]):
        items.append({
            'slot': slots[i % len(slots)],
            'title': p['title'],
            'retailer': p['retailer'],
            'price': p['price'],
            'match_score': p.get('match_score', 0.0),
            'url': p['url'],
            'placement_guidance': f"Place {p['title']} to balance the room layout.",
        })
    return {
        'design_story': f"A {', '.join(design_dna.get('style_tags', []))} board with warm natural textures.",
        'board_items': items,
    }

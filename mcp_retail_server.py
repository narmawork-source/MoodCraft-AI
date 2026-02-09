from typing import List
from mcp.server.fastmcp import FastMCP
from design_agents.retail_agent import search_products

mcp = FastMCP('retail-server')

@mcp.tool(name='search_products')
def search_products_tool(design_dna: dict, decor_types: List[str], budget_min: int, budget_max: int) -> dict:
    return {'products': search_products(design_dna, decor_types, budget_min, budget_max)}

if __name__ == '__main__':
    mcp.run('stdio')

from mcp.server.fastmcp import FastMCP
from design_agents.moodboard_agent import compose_moodboard

mcp = FastMCP('board-server')

@mcp.tool(name='compose_moodboard')
def compose_moodboard_tool(design_dna: dict, products: list) -> dict:
    return compose_moodboard(design_dna, products)

if __name__ == '__main__':
    mcp.run('stdio')

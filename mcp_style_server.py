from mcp.server.fastmcp import FastMCP
from design_agents.style_agent import analyze_style_dna

mcp = FastMCP('style-server')

@mcp.tool(name='analyze_room_images')
def analyze_room_images(user_prompt: str) -> dict:
    return analyze_style_dna(user_prompt)

if __name__ == '__main__':
    mcp.run('stdio')

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessageChunk
from typing import AsyncGenerator
from assistant.my_mcp.config import mcp_config

from assistant.graph import build_agent_graph, AgentState


async def stream_graph_response(
        input: AgentState, graph: StateGraph, config: dict = {}
        ) -> AsyncGenerator[str, None]:
    """
    Stream the response from the graph while parsing out tool calls.

    Args:
        input: The input for the graph.
        graph: The graph to run.
        config: The config to pass to the graph. Required for memory.

    Yields:
        A processed string from the graph's chunked response.
    """
    async for message_chunk, metadata in graph.astream(
        input=input,
        stream_mode="messages",
        config=config
        ):
        if isinstance(message_chunk, AIMessageChunk):
            if message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    yield "\n\n"

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]

                tool_name = tool_chunk.get("name", "")
                args = tool_chunk.get("args", "")
                
                if tool_name:
                    tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"
                if args:
                    tool_call_str = args

                yield tool_call_str
            else:
                yield message_chunk.content
            continue


async def main():
    """
    Initialize the MCP client and run the agent conversation loop.

    Uses MultiServerMCPClient with a single server config.
    """
    # Instantiate the client normally (not using async with)
    client = MultiServerMCPClient(connections=mcp_config)

    # Load tools from the server(s)
    tools = await client.get_tools()  # await because it's async
    graph = build_agent_graph(tools=tools)

    # Optional graph config for memory / thread
    graph_config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    while True:
        user_input = input("\n\nUSER: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting agent...")
            break

        #print("\n ----  USER  ---- \n\n", user_input)
        print("\n ----  ASSISTANT  ---- \n\n")

        async for response in stream_graph_response(
            input=AgentState(messages=[HumanMessage(content=user_input)]),
            graph=graph,
            config=graph_config,
            #stream=True # enable streaming of partial tokens
        ):
            print(response, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    # only needed if running in an ipykernel
    #import nest_asyncio
    #nest_asyncio.apply()

    asyncio.run(main())

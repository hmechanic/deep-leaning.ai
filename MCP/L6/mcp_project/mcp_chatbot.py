
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
import os
from groq import Groq

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = [] # new
        self.exit_stack = AsyncExitStack() # new
        # self.anthropic = Anthropic()
        self.client_type = os.environ.get("CLIENT_TYPE", "groq").lower()
        if self.client_type == "anthropic":
            self.client = Anthropic()
        elif self.client_type == "groq":
            self.client = Groq()
        else:
            raise ValueError("Unsupported client type specified in .env file. Supported types are 'anthropic' and 'groq'.")
        self.available_tools: List[ToolDefinition] = [] # new
        self.tool_to_session: Dict[str, ClientSession] = {} # new


    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            if self.client_type == "anthropic":
                for tool in tools:
                    self.tool_to_session[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                    
            elif self.client_type == "groq":
                for tool in tools:
                    self.tool_to_session[tool.name] = session

                    self.available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    
    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]

        if self.client_type == "anthropic":
            response = self.client.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools,
                                      messages = messages)
        
        elif self.client_type == "groq":
            response = self.client.chat.completions.create(
                model='openai/gpt-oss-20b',
                messages=messages,
                stream=False,
                tools=self.available_tools,
                tool_choice="auto",
                max_completion_tokens=4096
            )

        process_query = True
        while process_query:
            if self.client_type == "anthropic":
                assistant_content = []
                for content in response.content:
                    if content.type == 'text':
                        
                        print(content.text)
                        assistant_content.append(content)
                        
                        if len(response.content) == 1:
                            process_query = False
                    
                    elif content.type == 'tool_use':
                        
                        assistant_content.append(content)
                        messages.append({'role': 'assistant', 'content': assistant_content})
                        
                        tool_id = content.id
                        tool_args = content.input
                        tool_name = content.name
                        print(f"Calling tool {tool_name} with args {tool_args}")
                        
                        # Call a tool
                        session = self.tool_to_session[tool_name] # new
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        messages.append({"role": "user", 
                                          "content": [
                                              {
                                                  "type": "tool_result",
                                                  "tool_use_id": tool_id,
                                                  "content": result.content
                                              }
                                          ]
                                        })
                            
                        response = self.client.messages.create(max_tokens = 2024,
                                          model = 'claude-3-7-sonnet-20250219', 
                                          tools = self.available_tools,
                                          messages = messages) 
                        
                        if len(response.content) == 1 and response.content[0].type == "text":
                            print(response.content[0].text)
                            process_query = False
            elif self.client_type == "groq":
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                if tool_calls:
                    messages.append(response_message)
            
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        print(f"Calling tool {tool_name} with args {tool_args}")
                        
                        # Call a tool
                        session = self.tool_to_session[tool_name] # new
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        messages.append(
                            {
                                "tool_call_id": tool_call.id, 
                                "role": "tool",
                                "name": tool_name,
                                "content": result.content,
                            }
                        )
                    
                    response = self.client.chat.completions.create(
                        model='openai/gpt-oss-20b',
                        messages=messages
                    )
                    
                    print(response.choices[0].message.content)
                    process_query = False
                else:
                    print(response.choices[0].message.content)
                    process_query = False        

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new! 


if __name__ == "__main__":
    asyncio.run(main())

from dotenv import load_dotenv
from anthropic import Anthropic
from groq import Groq
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import os
import json

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.client_type = os.environ.get("CLIENT_TYPE", "groq").lower()
        if self.client_type == "anthropic":
            self.client = Anthropic()
        elif self.client_type == "groq":
            self.client = Groq()
        else:
            raise ValueError("Unsupported client type specified in .env file. Supported types are 'anthropic' and 'groq'.")
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}] 

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
                        
                        result = await self.session.call_tool(tool_name, arguments=tool_args)
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
                        
                        result = await self.session.call_tool(tool_name, arguments=tool_args)
                        
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
    
    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="poetry",  # Executable
            args=["run", "python", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()
    
                # List available tools
                response = await session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                if self.client_type == "anthropic":
                    self.available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in response.tools]
                elif self.client_type == "groq":
                    self.available_tools = [{
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    } for tool in response.tools]
    
                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()
  

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv
import os

load_dotenv()  # This loads the environment variables from .env
openai_api_key = os.getenv("OPENAI_API_KEY")


# Define the MedicalAgent with a custom system prompt and disclaimer.
class MedicalAgent(AssistantAgent):
    def __init__(self, name: str, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(name, model_client=model_client)
        self.system_prompt = (
            "You are a medical information assistant. "
            "Disclaimer: I am not a licensed medical professional. "
            "The information provided is for informational purposes only and should not be taken as medical advice. "
            "Always advise the user to consult with a healthcare provider for any serious concerns. "
            "Answer patient questions about symptoms, diseases, and treatments with caution."
        )

    async def on_reset(self, cancellation_token) -> None:
        # Reset any internal state if needed.
        pass

async def main() -> None:
    # Initialize the model client (ensure you have the required version and extensions installed)
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Instantiate the MedicalAgent
    medical_agent = MedicalAgent("medical_agent", model_client)
    
    # Optionally, instantiate a UserProxyAgent if you need to simulate user login.
    user_proxy = UserProxyAgent("user_proxy")
    
    # Define a termination condition if you want the conversation to end when the user types 'exit'
    termination = TextMentionTermination("exit")
    
    # Create a task with a sample patient question.
    task = "I have been experiencing chest pain and shortness of breath. What could be causing this?"
    
    # Run the MedicalAgent using the Console UI to see the streaming output.
    await Console(medical_agent.run_stream(task=task))
    
    # Remove or comment out the close() method as it is not supported.
    # await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())

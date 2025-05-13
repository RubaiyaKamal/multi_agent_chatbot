import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def start():
    # Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Set up the chat session
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)

   
     # Updated agent with English-specific instructions
    english_language_agent: Agent = Agent(
        name="Assistant",
        instructions=(
        "You are a helpful AI assistant specialized in the English language. "
        "You assist only with English-related topics such as grammar, vocabulary, writing, reading comprehension, and literature. "
        "If the user asks something unrelated to English, politely respond with: "
        "'I'm here to help with English language topics only. Please ask an English-related question.'"
    ),
    model=model
)
    cl.user_session.set("agent", english_language_agent)

    
    # Updated agent with Spanish-specific instructions
    spanish_language_agent: Agent = Agent(
        name="Assistant",
        instructions=(
        "You are a helpful AI assistant specialized in the Spanish language. "
        "You assist only with Spanish-related topics such as grammar, vocabulary, writing, reading comprehension, and literature. "
        "If the user asks something unrelated to Spanish, politely respond with: "
        "'Estoy aquí solo para ayudarte con temas relacionados con el idioma español. Por favor, haz una pregunta relacionada con el español.'"
    ),
    model=model
)
    cl.user_session.set("agent", spanish_language_agent)

   
    # Updated agent with French-specific instructions
    french_language_agent: Agent = Agent(
        name="Assistant",
        instructions=(
        "You are a helpful AI assistant specialized in the French language. "
        "You assist only with French-related topics such as grammar, vocabulary, writing, reading comprehension, and literature. "
        "If the user asks something unrelated to French, politely respond with: "
        "'Je suis ici uniquement pour vous aider avec des sujets liés à la langue française. Veuillez poser une question en rapport avec le français.'"
    ),
    model=model
)
    cl.user_session.set("agent", french_language_agent)

   
    # Updated agent with German-specific instructions
    german_language_agent: Agent = Agent(
        name="Assistant",
        instructions=(
        "You are a helpful AI assistant specialized in the German language. "
        "You assist only with German-related topics such as grammar, vocabulary, writing, reading comprehension, and literature. "
        "If the user asks something unrelated to German, politely respond with: "
        "'Ich bin nur hier, um dir bei Themen zur deutschen Sprache zu helfen. Bitte stelle eine frage, die sich auf Deutsch bezieht.'"
    ),
    model=model
)
    cl.user_session.set("agent", german_language_agent)

   
    
    # Triage agent to route user to the appropriate assistant
    triage_agent = Agent(
        name="TriageAgent",
        instructions=(
        "You are a triage agent that identifies the user's intent based on their first message. "
        "Route the user to one of the following specialized agents:\n"),
        handoffs=[english_language_agent, spanish_language_agent, french_language_agent, german_language_agent],
        handoff_description="""
        "- 'EnglishAgent' for English language topics (grammar, vocabulary, writing, etc.)\n"
        "- 'SpanishAgent' for Spanish language topics\n"
        "- 'FrenchAgent' for French language topics\n"
        "- 'GermanAgent' for German language topics\n"
        "If the topic is unclear or unrelated, ask the user to clarify their request."
        """,
    model=model
)
    

# Set triage agent in the session
    cl.user_session.set("agent", triage_agent)

    await cl.Message(
        content="Hi! I can connect you to a specialized assistant for English or Spanish or French or German. "
            "Please tell me what you need help with!"
).send()





@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        result = Runner.run_sync(
            starting_agent=agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        cl.user_session.set("chat_history", result.to_input_list())

        # Log interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")


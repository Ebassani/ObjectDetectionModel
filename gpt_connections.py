from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
from util import list_to_string
from dotenv import load_dotenv


class GPTAssistant:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        self.assistant = self.client.beta.assistants.create(
          name="Chef",
          instructions="You are a chef, you provide recipes based on the ingredients provided by the user. You also "
                       "modify it according to user preference. Use some or all of the ingredients provided."
                       "Give a name for the recipe",
          model="gpt-3.5-turbo",
        )
        self.thread = self.client.beta.threads.create()

    def ask_for_recipe(self, ingredients):
        ingredients = ''.join(list_to_string(ingredients))
        instructions = "Use the following ingredients: " + ingredients
        print(instructions)

        with self.client.beta.threads.runs.stream(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions,
                event_handler=EventHandler(),
        ) as stream:
            stream.until_done()


class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nAssistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nAssistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

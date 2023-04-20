import configparser
import json
import os
import requests
import tiktoken
import time
from datetime import datetime

# Retrieve configuration data
if os.path.exists('config.ini'):
    config = configparser.ConfigParser()
    config.read('config.ini')
    API_KEY = config.get('API_KEYS', 'OPENAI_API_KEY')
else:
    API_KEY = 'api_key_here'

API_MODEL = 'gpt-3.5-turbo'
API_TOKEN_LIMIT = 4096 # total number of tokens (sent and received) allowed by the API endpoint for a single call
API_URL = 'https://api.openai.com/v1/chat/completions'
MAX_TOKENS = 800 # maximum number of tokens to be returned from API endpoint
SUMMARY_PROMPT = 'Summarize this conversation so far.' # Prompt used to query API for a summary of current conversation for pruning purposes
TEMPERATURE = 1 # 0 to 2, the lower the number the more deterministic the response from the API

class ChatCompletionApi:
    """ manages api calls

    message_list: json list of messages in form [ { role: content }, { role: content }, ... ]
    role: role of speaker in relation to response, one of "system" (chatbot commands), "user", "assistant" (the chatbot)
    """
    @classmethod
    def get_response(cls, message_list, role="user"):
        headers =  {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        data = {
            'messages': message_list,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
            'model': API_MODEL
        }
        role = role

        while True:
            response = requests.post(API_URL, headers=headers, json=data)

            if response.status_code == 200:
                content = response.json()
                completion = content['choices'][0]['message']['content'].strip()
                if not isinstance(completion, str):
                    print("Error: completion response is not a string. Retrying in 1 second...")
                    time.sleep(1)
                    continue
                return ChatMessage(role, completion)
            elif response.status_code == 429:
                print("Rate limit exceeded. Retrying in 1 second...")
                time.sleep(1)
                continue
            else:
                print("Error:", response.status_code, response.reason, "\n")
                print("Response:", response.json())
                return None

class ChatManager:

    def __init__(self, personality):
        self.context = Context(personality)
        self.history = [ personality.get_personality_prompt() ]

    def get_history(self):
        return [ m.to_dict() for m in self.history ]

    def reset_chat(self):
        self.context.reset_context()
        del self.history[1:]

    def submit_prompt(self, prompt):
        """ submits a ChatMessage prompt and returns a ChatMessage response and updates history"""
        json_context = self.context.get_prepared_context(prompt)
        response = ChatCompletionApi.get_response(json_context, "assistant")
        self.update_history(prompt, response)
        return response

    def update_history(self, prompt, response):
        self.history.append(prompt)
        self.history.append(response)

class ChatMessage:
    """A message in a chatbot conversation

    role: speaker, one of "system" (chatbot commands), "user", "assistant" (the chatbot)
    content: the message given, typically the user's prompt or the chatbot's completion
    """

    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.creation_time = datetime.now()
        self.tokenlen = self.get_tokenlen()

    def get_content(self):
        return self.content

    def get_creation_time(self):
        return self.creation_time.strftime("%H:%M:%S, %d/%m/%Y")

    def to_dict(self):
        return { "role": self.role, "content": self.content }

    def get_role(self):
        return self.role

    def get_tokenlen(self):
        encoding = tiktoken.encoding_for_model(API_MODEL)
        return len(encoding.encode(self.to_json()))

    def to_json(self):
        return json.dumps( { "role": self.role, "content": self.content } )

    def to_string(self):
        return f"{self.to_json()}, {self.get_creation_time()}"

class Context:
    """ stores and manages historical chat context, providing conversational memory for API calls """

    def __init__(self, personality):
        self.context_list = [ personality.get_personality_prompt() ]
        self.summary_message = ChatMessage("user", SUMMARY_PROMPT)

    def get_context_tokenlen(self):
        tokenlen = sum([m.get_tokenlen() for m in self.context_list])
        return tokenlen

    def get_prepared_context(self, prompt):
        """ takes prompt ChatMessage and appends it to the context list, pruning if necessary, returns JSON of list """
        while (self.get_context_tokenlen() + prompt.get_tokenlen() + MAX_TOKENS + self.summary_message.get_tokenlen()) > API_TOKEN_LIMIT:
            self.prune_context()
        self.context_list.append(prompt)
        prepared_context = [ m.to_dict() for m in self.context_list ]
        return prepared_context

    def prune_context(self):
        context_summary = self.summarize_context()
        del self.context_list[1:9]
        self.context_list.insert(1, context_summary)

    def reset_context(self):
        del self.context_list[1:]

    def summarize_context(self):
        summary_list = self.context_list.copy()
        summary_list.append(self.summary_message)
        prepared_summary_list = json.dumps([ m.to_dict() for m in summary_list ])
        return ChatCompletionApi.get_response(prepared_summary_list)

class Personality:
    """The personality of a chatbot

    system_prompt: This is the instruction to tell the API how to flavor of how to respond.
    For example, it might say, "You are an sad assistant. Punctuate your conversation with crying and sad observations."
    """

    def __init__(self, name, description, system_prompt):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt

    def get_description(self):
        return self.description

    def get_name(self):
        return self.name

    def get_personality_prompt(self):
        return ChatMessage("system", self.system_prompt)

    def get_system_prompt(self):
        return self.system_prompt

    def set_description(self, description):
        self.description = description

    def set_name(self, name):
        self.name = name

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

class PersonalityManager:

    def __init__(self):
        self.personality_list = self.build_default_personalities()

    def add_personality(self, personality):
        self.personality_list.append(personality)

    def build_default_personalities(self):
        personalities = [
            Personality("Helpful", "A helpful assistant", "You are a helpful assistant."),
            Personality("Comedian", "A stand-up comedian buddy", "You are a professional stand-up commedian, but having a conversation with a friend. You commonly use bits of novel comedy routines conversationally."),
            Personality("Computer Expert", "An expert in computers and troubleshooting", "You are a computer expert, skilled in troubleshooting and problem-solving in the IT domain."),
            Personality("Snarky", "A sarcastic AI", "You are a sarcastic assistant, responding with witty and sardonic remarks."),
            Personality("Counselor", "An emotional well-being counselor", "You are a counselor, offering guidance and support to those seeking emotional wellness and personal growth. You are an advocate, and gently shine light where needed for growth.")
        ]
        return personalities

    def get_default_personality(self):
        return self.personality_list[0]

    def get_personalities_list(self):
        return self.personality_list.copy()

    def get_personality(self, personality_name):
        for p in self.personality_list:
            if p.get_name() == personality_name:
                return p

    def remove_personality(self, personality_name):
        self.personality_list.remove(self.get_personality(personality_name))

    def set_personality_description(self, personality_name, new_description):
        self.get_personality(personality_name).set_description(new_description)

    def set_personality_name(self, personality_name, new_name):
        self.get_personality(personality_name).set_name(new_name)

    def set_personality_system_prompt(self, personality_name, new_prompt):
        self.get_personality(personality_name).set_system_prompt(new_prompt)

def main():
    personality_manager = PersonalityManager()
    chat_manager = ChatManager(personality_manager.get_default_personality())
    
    print("Use command #Personality to choose different personality.")

    while True:
        user_prompt = input("User: ")
        
        if user_prompt.lower() == "quit":
            break
        if user_prompt.lower() == "#personality":
            i = 0
            for p in personality_manager.get_personalities_list():
                i += 1
                print(f"{i}. {p.get_name()}")
            print(f"{i + 1}. Custom")
            personality_select = input("Select number: ")
            try:
                personality_int = (int(personality_select) - 1)
            except ValueError:
                print("Invalid selection.")
                continue

            if personality_int == (i):
                personality_name = input("Enter personality name: ")
                personality_prompt = input("Write an instructional prompt for the chatbot's personality: ")
                personality_manager.add_personality(Personality(personality_name, "Custom", personality_prompt))
            elif personality_int > (i):
                print("Invalid selection.")
                continue
            chat_manager = ChatManager(personality_manager.get_personalities_list()[personality_int])
            continue
        elif user_prompt.lower() == "#history":
            print(json.dumps(chat_manager.get_history()))
        else:
            user_prompt_message = ChatMessage("user", user_prompt)
            response_message = chat_manager.submit_prompt(user_prompt_message)
        
            response = response_message.get_content()
            print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    main()
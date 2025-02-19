from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class Message:
    type: str
    content: str

    def serialize(self):
        return {
            'type': self.type,
            'content': self.content
        }


class ConfigureApiModelStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Configures an API model with connection details and parameters'

    def verify_executable(self, entity, strategy_request):
        config = strategy_request.param_config
        required = ['model_type', 'env_key']
        return all(k in config for k in required)

    def apply(self, entity) -> StrategyRequestEntity:
        config = self.strategy_request.param_config
        
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv(config['env_key'])
        if not api_key:
            raise ValueError(f"API key not found in environment for key: {config['env_key']}")

        # Set model type
        model_type = config.get('model_type', 'openai')
        entity.set_attribute('model_type', model_type)
        
        # Set model name with defaults based on type
        model_name = config.get('model_name')
        if not model_name:
            if model_type == 'openai':
                model_name = 'gpt-4o-mini'
        entity.set_attribute('model_name', model_name)
        
        # Set API key
        entity.set_attribute('api_key', api_key)
        
        # Set config with OpenAI defaults if not provided
        default_config = {
            'temperature': 0.7,
            'max_tokens': 5000,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stream': False
        }
        
        if 'model_config' in config:
            default_config.update(config['model_config'])
        
        entity.set_attribute('config', default_config)
            
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'model_type': 'openai',  # Optional, defaults to openai
            'model_name': 'gpt-4o-mini',  # Optional, has defaults per model_type
            'env_key': 'OPENAI_API_KEY',  # Required - environment variable key
            'model_config': {  # Optional - has defaults
                'temperature': 0.7,
                'max_tokens': 4000,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'stream': False
            }
        }
    

class CallApiModelStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Makes a call to the configured API model using LangChain'

    def verify_executable(self, entity, strategy_request):
        required_attrs = ['model_type', 'model_name', 'api_key']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def form_context(self, entity) -> str:
        """Form context from document children"""
        doc_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.DOCUMENT)

        history = entity.get_attribute('message_history')
        
        if not doc_ids:
            return ""
            
        contexts = []
        for doc_id in doc_ids:
            doc = self.entity_service.get_entity(doc_id)
            if doc and doc.has_attribute('text'):
                doc_type = doc.get_document_type() or 'unknown'
                if doc.has_attribute('path'):
                    doc_type = f"{doc_type} ({doc.get_attribute('path')})"
                if doc.has_attribute('name'):
                    doc_type = f"{doc_type} - {doc.get_attribute('name')}"
                contexts.append(
                    f"{'='*50}\n"
                    f"Document Type and Name and Path : {doc_type.upper()}\n"
                    f"{'-'*50,'DOCUMENT_BEGIN'}\n"
                    f"{doc.get_text()}\n"
                    f"{'='*50,"DOCUMENT_END"}\n"
                )

        for message in history:
            contexts.append(
                f"{'='*50}\n"
                f"Message Type: {message.type.upper()}\n"
                f"{'-'*50}\n"
                f"{message.content}\n"
                f"{'='*50}\n"
            )
        
        return "\n".join(contexts)

    def format_response_text(self, text: str, max_line_length: int = 80) -> str:
        """Format response text with proper line breaks"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word exceeds max line length
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                # Start new line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        # Add final line
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)

    def apply(self, entity) -> StrategyRequestEntity:
        # Get context from documents
        context = self.form_context(entity)
        
        # Get user input and system prompt from config
        config = self.strategy_request.param_config
        user_input = config.get('user_input', '')
        system_prompt = config.get('system_prompt', '')
        context_prefix = config.get('context_prefix', 'Here is the relevant context:')
        
        # Combine context with user input
        if context:
            combined_input = (
                f"{context_prefix}\n\n"
                f"{context}\n\n"
                f"{'='*50}\n"
                f"USER QUERY:\n"
                f"{'-'*50}\n"
                f"{user_input}\n"
                f"{'='*50}"
            )
        else:
            combined_input = user_input

        # Initialize LangChain chat model
        chat = ChatOpenAI(
            model_name=entity.get_attribute('model_name'),
            openai_api_key=entity.get_attribute('api_key'),
            temperature=entity.get_attribute('config').get('temperature', 0.7),
            max_tokens=entity.get_attribute('config').get('max_tokens', 1000)
        )

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=combined_input))

        # Use invoke() instead of __call__
        response = chat.invoke(messages)
        
        # Get message history
        history = entity.get_attribute('message_history')
        
        # Format complete response with history
        
        # Update history and attributes
        history.append(Message(type='context', content=user_input))
        history.append(Message(type='response', content=response.content))
        
        entity.set_attribute('last_context', context)
        entity.set_attribute('message_history', history)
        entity.set_attribute('response', [message.serialize() for message in history])
        
        self.strategy_request.ret_val['response'] = history
        self.strategy_request.ret_val['context_used'] = context
        
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'user_input': '',  # Optional additional input
            'system_prompt': '',  # Optional system prompt
            'context_prefix': 'Here is the relevant context:',  # Optional prefix for context
        }

class CallDeepSeekApiStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Makes a call to the DeepSeek API using the OpenAI SDK'

    def verify_executable(self, entity, strategy_request):
        required_attrs = ['model_type', 'model_name', 'api_key']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def form_context(self, entity) -> str:
        """Form context from document children"""
        doc_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.DOCUMENT)

        history = entity.get_attribute('message_history')

        if not doc_ids:
            return ""

        contexts = []
        for doc_id in doc_ids:
            doc = self.entity_service.get_entity(doc_id)
            if doc and doc.has_attribute('text'):
                doc_type = doc.get_document_type() or 'unknown'
                if doc.has_attribute('path'):
                    doc_type = f"{doc_type} ({doc.get_attribute('path')})"
                if doc.has_attribute('name'):
                    doc_type = f"{doc_type} - {doc.get_attribute('name')}"
                contexts.append(
                    f"{'=' * 50}\n"
                    f"Document Type and Name and Path : {doc_type.upper()}\n"
                    f"{'-' * 50, 'DOCUMENT_BEGIN'}\n"
                    f"{doc.get_text()}\n"
                    f"{'=' * 50, "DOCUMENT_END"}\n"
                )

        for message in history:
            contexts.append(
                f"{'=' * 50}\n"
                f"Message Type: {message.type.upper()}\n"
                f"{'-' * 50}\n"
                f"{message.content}\n"
                f"{'=' * 50}\n"
            )

        return "\n".join(contexts)

    def format_response_text(self, text: str, max_line_length: int = 80) -> str:
        """Format response text with proper line breaks"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            # Check if adding this word exceeds max line length
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                # Start new line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        # Add final line
        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def apply(self, entity) -> StrategyRequestEntity:
        # Get context from documents
        context = self.form_context(entity)

        # Get user input and system prompt from config
        config = self.strategy_request.param_config
        user_input = config.get('user_input', '')
        system_prompt = config.get('system_prompt', '')
        context_prefix = config.get('context_prefix', 'Here is the relevant context:')

        # Combine context with user input
        if context:
            combined_input = (
                f"{context_prefix}\n\n"
                f"{context}\n\n"
                f"{'=' * 50}\n"
                f"USER QUERY:\n"
                f"{'-' * 50}\n"
                f"{user_input}\n"
                f"{'=' * 50}"
            )
        else:
            combined_input = user_input

        # Initialize the OpenAI client for DeepSeek
        client = OpenAI(
            api_key=entity.get_attribute('api_key'),
            base_url="https://api.deepseek.com"
        )

        # Prepare the messages for the API call
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": combined_input})

        # Make the API call to DeepSeek
        response = client.chat.completions.create(
            model=entity.get_attribute('model_name'),
            messages=messages,
            temperature=entity.get_attribute('config').get('temperature', 0.7),
            max_tokens=entity.get_attribute('config').get('max_tokens', 1000),
            top_p=entity.get_attribute('config').get('top_p', 1.0),
            frequency_penalty=entity.get_attribute('config').get('frequency_penalty', 0.0),
            presence_penalty=entity.get_attribute('config').get('presence_penalty', 0.0),
            stream=entity.get_attribute('config').get('stream', False)
        )

        # Extract the response content
        response_content = response.choices[0].message.content

        # Get message history
        history = entity.get_attribute('message_history')

        # Update history and attributes
        history.append(Message(type='context', content=user_input))
        history.append(Message(type='response', content=response_content))

        entity.set_attribute('last_context', context)
        entity.set_attribute('message_history', history)
        entity.set_attribute('response', [message.serialize() for message in history])

        self.strategy_request.ret_val['response'] = history
        self.strategy_request.ret_val['context_used'] = context

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'user_input': '',  # Optional additional input
            'system_prompt': '',  # Optional system prompt
            'context_prefix': 'Here is the relevant context:',  # Optional prefix for context
        }

class GenerateEmbeddingsStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Generates embeddings from input text using OpenAI\'s text-embedding-3-small model'

    def verify_executable(self, entity, strategy_request):
        """
        Verify that the required attributes and configurations are present.
        """
        required_attrs = ['api_key', 'model_name']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def apply(self, entity) -> StrategyRequestEntity:
        """
        Generate embeddings for the input text using the text-embedding-3-small model.
        """
        # Load environment variables
        load_dotenv()

        # Get API key from entity or environment
        api_key = entity.get_attribute('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please provide an API key.")

        # Get input text from strategy request
        config = self.strategy_request.param_config
        input_text = config.get('input_text')
        if not input_text:
            input_text = "Convert the following text into a numerical vector representation that captures its semantic meaning: "
            # raise ValueError("Input text is required to generate embeddings.")

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Generate embeddings
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Use the text-embedding-3-small model
            input=input_text
        )

        # Extract the embedding vector
        embedding = response.data[0].embedding

        # Store the embedding in the entity
        entity.set_attribute('embedding', str(embedding))

        # Update the strategy request with the result
        self.strategy_request.ret_val['embedding'] = str(embedding)
        self.strategy_request.ret_val['input_text'] = input_text

        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Define the required configuration for this strategy.
        """
        return {
            'input_text': '',  # Required: The text to generate embeddings for
        }
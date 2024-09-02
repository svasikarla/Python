import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below 

Here is the conversation history : {context}

Qestion: {question}

Answer:
"""

model =OllamaLLM(model="gemma2:2B")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt|model

class ConversationModel:
    def __init__(self):
        self.model = self.load_model()
        self.conversation_history = []

    def load_model(self):
        # Replace this with your actual model loading code
        model =OllamaLLM(model="gemma2:2B")
        return model

    def predict(self, user_input):
        # Append the user input to the conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Combine the conversation history into a single input string
        conversation_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.conversation_history])

        # Generate a response from the model (replace this with the actual model call)
        # For example, with a hypothetical model's generate function:
        # response = self.model.generate(conversation_text)
        response = chain.invoke({"context": conversation_text, "question": user_input})
        # response = f"Model response based on the context:\n{conversation_text}"

        # Append the model's response to the conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        # Return the updated conversation
        conversation_display = "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.conversation_history])
        return conversation_display

    def reset_conversation(self):
        self.conversation_history = []
        return "Conversation reset. Start a new conversation!"

# Create an instance of the ConversationModel class
model_instance = ConversationModel()

# Define the interface layout
with gr.Blocks() as interface:
    # Input and output fields
    user_input = gr.Textbox(label="Input")
    output = gr.Textbox(label="Conversation", lines=10)
    
    # Buttons for generating response and resetting conversation
    submit_button = gr.Button("Generate Response")
    clear_button = gr.Button("Clear Conversation")
    
    # Define interactions
    submit_button.click(fn=model_instance.predict, inputs=user_input, outputs=output)
    clear_button.click(fn=model_instance.reset_conversation, inputs=None, outputs=output)

# Launch the interface
interface.launch(server_name="127.0.0.1", server_port=7869)

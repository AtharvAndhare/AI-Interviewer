import streamlit as st
from huggingface_hub import InferenceClient
import json



# Set up the Hugging Face
repo_id = "microsoft/Phi-3-mini-4k-instruct" #AI MODEL

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

# Function to call the LLM with a given prompt
def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

# Function to generate technical interview questions
def generate_questions(llm_client, tech_stack):
    prompt = f"Generate 5 distinct and detailed technical interview questions for a candidate with skills in {tech_stack}. Number the questions from 1 to 5."
    
    questions = []
    max_retries = 3  # Limiting retries to prevent infinite loops
    retries = 0

    while len(questions) < 5 and retries < max_retries:
        response = call_llm(llm_client, prompt)
        lines = response.strip().split("\n")
        
        # Extract numbered lines (ensure they start with a digit)
        new_questions = [line for line in lines if line.strip() and line[0].isdigit()]
        
        # Add only unique questions to avoid duplicates
        for q in new_questions:
            if q not in questions:
                questions.append(q)
                if len(questions) == 5:
                    break  # Stop if we have 5 questions
        
        retries += 1
    
    # If still fewer than 5 questions, pad with default generic questions
    while len(questions) < 5:
        questions.append(f"Placeholder question {len(questions) + 1}: Please discuss your experience with {tech_stack}.")
    
    return questions[:5]

# Function to generate AI responses to candidate answers
def generate_response(llm_client, question, answer):
    # Refined, clear prompt to reduce prompt echoing
    prompt = f"""
You are an interviewer. Provide a brief, constructive feedback on the following answer:
Answer: '{answer}'
Focus only on clarity, relevance, and communication, and do not repeat the answer or any instructions.
"""

    response = call_llm(llm_client, prompt).strip()
    
    # Post-process response: remove any repeated prompts or instructions
    feedback_lines = [
        line for line in response.splitlines() 
        if "Answer" not in line and "Provide" not in line and "Focus" not in line
    ]
    
    final_feedback = "\n".join(feedback_lines[:5])  # Limit to 5 lines max
    
    # Ensure feedback is non-empty
    return final_feedback if final_feedback else "No valid feedback provided."


# Streamlit function to greet the user
def greet_user():
    st.title("AI Interviewer")
    st.markdown("Welcome to the AI Interviewer. Let's start by gathering some basic information.")

# collect user details (st for streamlit UI)
def get_user_details():
    details = {}
    details["name"] = st.text_input("Please enter your name:")
    details["email"] = st.text_input("Please enter your email:")
    details["phone"] = st.text_input("Please enter your phone number:")
    details["experience"] = st.text_input("How many years of experience do you have?")
    details["desired_position"] = st.text_input("What position are you applying for?")
    details["tech_stack"] = st.text_input("Please list your tech stack (comma-separated):")
   
    return details

#Ask the questions and provide feedback
def ask_questions(questions):
    # Initialize the session state for the current question
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.feedbacks = []
        st.session_state.answers = []

    # Display all the answered questions and feedback so far
    for idx in range(st.session_state.question_index):
        st.write(f"**Q{idx + 1}:** {questions[idx]}")
        st.write(f"**Your Answer:** {st.session_state.answers[idx]}")
        st.write(f"**AI Interviewer Feedback:** {st.session_state.feedbacks[idx]}")
        st.write("---")

    # Get the current question and the answer from the user
    if st.session_state.question_index < len(questions):
        question = questions[st.session_state.question_index]
        st.write(f"**Q{st.session_state.question_index + 1}:** {question}")

        answer = st.text_area(f"Your Answer for Q{st.session_state.question_index + 1}:", key=f"answer_{st.session_state.question_index}")
        
        if st.button("Submit Answer", key=f"submit_{st.session_state.question_index}"):

            # Ensure the answer is provided
            if answer:
                # Generate feedback for the answer
                feedback = generate_response(llm_client, question, answer)
                st.session_state.answers.append(answer)
                st.session_state.feedbacks.append(feedback)

                st.write(f"**AI Interviewer Feedback:** {feedback}")

                # Move to the next question after submitting
                if st.session_state.question_index < len(questions) - 1:
                    
                    st.session_state.question_index += 1
                else:
                    st.write("\nAI Interviewer: Thank you for your time! If we like your profile, you will receive an email response from us.")
            else:
                st.warning("Please provide an answer before submitting.")

    else:
        st.write("\nAI Interviewer: Thank you for your time! If we like your profile, you will receive an email response from us.")

# Main function to run the AI Interviewer
def main():
    greet_user()

    details = get_user_details()
    
    if details["tech_stack"]:
        tech_stack = details["tech_stack"]
        questions = generate_questions(llm_client, tech_stack)
        ask_questions(questions)
    else:
        st.warning("Please provide your tech stack to get interview questions.")

# Run the main function
if __name__ == "__main__":
    main()

from flask import Flask, render_template, request

app = Flask(__name__)

# Claim Extraction Module

def claim_extraction_module(user_input: str) -> str:
    """
    MVP: Extracts a primary claim from user input.
    In a full system, this would involve sophisticated NLP.
    """
    print(f"[DEBUG] ClaimExtractionModule: Input '{user_input}'")
    return user_input.strip()

# Question Generation Module

def question_generation_module(claim: str) -> str:
    """
    MVP: Generates a Socratic Question to verify the claim externally.
    Uses simple pattern matching for demonstration.
    """
    print(f"[DEBUG] QuestionGenerationModule: Generating SQ for '{claim}'")
    claim_lower = claim.lower()
    if "is the capital of" in claim_lower:
        parts = claim_lower.split(" is the capital of ")
        country = parts[1].strip().replace(".", "")
        return f"What is the official capital of {country}?"
    elif "is in" in claim_lower:
        parts = claim_lower.split(" is in ")
        thing = parts[0].strip()
        return f"Where is {thing} actually located?"
    elif "is" in claim_lower:
        parts = claim_lower.split(" is ")
        subject = parts[0].strip()
        predicate = parts[1].strip().replace(".", "")
        return f"Could you provide independent verification for the statement: '{subject} is {predicate}'?"
    return f"Can you confirm the fact: '{claim}'?"

# Question Answerer Module (Mock external source)

def Youtubeer_module(question: str) -> str:
    """
    MVP: Simulates an external knowledge source (e.g., RAG, web search).
    Hardcoded knowledge base for demonstration.
    """
    print(f"[DEBUG] QuestionAnswererModule: Querying external source with '{question}'")
    knowledge_base = {
        "what is the official capital of france?": "Paris",
        "where is the eiffel tower actually located?": "Paris, France",
        "what is the official capital of germany?": "Berlin",
        "what is the official capital of italy?": "Rome",
        "where is the statue of liberty actually located?": "New York City, USA",
        "what is the color of the sky?": "The sky is typically blue due to Rayleigh scattering.",
        "can you confirm the fact: 'water boils at 100 degrees celsius'?": "Yes, water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
        "could you provide independent verification for the statement: 'the sky is green'?": "The sky is typically blue due to Rayleigh scattering."
    }
    return knowledge_base.get(question.lower().strip(), "Information not found in my current knowledge base.")

# Clarification Module

def clarification_module(user_claim: str, external_fact: str) -> str:
    """
    MVP: Generates a clarifying question for the user if a contradiction is found.
    """
    print(f"[DEBUG] ClarificationModule: Crafting clarification for user '{user_claim}', external '{external_fact}'")
    return (f"It seems your claim: '{user_claim}' contradicts information from external sources: '{external_fact}'. "
            f"Could you please clarify your statement or provide more context?")

# Socrates Agent Core Logic

def socrates_agent_process(user_input: str) -> dict:
    print("\n--- Socrates Agent Initiated ---")
    print(f"User Input: '{user_input}'")

    user_claim = claim_extraction_module(user_input)
    print(f"Detected User Claim: '{user_claim}'")

    print("\n--- Performing External Factuality Check ---")
    socratic_question = question_generation_module(user_claim)
    external_answer = Youtubeer_module(socratic_question)

    print(f"External Source Answer: '{external_answer}'")

    is_consistent = True
    feedback = ""
    clarification_needed = None

    user_claim_lower = user_claim.lower()
    external_answer_lower = external_answer.lower()

    if "information not found" in external_answer_lower:
        is_consistent = True
        feedback = "External information not found for this claim. Cannot perform a full contradiction check."
    elif (("eiffel tower" in user_claim_lower and "rome" in user_claim_lower) and
          ("paris" in external_answer_lower)):
        is_consistent = False
        feedback = "Contradiction detected: Eiffel Tower location."
        clarification_needed = clarification_module(user_claim, external_answer)
    elif ("sky is green" in user_claim_lower and "blue" in external_answer_lower):
        is_consistent = False
        feedback = "Contradiction detected: Sky color."
        clarification_needed = clarification_module(user_claim, external_answer)
    elif ("berlin is the capital of france" in user_claim_lower and "paris" in external_answer_lower):
        is_consistent = False
        feedback = "Contradiction detected: Capital of France."
        clarification_needed = clarification_module(user_claim, external_answer)
    elif user_claim_lower == external_answer_lower:
        is_consistent = True
        feedback = "User claim is consistent with external facts (direct match)."
    else:
        if not is_consistent and clarification_needed is None:
            if "yes, water boils at 100 degrees celsius" in external_answer_lower and "water boils at 100 degrees celsius" in user_claim_lower:
                is_consistent = True
                feedback = "User claim is consistent with external facts."
            else:
                if external_answer_lower not in user_claim_lower and user_claim_lower not in external_answer_lower:
                    is_consistent = False
                    feedback = "Potential contradiction detected. User claim and external answer differ significantly or are not directly alignable (requires more sophisticated semantic check)."
                    clarification_needed = clarification_module(user_claim, external_answer)
                else:
                    is_consistent = True
                    feedback = "User claim is generally consistent with external facts (partial or semantic match not fully captured by simple MVP logic)."

    print(f"Consistency Check Result: {'PASS' if is_consistent else 'FAIL'}")
    print(f"Detailed Feedback: {feedback}")

    return {
        "user_input": user_input,
        "user_claim": user_claim,
        "socratic_question": socratic_question,
        "external_answer": external_answer,
        "is_consistent": is_consistent,
        "feedback": feedback,
        "clarification_needed": clarification_needed
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_fact', methods=['POST'])
def check_fact():
    user_input = request.form['user_input']
    results = socrates_agent_process(user_input)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

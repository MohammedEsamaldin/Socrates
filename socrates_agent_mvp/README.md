# Socrates Agent MVP: External Hallucination Detection

This project is a minimal demonstration of a "Socrates Agent" focused on detecting external hallucinations in user-provided claims. The agent performs a simple workflow:

1. **Claim Extraction** – Identifies the main claim from the user's input (for the MVP the entire input is treated as the claim).
2. **Socratic Question Generation** – Produces a question that could verify the claim using basic pattern matching.
3. **Mock External Fact Check** – Looks up an answer in a hardcoded knowledge base simulating an external source.
4. **Contradiction Detection** – Compares the external answer with the claim to detect contradictions.
5. **Clarification** – If a contradiction is found, suggests a clarifying question for the user.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Navigate to `http://localhost:5000` in your browser and enter a factual claim to test.

## Demonstration Scenarios

- `Paris is the capital of France.` → **PASS**
- `The Eiffel Tower is in Rome.` → **FAIL** with clarification
- `Water boils at 100 degrees Celsius.` → **PASS**
- `The sky is green.` → **FAIL** with clarification
- `The moon is made of cheese.` → Information not found (treated as PASS for MVP)

## Future Work

- Incorporate cross-alignment and self-contradiction checks.
- Add a structured knowledge graph for more robust fact checking.
- Expand the agent to detect internal hallucinations and handle complex claims.


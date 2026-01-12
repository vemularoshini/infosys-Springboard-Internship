import pandas as pd
import os
import difflib

class KnowledgeEngine:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        self.ticket_file = "tickets.txt"
        if not os.path.exists(self.ticket_file):
            open(self.ticket_file, "w").close()

    def answer_query(self, query):
        questions = self.data["question"].tolist()

        # Find closest matching question
        match = difflib.get_close_matches(
            query, questions, n=1, cutoff=0.4
        )

        if match:
            answer = self.data[
                self.data["question"] == match[0]
            ]["answer"].values[0]
            return answer

        # If no good match → AI-style response + ticket
        with open(self.ticket_file, "a") as f:
            f.write(query + "\n")

        return (
            "Thank you for your question. "
            "Our AI system has analyzed your query and "
            "forwarded it to the support team for further review. "
            "You will receive assistance shortly."
        )


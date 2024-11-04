import getpass
import os
import sys
from langchain_groq import ChatGroq
from sql_agent import SQLAgent
from pyprojroot import here
from sqlalchemy import create_engine
import pandas as pd


def load_sample_csv_data_to_db(csv_file_path: str, db_path: str, table_name: str):
    if not csv_file_path:
        print("ERROR: No path specified, exitting")
        sys.exit(1)

    df = pd.read_csv(here(csv_file_path))
    engine = create_engine(db_path)
    df.to_sql(table_name, engine, index=False)


if __name__ == "__main__":
    llm_model = "llama-3.1-8b-instant"

    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

    llm = ChatGroq(
        model=llm_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    csv_file_path = str(here("example")) + "/books_data.csv"
    db_file_path = str(here("example")) + "/sqldb.db"
    db_path = f"sqlite:///{db_file_path}"

    load_sample_csv_data_to_db(csv_file_path=csv_file_path, db_path=db_path, table_name="books")

    # Initialize the agent with your database
    agent = SQLAgent(db_file_path, llm)

    question = "What books are related to Art?"
    # Convert a natural language query to SQL
    result = agent.generate_sql(question)
    print(f"""INFO: Generated SQL query below \n {result["sql"]}""")

    llm = ChatGroq(
        model=llm_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    # Initialize the agent with your database
    agent = SQLAgent(db_file_path, llm)

    question = "What books are related to Art?"
    # Convert a natural language query to SQL
    result = agent.generate_sql(question)

    # If successful, execute the query
    if result["success"]:
        df = agent.execute_query(result["sql"])
        result = llm.invoke(f"Convert the following results in CSV format in a natural conversational language tone based on the user question: {question} and results: {df.to_csv()}")
        print()
        print("Final response in user friendly language below:")
        print()
        print(result.content)

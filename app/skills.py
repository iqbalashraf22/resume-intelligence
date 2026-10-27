import re

# Skill ka standard naam -> resume/JD mein likhne ke alag-alag tareeke (aliases).
# Yeh hamari apni chhoti taxonomy hai; naye skills yahin jodo.
SKILLS: dict[str, list[str]] = {
    # Languages
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "Go": ["Go", "Golang"],
    "C++": ["c++"],
    "C#": ["c#"],
    "SQL": ["sql"],
    "R": ["R"],
    "Bash": ["bash", "shell scripting"],
    "Scala": ["scala"],
    "Rust": ["Rust"],
    "Kotlin": ["kotlin"],
    "Swift": ["Swift"],
    "Ruby": ["Ruby"],
    "PHP": ["php"],
    "MATLAB": ["matlab"],
    # ML, data science, NLP
    "Machine learning": ["machine learning", "ml"],
    "Deep learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer vision": ["computer vision"],
    "Statistics": ["statistics", "statistical analysis"],
    "A/B testing": ["a/b testing", "a/b tests", "a/b test", "ab testing"],
    "Time series": ["time series", "time-series"],
    "Forecasting": ["forecasting"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "Keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "XGBoost": ["xgboost"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "spaCy": ["spacy"],
    "NLTK": ["nltk"],
    "Hugging Face": ["hugging face", "huggingface"],
    "LangChain": ["langchain"],
    "LlamaIndex": ["llamaindex", "llama index"],
    "OpenCV": ["opencv"],
    # LLM apps and search
    "RAG": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "LLMs": ["llm", "llms", "large language model", "large language models"],
    "Embeddings": ["embedding", "embeddings"],
    "Vector databases": ["vector database", "vector databases", "vector db", "vector store", "vector stores"],
    "Semantic search": ["semantic search"],
    # Data engineering and MLOps
    "MLflow": ["mlflow"],
    "Airflow": ["airflow"],
    "Spark": ["Spark", "PySpark", "Apache Spark"],
    "Hadoop": ["hadoop"],
    "Kafka": ["kafka"],
    "dbt": ["dbt"],
    "ETL": ["etl"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch"],
    "SQLite": ["sqlite"],
    "DynamoDB": ["dynamodb"],
    "ChromaDB": ["chromadb"],
    "FAISS": ["faiss"],
    "Pinecone": ["pinecone"],
    # Analytics and BI
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Excel": ["Excel"],
    "Looker": ["looker"],
    "Google Analytics": ["google analytics"],
    "Jira": ["jira"],
    # Cloud and DevOps
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions"],
    "Git": ["git"],
    "CI/CD": ["ci/cd", "ci cd", "continuous integration"],
    "Linux": ["linux"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    # Backend and web
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Spring Boot": ["spring boot", "springboot"],
    "React": ["React", "ReactJS", "React.js"],
    "Node.js": ["node.js", "nodejs"],
    "Microservices": ["microservice", "microservices"],
    "REST APIs": ["rest api", "rest apis", "restful api", "restful apis", "restful"],
    "GraphQL": ["graphql"],
    "Agile": ["agile", "scrum"],
}

# Jin naamon ka matlab aam shabd bhi hota hai, unhe sirf isi capital letter ke saath pehchano
CASE_SENSITIVE = {"Go", "R", "Excel", "React", "Swift", "Rust", "Ruby", "Spark"}


def _compile(name: str, aliases: list[str]) -> re.Pattern:
    """Poora shabd match karo: 'Java' 'JavaScript' mein nahi, 'Git' 'GitHub' mein nahi milega."""
    alts = "|".join(sorted((re.escape(a) for a in aliases), key=len, reverse=True))
    flags = 0 if name in CASE_SENSITIVE else re.IGNORECASE
    return re.compile(rf"(?<![\w+#])(?:{alts})(?![\w+#])", flags)


_PATTERNS = {name: _compile(name, aliases) for name, aliases in SKILLS.items()}


def has_skill(text: str, skill: str) -> bool:
    return bool(_PATTERNS[skill].search(text))


def extract_skills(text: str) -> list[str]:
    """Text mein jo skills mile unke standard naam, jis order mein text mein aaye."""
    found = []
    for name, pattern in _PATTERNS.items():
        m = pattern.search(text)
        if m:
            found.append((m.start(), name))
    return [name for _, name in sorted(found)]

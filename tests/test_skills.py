from app.skills import extract_skills, has_skill


def test_order_of_appearance():
    assert extract_skills("Experience with Docker and Python") == ["Docker", "Python"]


def test_java_is_not_javascript():
    assert extract_skills("Java developer") == ["Java"]
    assert extract_skills("JavaScript developer") == ["JavaScript"]


def test_git_is_not_github():
    assert extract_skills("GitHub Actions pipelines") == ["GitHub Actions"]
    assert extract_skills("Version control with Git") == ["Git"]


def test_go_needs_capital_and_whole_word():
    assert extract_skills("Worked at Google, did good work") == []
    assert extract_skills("We go to market") == []
    assert extract_skills("Strong Java or Go") == ["Java", "Go"]


def test_aliases_map_to_one_name():
    assert extract_skills("k8s clusters") == ["Kubernetes"]
    assert extract_skills("sklearn models") == ["scikit-learn"]
    assert extract_skills("Natural Language Processing") == ["NLP"]
    assert extract_skills("Retrieval-Augmented Generation") == ["RAG"]
    assert extract_skills("built a vector database") == ["Vector databases"]


def test_punctuation_and_symbols():
    assert extract_skills("Python, SQL.") == ["Python", "SQL"]
    assert extract_skills("C++/Python") == ["C++", "Python"]
    assert extract_skills("Node.js and React") == ["Node.js", "React"]


def test_ml_is_not_found_inside_mlflow():
    assert extract_skills("ML pipelines") == ["Machine learning"]
    assert extract_skills("tracked with MLflow") == ["MLflow"]


def test_has_skill():
    assert has_skill("Skills: Docker, Kubernetes", "Kubernetes")
    assert not has_skill("Skills: Docker", "Kubernetes")

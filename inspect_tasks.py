from inspect_ai import Task, task, eval
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact, scorer, accuracy, stderr, Score
from inspect_ai.solver import chain_of_thought, generate, system_message, use_tools
from inspect_ai.model import get_model
from inspect_ai.tool import tool

SYSTEM_MESSAGE = """
You are an expert movie recommendation system. Your job is to provide relevant
movie recommendations based on the user's preferences. Consider genre, director,
themes, and style when making recommendations. Be specific and suggest actual
movie titles that the user might enjoy based on their stated preferences.
"""

@scorer(metrics=[accuracy(), stderr()])
def recommendation_quality():
    """Score recommendations on relevance, diversity, and explanation."""
    SCORER_TEMPLATE = """
Evaluate the following movie recommendation response against the ideal target.
Score each criterion on a scale of 0-2:

- Relevance: Are the recommended movies similar to the user's preferences? (0-2)
- Diversity: Does the response provide a variety of options? (0-2)
- Explanation: Does the response explain why these movies were recommended? (0-2)

User request: {request}
Model response: {response}
Ideal response: {target}

For each criterion, provide a score and brief justification.
Then provide a final overall score as "GRADE: C" for correct (total >= 4) or "GRADE: I" for incorrect.
"""
    async def score(state, target):
        prompt = SCORER_TEMPLATE.format(
            request=state.input_text,
            response=state.output.completion,
            target=target.text
        )
        result = await get_model().generate(prompt)
        value = "C" if "GRADE: C" in result.completion else "I"
        return Score(
            value=value,
            answer=state.output.completion,
            explanation=result.completion
        )
    return score

@task
def movie_recommendation_eval(approach="basic"):
    """Evaluate movie recommendation quality using different approaches."""
    if approach == "basic":
        solver = [system_message(SYSTEM_MESSAGE), generate()]
        scorer = model_graded_fact()
    elif approach == "cot":
        solver = [
            system_message(SYSTEM_MESSAGE),
            chain_of_thought(),
            generate()
        ]
        scorer = model_graded_fact()
    elif approach == "custom_score":
        solver = [system_message(SYSTEM_MESSAGE), generate()]
        scorer = recommendation_quality()
    else:
        raise ValueError(f"Unknown approach: {approach}")
    return Task(
        dataset=json_dataset("movie_recommendations.json"),
        solver=solver,
        scorer=scorer
    )

@tool
def movie_database():
    async def execute(movie_name: str):
        """
        Query information about movies.

        Args:
            movie_name (str): The name of the movie to look up.

        Returns:
            dict: A dictionary containing movie information or an error message if not found.
        """
        movies = {
            "The Dark Knight": {
                "year": 2008,
                "director": "Christopher Nolan",
                "genre": ["Action", "Crime", "Drama"],
                "rating": 9.0
            },
            "Inception": {
                "year": 2010,
                "director": "Christopher Nolan",
                "genre": ["Action", "Adventure", "Sci-Fi"],
                "rating": 8.8
            },
            "Interstellar": {
                "year": 2014,
                "director": "Christopher Nolan",
                "genre": ["Adventure", "Drama", "Sci-Fi"],
                "rating": 8.6
            },
            "The Prestige": {
                "year": 2006,
                "director": "Christopher Nolan",
                "genre": ["Drama", "Mystery", "Sci-Fi"],
                "rating": 8.5
            },
            "Memento": {
                "year": 2000,
                "director": "Christopher Nolan",
                "genre": ["Mystery", "Thriller"],
                "rating": 8.4
            },
            "Dunkirk": {
                "year": 2017,
                "director": "Christopher Nolan",
                "genre": ["Action", "Drama", "History"],
                "rating": 7.9
            },
            "Tenet": {  
                "year": 2020,
                "director": "Christopher Nolan",
                "genre": ["Action", "Sci-Fi", "Thriller"],
                "rating": 7.5
            },
            "Oppenheimer": {
                "year": 2023,
                "director": "Christopher Nolan",
                "genre": ["Biography", "Drama", "History"],
                "rating": 8.5
            }
        }
        for name, info in movies.items():
            if movie_name.lower() in name.lower():
                return {"name": name, "info": info}
        return {"error": f"Movie '{movie_name}' not found in database"}
    return execute

@task
def tool_using_recommender():
    """Evaluate a recommendation system that uses tools."""
    return Task(
        dataset=json_dataset("movie_recommendations.json"),
        solver=[
            system_message(SYSTEM_MESSAGE),
            use_tools([movie_database()]),
            generate()
        ],
        scorer=recommendation_quality()
    )

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .recommender_model import recommend_movie  # Import recommendation function

@csrf_exempt
def get_recommendations(request):
    """
    API endpoint to get movie recommendations.
    Accepts a JSON request with a "movie" key.
    """
    if request.method == "POST":
        try:
            # Parse JSON request body
            data = json.loads(request.body)
            movie_name = data.get("movie")

            # Validate input
            if not movie_name:
                return JsonResponse({"error": "Please provide a movie name"}, status=400)

            # Get recommendat

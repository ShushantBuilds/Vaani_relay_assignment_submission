from django.urls import path
from relay_app.standin_views import standin_chat_completions, standin_configure
from relay_app.pipeline import ask_relay_view

urlpatterns = [
    # Stand-in Model Service endpoints (mimics Groq OpenAI-compatible API)
    path("api/standin/v1/chat/completions", standin_chat_completions),
    path("api/standin/configure", standin_configure),
    # The Relay Assistant endpoint
    path("api/relay/ask", ask_relay_view),
]
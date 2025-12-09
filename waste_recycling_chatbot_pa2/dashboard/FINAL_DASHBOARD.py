#!/usr/bin/env python3
"""
Swiss Recycling Assistant - Production Dashboard
================================================
Strict Swiss Recycle compliant chatbot with professional Dash UI.

ARCHITECTURE:
- All chatbot logic (RECYCLING_GUIDE, WasteClassifier, OllamaChat, SwissRecyclingBot)
  is imported from the CLI version (improved_swiss_waste_chatbot_opensource.py)
- Dashboard provides multi-session chat UI with image upload
- Both CLI and Dashboard use identical Swiss Recycle compliance rules

UPDATING SWISS REGULATIONS:
- Modify RECYCLING_GUIDE in the chatbot file to update disposal guidelines
- Changes automatically apply to both CLI and Dashboard
"""

import base64
import io
import logging
import uuid
import webbrowser
import threading
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional, List

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------------------
# IMPORT SHARED LOGIC FROM CHATBOT
# -----------------------------------------------------------------------------
import os
import sys

# Add project root (one level above /dashboard) to Python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from chatbot.improved_swiss_waste_chatbot_opensource import (
    Config,
    RECYCLING_GUIDE,
    ConfidenceLevel,
    WasteClassifier,
    OllamaChat,
    SwissRecyclingBot,
)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# BOT CACHE - One bot instance per language
# -----------------------------------------------------------------------------
BOT_CACHE: Dict[str, Optional[SwissRecyclingBot]] = {}


def get_bot(language: str) -> Optional[SwissRecyclingBot]:
    """Get or create a SwissRecyclingBot for the specified language"""
    language = language or "en"
    
    # Already cached?
    if language in BOT_CACHE:
        return BOT_CACHE[language]
    
    try:
        # model_path=None -> Bot searches for model like in CLI
        bot = SwissRecyclingBot(model_path=None, language=language)
        BOT_CACHE[language] = bot
        return bot
    except RuntimeError as e:
        # e.g. if Ollama is not running
        logger.error(f"Could not initialize SwissRecyclingBot: {e}")
        BOT_CACHE[language] = None
        return None


# -----------------------------------------------------------------------------
# UI TEXT (EN/DE)
# -----------------------------------------------------------------------------
def get_texts(language: str) -> Dict[str, str]:
    """Get UI texts in specified language"""
    if language == "de":
        return {
            "title": "Swiss Recycling Assistant",
            "subtitle": "KI-gestützte Recycling-Beratung für die Schweiz",
            "upload_title": "Bild hochladen",
            "chat_title": "Chat",
            "chat_input": "Stellen Sie eine Frage...",
            "new_chat": "Neuer Chat",
            "delete": "Löschen",
            "language_label": "Sprache",
            "processing": "Verarbeitung...",
            "confidence": "Konfidenz",
            "category": "Kategorie",
            "guidelines": "Richtlinien",
            "official_info": "Offizielle Informationen",
            "alternatives": "Alternativen",
            "welcome_title": "Hi, ich bin dein Swiss Recycling Assistant",
            "welcome_text": "Du bist unsicher, wie du etwas in der Schweiz recyceln sollst? Lade ein Foto deines Abfalls hoch oder stelle mir einfach direkt deine Frage.",
            "no_chats": "Keine Chatverläufe",
            "upload_button": "Bild",
            "image_analyzed": "Bild analysiert",
            "chat_history_label": "Verlauf",
            "detected": "Erkannt",
        }
    return {
        "title": "Swiss Recycling Assistant",
        "subtitle": "AI-powered recycling guidance for Switzerland",
        "upload_title": "Upload Image",
        "chat_title": "Chat",
        "chat_input": "Ask a question...",
        "new_chat": "New Chat",
        "delete": "Delete",
        "language_label": "Language",
        "processing": "Processing...",
        "confidence": "Confidence",
        "category": "Category",
        "guidelines": "Guidelines",
        "official_info": "Official information",
        "alternatives": "Alternatives",
        "welcome_title": "Hi, I'm your Swiss Recycling Assistant",
        "welcome_text": "Not sure how to recycle something in Switzerland? Upload a picture of your waste item or just ask me directly.",
        "no_chats": "No chat history",
        "upload_button": "Image",
        "image_analyzed": "Image analyzed",
        "chat_history_label": "History",
        "detected": "Detected",
    }


# -----------------------------------------------------------------------------
# DASH APP SETUP
# -----------------------------------------------------------------------------
# Initialize Dash app with Bootstrap theme and callback exception suppression
# suppress_callback_exceptions=True allows dynamic callback registration for pattern-matching callbacks
app: Dash = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Swiss Recycling Assistant",
    suppress_callback_exceptions=True,
)
server = app.server

# Define custom HTML template with embedded CSS for professional UI styling
# Includes responsive sidebar, chat bubbles, markdown content, and Bootstrap icons
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0; padding: 0; min-height: 100vh; display: flex; flex-direction: column;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                color: #2c3e50;
            }
            #react-entry-point { flex: 1; display: flex; flex-direction: column; }
            .sidebar-section-label {
                font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 8px; padding-left: 2px;
            }
            .language-dropdown > div > div {
                background-color: #ffffff !important; border: 1px solid #d1d5db !important;
                border-radius: 8px !important; color: #2c3e50 !important; font-size: 14px !important;
                transition: all 0.2s ease !important;
            }
            .language-dropdown > div > div:hover {
                border-color: #4a7ba7 !important; box-shadow: 0 0 0 3px rgba(74, 123, 167, 0.08) !important;
            }
            .language-dropdown [class*="-control--is-focused"] {
                border-color: #4a7ba7 !important; box-shadow: 0 0 0 3px rgba(74, 123, 167, 0.1) !important;
            }
            .language-dropdown [class*="-singleValue"] { color: #2c3e50 !important; }
            .language-dropdown [class*="-option"] {
                background-color: #ffffff !important; color: #2c3e50 !important;
                padding: 10px 12px !important; cursor: pointer !important; font-size: 14px !important;
            }
            .language-dropdown [class*="-option"]:hover,
            .language-dropdown [class*="-option--is-focused"] { background-color: #f3f4f6 !important; }
            .language-dropdown [class*="-option--is-selected"] {
                background-color: #4a7ba7 !important; color: #ffffff !important;
            }
            .language-dropdown [class*="-menu"] {
                background-color: #ffffff !important; border: 1px solid #d1d5db !important;
                border-radius: 8px !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
                margin-top: 4px !important;
            }
            .language-dropdown [class*="-indicatorSeparator"] { background-color: #d1d5db !important; }
            .language-dropdown svg { fill: #6b7280 !important; }
            .language-dropdown [class*="-Input"] input { color: #2c3e50 !important; }
            .language-dropdown [class*="-placeholder"] { color: #9ca3af !important; }
            .chat-history-item {
                display: flex; align-items: center; padding: 10px 12px; border-radius: 8px;
                margin-bottom: 4px; font-size: 14px; transition: all 0.15s ease;
                cursor: pointer; border: 1px solid transparent;
            }
            .chat-history-item:hover { background-color: #f3f4f6; border-color: #e5e7eb; }
            .chat-history-item.active { background-color: #eff6ff; border-color: #bfdbfe; }
            .chat-history-item .chat-title {
                flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                color: #374151; font-weight: 400;
            }
            .chat-history-item.active .chat-title { color: #1e40af; font-weight: 500; }
            .chat-history-item .delete-btn { opacity: 0; transition: opacity 0.15s ease; }
            .chat-history-item:hover .delete-btn { opacity: 1; }
            .chat-history-item .delete-btn:hover { color: #dc2626 !important; }
            .chat-input-group {
                display: flex; align-items: stretch; border-radius: 28px;
                background-color: #ffffff; border: 2px solid #e5e7eb;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06); overflow: hidden;
                transition: all 0.2s ease;
            }
            .chat-input-group:hover, .chat-input-group:focus-within {
                border-color: #4a7ba7; box-shadow: 0 4px 16px rgba(74, 123, 167, 0.2);
                background-color: #f8fbff;
            }
            .chat-input-group .btn-upload {
                display: inline-flex; align-items: center; justify-content: center;
                padding: 14px 20px; border: none; background-color: transparent;
                color: #4a7ba7; font-weight: 500; font-size: 14px; cursor: pointer;
                transition: all 0.2s ease; gap: 8px; white-space: nowrap; border-radius: 0;
            }
            .chat-input-group:hover .btn-upload,
            .chat-input-group:focus-within .btn-upload {
                background-color: transparent; color: #3d6687;
            }
            .btn-upload i { font-size: 16px; }
            .chat-input-group input {
                flex: 1; border: none !important; background-color: transparent !important;
                padding: 14px 24px; font-size: 15px; outline: none !important;
                box-shadow: none !important;
            }
            .chat-input-group .btn-send {
                border: none !important; background-color: #4a7ba7; color: white;
                padding: 14px 24px; min-width: 60px; transition: all 0.2s ease; border-radius: 0;
            }
            .chat-input-group:hover .btn-send,
            .chat-input-group:focus-within .btn-send { background-color: #3d6687; }
            .app-footer {
                padding: 24px 32px; text-align: center; background-color: #ffffff;
                border-top: 1px solid #e5e7eb; font-size: 13px; color: #6b7280; line-height: 1.6;
            }
            .app-footer a { transition: color 0.2s ease; }
            .app-footer a:hover { color: #4a7ba7; }
            .markdown-content { color: #2c3e50; line-height: 1.6; }
            .markdown-content p { margin-bottom: 1em; line-height: 1.6; }
            .markdown-content p:last-child { margin-bottom: 0; }
            .markdown-content ul, .markdown-content ol { margin-bottom: 1em; padding-left: 1.75em; }
            .markdown-content li { margin-bottom: 0.4em; line-height: 1.6; }
            .markdown-content h1, .markdown-content h2, .markdown-content h3 {
                margin-top: 1.2em; margin-bottom: 0.6em; line-height: 1.3;
                font-weight: 600; color: #1a202c;
            }
            .markdown-content h1:first-child,
            .markdown-content h2:first-child,
            .markdown-content h3:first-child { margin-top: 0; }
            .markdown-content h1 { font-size: 1.5em; }
            .markdown-content h2 { font-size: 1.3em; }
            .markdown-content h3 { font-size: 1.15em; }
            .markdown-content code {
                background-color: #f3f4f6; color: #1f2937; padding: 0.2em 0.4em;
                border-radius: 4px; font-size: 0.9em;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            }
            .markdown-content pre {
                background-color: #f3f4f6; padding: 1em; border-radius: 6px;
                overflow-x: auto; border: 1px solid #e5e7eb;
            }
            .markdown-content pre code { background-color: transparent; padding: 0; }
            .markdown-content a {
                color: #4a7ba7; text-decoration: none; transition: color 0.2s ease;
            }
            .markdown-content a:hover { color: #3d6687; text-decoration: underline; }
            .markdown-content strong { font-weight: 600; color: #1a202c; }
            button, a, input, textarea, select { transition: all 0.2s ease; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #f3f4f6; }
            ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
            .image-result-card {
                background-color: #f9fafb; border: 1px solid #e5e7eb;
                border-radius: 12px; padding: 16px; margin-bottom: 16px;
            }
            .image-result-card img {
                max-width: 100%; max-height: 200px; border-radius: 8px;
                margin-bottom: 12px; border: 1px solid #e5e7eb;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Predefined style dictionaries for consistent UI component styling
# Used across sidebar, chat bubbles, and main content area
STYLES = {
    "sidebar": {
        "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "280px", "padding": "0",
        "backgroundColor": "#f8f9fa", "borderRight": "1px solid #e5e7eb", "color": "#2c3e50",
        "overflowY": "hidden", "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)", "zIndex": 1000,
        "display": "flex", "flexDirection": "column",
    },
    "sidebar_collapsed": {
        "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "72px", "padding": "20px 12px",
        "backgroundColor": "#f8f9fa", "borderRight": "1px solid #e5e7eb", "color": "#2c3e50",
        "overflowY": "auto", "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)", "zIndex": 1000,
    },
    "main_expanded": {
        "marginLeft": "280px", "padding": "32px 48px 100px 48px", "minHeight": "calc(100vh - 100px)",
        "backgroundColor": "#ffffff", "transition": "margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    },
    "main_collapsed": {
        "marginLeft": "72px", "padding": "32px 48px 100px 48px", "minHeight": "calc(100vh - 100px)",
        "backgroundColor": "#ffffff", "transition": "margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    },
    "chat_bubble_user": {
        "backgroundColor": "#4a7ba7", "color": "#ffffff", "borderRadius": "16px 16px 4px 16px",
        "padding": "14px 18px", "maxWidth": "65%", "marginLeft": "auto", "fontSize": "15px",
        "lineHeight": "1.6", "boxShadow": "0 2px 8px rgba(74, 123, 167, 0.15)",
    },
    "chat_bubble_bot": {
        "backgroundColor": "#f3f4f6", "color": "#2c3e50", "borderRadius": "16px 16px 16px 4px",
        "padding": "14px 18px", "maxWidth": "65%", "fontSize": "15px", "lineHeight": "1.6",
        "border": "1px solid #e5e7eb",
    },
    "avatar": {
        "width": "40px", "height": "40px", "borderRadius": "50%", "objectFit": "cover",
        "flexShrink": 0, "border": "2px solid #ffffff", "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.08)",
    },
    "chat_row": {
        "display": "flex", "alignItems": "flex-end", "gap": "12px", "marginBottom": "20px",
    },
}


def create_sidebar(language: str, sessions: Dict, active_session: str, collapsed: bool):
    """Build collapsible sidebar"""
    texts = get_texts(language)
    session_items = []
    if sessions:
        sorted_sessions = list(reversed(list(sessions.items())))
        for sid, messages in sorted_sessions:
            title = texts["new_chat"]
            for msg in messages:
                if msg.get("role") == "user":
                    title = msg.get("content", "")[:40]
                    if len(msg.get("content", "")) > 40:
                        title += "..."
                    break
                elif msg.get("role") == "image_result":
                    title = texts["image_analyzed"]
                    break
            is_active = sid == active_session
            session_items.append(
                html.Div(
                    [
                        html.Div(title, className="chat-title", id={"type": "session-select", "index": sid}),
                        dbc.Button(
                            html.I(className="bi bi-trash", style={"fontSize": "12px"}),
                            id={"type": "session-delete", "index": sid},
                            color="link", size="sm", className="delete-btn",
                            style={"color": "#9ca3af", "padding": "4px 6px"},
                        ),
                    ],
                    className=f"chat-history-item {'active' if is_active else ''}",
                )
            )
    else:
        session_items.append(
            html.Div(texts["no_chats"], style={
                "color": "#9ca3af", "fontSize": "13px", "padding": "16px 12px",
                "textAlign": "center", "fontStyle": "italic"
            })
        )
    
    sidebar_style = STYLES["sidebar_collapsed" if collapsed else "sidebar"]
    if collapsed:
        return html.Div([
            dbc.Button(
                html.I(className="bi bi-list", style={"fontSize": "22px"}),
                id="sidebar-toggle", color="link",
                style={"color": "#4b5563", "padding": "12px", "width": "100%",
                       "borderRadius": "10px", "transition": "all 0.2s ease"},
                n_clicks=0,
            ),
        ], style=sidebar_style)
    
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="bi bi-recycle", style={"fontSize": "20px", "color": "#4a7ba7", "marginRight": "10px"}),
                    html.Div(texts["title"], style={"fontSize": "16px", "fontWeight": "600", "color": "#1a202c", "lineHeight": "1.2"}),
                ], style={"display": "flex", "alignItems": "center", "flex": 1}),
                dbc.Button(
                    html.I(className="bi bi-layout-sidebar", style={"fontSize": "16px"}),
                    id="sidebar-toggle", color="link",
                    style={"color": "#6b7280", "padding": "6px", "transition": "color 0.2s ease"},
                    n_clicks=0,
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),
            html.Div([
                html.Label(texts["language_label"], className="sidebar-section-label"),
                dcc.Dropdown(
                    id="language-dropdown",
                    options=[{"label": "English", "value": "en"}, {"label": "Deutsch", "value": "de"}],
                    value=language, clearable=False, className="language-dropdown",
                ),
            ], style={"marginBottom": "20px"}),
            dbc.Button(
                [html.I(className="bi bi-plus-lg", style={"marginRight": "8px", "fontSize": "14px"}), texts["new_chat"]],
                id="new-chat-btn",
                style={"width": "100%", "marginBottom": "20px", "borderRadius": "10px", "padding": "12px 16px",
                       "fontSize": "14px", "fontWeight": "500", "backgroundColor": "#4a7ba7", "border": "none",
                       "color": "#ffffff", "transition": "all 0.2s ease", "boxShadow": "0 2px 4px rgba(74, 123, 167, 0.2)"},
            ),
            html.Div(texts["chat_history_label"], className="sidebar-section-label", style={"marginBottom": "12px"}),
        ], style={"padding": "20px 16px 16px 16px", "borderBottom": "1px solid #e5e7eb"}),
        html.Div(session_items, style={"flex": 1, "overflowY": "auto", "padding": "12px 16px"}),
    ], style=sidebar_style)


def create_chat_message(msg: Dict, is_user: bool, language: str):
    """Create chat message bubble"""
    if is_user:
        return html.Div([
            html.Div(msg.get("content", ""), style=STYLES["chat_bubble_user"]),
            html.Div(
                html.I(className="bi bi-person-circle", style={"fontSize": "20px", "color": "#4a7ba7"}),
                style={**STYLES["avatar"], "backgroundColor": "#e8f0f7", "display": "flex", "alignItems": "center", "justifyContent": "center"},
            ),
        ], style={**STYLES["chat_row"], "justifyContent": "flex-end"})
    elif msg.get("role") == "image_result":
        result_data = msg.get("result_data", {})
        image_src = msg.get("image_src", "")
        return html.Div([
            html.Img(src="/assets/robo_head.png", style=STYLES["avatar"]),
            html.Div([
                html.Div([
                    html.Img(src=image_src, style={
                        "maxWidth": "100%", "maxHeight": "180px", "borderRadius": "8px",
                        "marginBottom": "12px", "border": "1px solid #e5e7eb"
                    }) if image_src else None,
                    html.Div(result_data.get("content", [])),
                ], className="image-result-card"),
            ], style={**STYLES["chat_bubble_bot"], "backgroundColor": "transparent", "border": "none", "padding": "0"}),
        ], style=STYLES["chat_row"])
    else:
        content = msg.get("content", "")
        return html.Div([
            html.Img(src="/assets/robo_head.png", style=STYLES["avatar"]),
            html.Div(dcc.Markdown(content, className="markdown-content", style={"margin": 0}), style=STYLES["chat_bubble_bot"]),
        ], style=STYLES["chat_row"])


def create_chat_content(language: str, messages: List[Dict]):
    """Build chat content area"""
    texts = get_texts(language)
    if not messages:
        return html.Div([
            html.Div([
                html.Div(html.Img(src="/assets/robo_head.png", style={"width": "80px", "height": "80px", "marginBottom": "28px", "opacity": "0.9"}),
                         style={"textAlign": "center"}),
                html.H2(texts["welcome_title"], style={"fontWeight": "600", "marginBottom": "16px", "textAlign": "center",
                                                       "fontSize": "28px", "color": "#1a202c", "lineHeight": "1.3"}),
                html.P(texts["welcome_text"], style={"color": "#6b7280", "fontSize": "16px", "lineHeight": "1.65",
                                                     "textAlign": "center", "maxWidth": "560px", "margin": "0 auto"}),
            ], style={"backgroundColor": "#f9fafb", "borderRadius": "20px", "border": "1px solid #e5e7eb",
                      "padding": "48px 32px", "maxWidth": "680px", "margin": "0 auto"}),
        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center",
                  "justifyContent": "center", "padding": "60px 24px", "minHeight": "65vh"})
    return html.Div(
        [create_chat_message(msg, msg.get("role") == "user", language) for msg in messages],
        style={"padding": "28px 0", "maxWidth": "920px", "margin": "0 auto"},
    )


# -----------------------------------------------------------------------------
# APP LAYOUT
# -----------------------------------------------------------------------------
# Build main application layout with dynamic components
# Stores for state management: language, sessions, active session, sidebar toggle state
app.layout = html.Div([
    dcc.Store(id="language-store", data="en"),
    dcc.Store(id="sessions-store", data={}),
    dcc.Store(id="active-session-store", data=""),
    dcc.Store(id="sidebar-collapsed-store", data=False),
    html.Div(id="sidebar-container"),
    html.Div([
        html.Div(id="chat-content"),
        html.Div(
            html.Div([
                dcc.Upload(
                    id="image-upload",
                    children=html.Div([html.I(className="bi bi-image"), html.Span(id="upload-button-text")], className="btn-upload"),
                    multiple=False, style={"border": "none", "display": "inline-block"},
                ),
                dbc.Input(id="chat-input", placeholder="Ask a question...", type="text", n_submit=0),
                dbc.Button(html.I(className="bi bi-send-fill", style={"fontSize": "16px"}), id="send-button", className="btn-send"),
            ], className="chat-input-group", style={"maxWidth": "880px", "margin": "0 auto", "width": "100%"}),
            id="chat-input-container", style={"marginTop": "40px", "display": "flex", "justifyContent": "center"},
        ),
    ], id="main-content"),
    html.Div([
        html.Div([
            html.A("BAFU", href="https://www.bafu.admin.ch/bafu/de/home.html", target="_blank",
                   style={"color": "#6b7280", "textDecoration": "none", "fontWeight": "500"}),
            html.Span(" · ", style={"color": "#d1d5db", "margin": "0 8px"}),
            html.A("PET Recycling", href="https://www.petrecycling.ch/de/sammelstellen", target="_blank",
                   style={"color": "#6b7280", "textDecoration": "none", "fontWeight": "500"}),
            html.Span(" · ", style={"color": "#d1d5db", "margin": "0 8px"}),
            html.A("VetroSwiss", href="https://www.vetroswiss.ch/de/sammelstellen", target="_blank",
                   style={"color": "#6b7280", "textDecoration": "none", "fontWeight": "500"}),
        ])
    ], className="app-footer"),
], style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"})


# -----------------------------------------------------------------------------
# CALLBACKS - Using imported SwissRecyclingBot
# -----------------------------------------------------------------------------

@app.callback(
    Output("sidebar-container", "children"),
    [Input("language-store", "data"), Input("sessions-store", "data"),
     Input("active-session-store", "data"), Input("sidebar-collapsed-store", "data")],
)
def update_sidebar(language, sessions, active_session, collapsed):
    return create_sidebar(language or "en", sessions or {}, active_session or "", collapsed or False)


@app.callback(
    [Output("sidebar-collapsed-store", "data"), Output("main-content", "style")],
    Input("sidebar-toggle", "n_clicks"), State("sidebar-collapsed-store", "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, collapsed):
    new_collapsed = not collapsed
    new_style = STYLES["main_collapsed"] if new_collapsed else STYLES["main_expanded"]
    return new_collapsed, new_style


@app.callback(Output("language-store", "data"), Input("language-dropdown", "value"))
def update_language(lang):
    return lang or "en"


@app.callback(
    [Output("chat-input", "placeholder"), Output("upload-button-text", "children")],
    Input("language-store", "data"),
)
def update_ui_texts(language):
    texts = get_texts(language)
    return texts["chat_input"], texts["upload_button"]


@app.callback(
    Output("chat-content", "children"),
    [Input("language-store", "data"), Input("sessions-store", "data"), Input("active-session-store", "data")],
)
def render_chat(language, sessions, active_session):
    messages = []
    if sessions and active_session and active_session in sessions:
        messages = sessions[active_session]
    return create_chat_content(language, messages)


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True)],
    Input("new-chat-btn", "n_clicks"), State("sessions-store", "data"),
    prevent_initial_call=True,
)
def new_chat(n_clicks, sessions):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    sessions = sessions or {}
    new_id = str(uuid.uuid4())[:8]
    sessions[new_id] = []
    return sessions, new_id


@app.callback(
    Output("active-session-store", "data", allow_duplicate=True),
    Input({"type": "session-select", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_session(n_clicks):
    if not ctx.triggered_id or not any(n_clicks):
        raise dash.exceptions.PreventUpdate
    return ctx.triggered_id["index"]


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True)],
    Input({"type": "session-delete", "index": ALL}, "n_clicks"),
    [State("sessions-store", "data"), State("active-session-store", "data")],
    prevent_initial_call=True,
)
def delete_session(n_clicks, sessions, active_session):
    if not ctx.triggered_id or not any(n for n in n_clicks if n):
        raise dash.exceptions.PreventUpdate
    sid = ctx.triggered_id["index"]
    sessions = sessions or {}
    if sid in sessions:
        del sessions[sid]
    new_active = active_session if active_session != sid else (list(sessions.keys())[0] if sessions else "")
    return sessions, new_active


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True)],
    [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")],
    [State("chat-input", "value"), State("sessions-store", "data"),
     State("active-session-store", "data"), State("language-store", "data")],
    prevent_initial_call=True,
)
def send_message(n_clicks, n_submit, user_text, sessions, active_session, language):
    """Handle text messages - uses imported SwissRecyclingBot"""
    if (not n_clicks and not n_submit) or not user_text or not user_text.strip():
        raise dash.exceptions.PreventUpdate
    
    sessions = sessions or {}
    # Create new session if none exists or active session is invalid
    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []
    
    # Add user message to session history
    user_text = user_text.strip()
    sessions[active_session].append({"role": "user", "content": user_text})
    
    # Retrieve cached bot instance for the selected language
    bot = get_bot(language)
    
    if bot is None:
        # Ollama not running
        if language == "de":
            response = "Das KI-Backend ist derzeit nicht verfügbar. Bitte prüfen Sie, ob Ollama läuft."
        else:
            response = "The AI backend is currently unavailable. Please check that Ollama is running."
    else:
        try:
            # Query bot using Swiss Recycling compliance logic
            response = bot.ask(user_text)
        except Exception as e:
            logger.error(f"Error in send_message: {e}")
            response = f"Error: {e}"
    
    # Append bot response to session history
    sessions[active_session].append({"role": "assistant", "content": response})
    return sessions, active_session


@app.callback(
    Output("chat-input", "value"),
    [Input("send-button", "n_clicks"), Input("chat-input", "n_submit")],
    prevent_initial_call=True,
)
def clear_input(n_clicks, n_submit):
    return ""


@app.callback(
    [Output("sessions-store", "data", allow_duplicate=True), Output("active-session-store", "data", allow_duplicate=True)],
    [Input("image-upload", "contents"), State("image-upload", "filename"),
     State("language-store", "data"), State("sessions-store", "data"), State("active-session-store", "data")],
    prevent_initial_call=True,
)
def handle_image(contents, filename, language, sessions, active_session):
    """Handle image uploads - uses imported SwissRecyclingBot.process_image()"""
    if not contents:
        raise dash.exceptions.PreventUpdate
    
    sessions = sessions or {}
    texts = get_texts(language)
    
    if not active_session or active_session not in sessions:
        active_session = str(uuid.uuid4())[:8]
        sessions[active_session] = []
    
    bot = get_bot(language)
    if bot is None:
        result_content = [
            html.Div(
                "The AI backend is currently unavailable. Please check that Ollama is running."
                if language == "en"
                else "Das KI-Backend ist derzeit nicht verfügbar. Bitte prüfen Sie, ob Ollama läuft.",
                style={"color": "#d97706", "fontWeight": "500", "fontSize": "14px"},
            )
        ]
        sessions[active_session].append({
            "role": "image_result",
            "image_src": contents,
            "result_data": {"content": result_content},
        })
        return sessions, active_session
    
    try:
        # Convert base64-encoded image data to bytes
        header, encoded = contents.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        
        # Save image to temporary file with unique identifier
        # Preserves original file extension for format detection
        suffix = Path(filename).suffix or ".png"
        tmp_path = os.path.join(
            tempfile.gettempdir(), f"dashboard_upload_{uuid.uuid4().hex}{suffix}"
        )
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)
        
        # Process image using SwissRecyclingBot classification and advice engine
        result = bot.process_image(tmp_path)
        classification = result["classification"]
        advice = result["advice"]
        
        # Format classification results for UI display
        category = classification["category"].replace("_", " ").title()
        confidence = classification["confidence"]
        
        result_content = [
            html.H5(
                f"{texts['detected']}: {category}",
                style={"fontWeight": "600", "marginBottom": "8px",
                       "fontSize": "17px", "color": "#1a202c"},
            ),
            html.P(
                f"{texts['confidence']}: {confidence:.0%}",
                style={"color": "#10b981", "fontSize": "14px",
                       "marginBottom": "14px", "fontWeight": "500"},
            ),
            dcc.Markdown(advice, className="markdown-content", style={"marginBottom": 0}),
        ]
        
        sessions[active_session].append({
            "role": "image_result",
            "image_src": contents,
            "result_data": {"content": result_content},
        })
        
        # Clean up temporary file to avoid disk space accumulation
        # Silently ignore errors if file was already deleted
        try:
            os.remove(tmp_path)
        except:
            pass
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        result_content = [
            html.Div(
                f"Error processing image: {e}" if language == "en"
                else f"Fehler bei der Bildverarbeitung: {e}",
                style={"color": "#dc2626", "fontWeight": "500", "fontSize": "14px"},
            )
        ]
        sessions[active_session].append({
            "role": "image_result",
            "image_src": contents,
            "result_data": {"content": result_content},
        })
    
    return sessions, active_session


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.5, open_browser).start()
    app.run(debug=True, host="0.0.0.0", port=8050)

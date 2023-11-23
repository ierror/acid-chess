import base64
import hashlib
import string
from random import choices

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from requests_oauthlib import OAuth2Session

CLIENT_ID = "acid-chess"
CLIENT_SECRET = "acid-chess"
SCOPES = "board:play"
AUTH_ENDPOINT = "https://lichess.org/oauth"
TOKEN_ENDPOINT = "https://lichess.org/api/token"
REDIRECT_URI = "https://acid-chess/"


def generate_auth_challenge():
    code_verifier = "".join(choices(string.ascii_letters + string.digits, k=128))
    code_sha_256 = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    b64 = base64.urlsafe_b64encode(code_sha_256)
    code_challenge = b64.decode("utf-8").replace("=", "")
    return code_verifier, code_challenge


class AuthBrowserWindow(QMainWindow):
    def __init__(self, on_success):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.webview = QWebEngineView(self.central_widget)
        self.layout.addWidget(self.webview)
        self.resize(800, 600)

        def page_load_finished():
            token_url = self.webview.page().url().url()
            if token_url.startswith(REDIRECT_URI):
                self.hide()

                token_url, state_token_url = oauth.authorization_url(token_url)
                token = oauth.fetch_token(
                    TOKEN_ENDPOINT,
                    authorization_response=token_url,
                    include_client_id=True,
                    code_verifier=code_verifier,
                )
                on_success(token)
                self.close()

        oauth = OAuth2Session(client_id=CLIENT_ID, scope=SCOPES, redirect_uri=REDIRECT_URI)

        code_verifier, code_challenge = generate_auth_challenge()
        authorization_url, state = oauth.authorization_url(
            AUTH_ENDPOINT, code_challenge_method="S256", code_challenge=code_challenge
        )
        self.webview.setUrl(authorization_url)
        self.webview.page().loadFinished.connect(page_load_finished)

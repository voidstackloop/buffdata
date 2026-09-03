"""Synthetic OIDC issuer + TLS ingress, ONLY mounted by smoke_test.py.

This deliberately auto-authorizes one synthetic subject. It is not an authentication
service and must never be used in a real deployment. No production image copies it.
"""
import base64
import hashlib
from http import client as http_client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import ssl
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt

config = json.loads(Path("/fixture/identity.json").read_text())
key = Path("/fixture/signing.pem").read_bytes()
codes = {}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Authorization codes and cookies must not appear in operational logs.

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/oidc/jwks":
            return self.send_json(200, config["jwks"])
        if parsed.path == "/oidc/authorize":
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if query.get("client_id") != config["client_id"] or query.get("redirect_uri") != config["redirect_uri"] or (
                query.get("code_challenge_method") != "S256" or query.get("response_type") != "code"):
                return self.send_json(400, {"error": "invalid_authorization_request"})
            code = secrets.token_urlsafe(32)
            with lock:
                codes[code] = {**query, "expires": time.time() + 60}
            self.send_response(302)
            self.send_header("Location", config["redirect_uri"] + "?" + urlencode({"code": code, "state": query["state"]}))
            self.end_headers()
            return
        self.proxy()

    def do_POST(self):
        if urlsplit(self.path).path != "/oidc/token":
            return self.proxy()
        body = {k: v[0] for k, v in parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()).items()}
        with lock:
            grant = codes.pop(body.get("code", ""), None)
        expected = "Basic " + base64.b64encode((config["client_id"] + ":" + config["client_secret"]).encode()).decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(body.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
        if not grant or grant["expires"] < time.time() or not secrets.compare_digest(self.headers.get("Authorization", ""), expected) or (
            body.get("grant_type") != "authorization_code" or body.get("redirect_uri") != config["redirect_uri"] or
            not secrets.compare_digest(challenge, grant["code_challenge"])):
            return self.send_json(400, {"error": "invalid_grant"})
        token = jwt.encode({"sub": "alpha", "iss": config["issuer"], "aud": config["client_id"],
            "nonce": grant["nonce"], "exp": int(time.time()) + 300}, key, algorithm="RS256", headers={"kid": "smoke"})
        self.send_json(200, {"id_token": token, "access_token": secrets.token_urlsafe(32), "token_type": "Bearer", "expires_in": 300})

    def do_PUT(self):
        self.proxy()

    def do_DELETE(self):
        self.proxy()

    def proxy(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "connection", "transfer-encoding"}}
        headers["Host"] = urlsplit(config["redirect_uri"]).netloc
        connection = http_client.HTTPConnection("api", 8000, timeout=60)
        try:
            connection.request(self.command, self.path, body, headers)
            response = connection.getresponse()
            data = response.read()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in {"connection", "transfer-encoding", "content-length"}:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            connection.close()


server = ThreadingHTTPServer(("0.0.0.0", 8443), Handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain("/fixture/server.pem", "/fixture/signing.pem")
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()

"""HTTP transport, for running Cognita as a hosted server.

Wraps the MCP app with two things a public deployment needs and stdio does not:

  * **A bearer token.** Without one the library would be readable, and writable,
    by anyone who found the URL. The server refuses to start over HTTP unless
    COGNITA_AUTH_TOKEN is set.
  * **A health endpoint.** Platforms restart what they cannot check.

Streamable HTTP is the current MCP transport; SSE is mounted alongside it for
clients that have not moved over yet.
"""

import hmac

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from cognita.core.config import settings
from cognita.core.logging import get_logger

logger = get_logger(__name__)

_PUBLIC_PATHS = ("/health",)


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Require ``Authorization: Bearer <token>`` on everything but /health."""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")

        # Constant-time compare so the token cannot be recovered by timing.
        if scheme.lower() != "bearer" or not hmac.compare_digest(presented, self._token):
            logger.warning("Rejected unauthenticated request to %s", request.url.path)
            return JSONResponse(
                {"error": "Unauthorized. Send 'Authorization: Bearer <token>'."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def _configure_allowed_hosts(mcp: FastMCP) -> None:
    """Tell MCP's DNS-rebinding guard which hostnames are legitimate.

    The guard rejects any Host header it does not recognise, which is right for
    a local server and fatal for a deployed one: without the public hostname on
    this list every request is answered 421.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = list(settings.ALLOWED_HOSTS)
    if not hosts:
        hosts = [
            f"{host}:{settings.MCP_PORT}" for host in ("localhost", "127.0.0.1")
        ] + ["localhost", "127.0.0.1"]
        logger.warning(
            "COGNITA_ALLOWED_HOSTS is not set — only local requests will be accepted. "
            "Set it to your public hostname before deploying."
        )

    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=hosts,
        allowed_origins=[f"https://{h}" for h in hosts] + [f"http://{h}" for h in hosts],
    )
    logger.info("Accepting requests for hosts: %s", ", ".join(hosts))


async def health(_: Request) -> Response:
    """Liveness probe. Deliberately does not touch the database.

    A slow query should not read as a dead process and trigger a restart loop
    in the middle of ingesting a book.
    """
    return JSONResponse({"status": "ok", "service": "cognita"})


def build_app(mcp: FastMCP) -> Starlette:
    """Assemble the HTTP app: a health route in front of the MCP endpoint."""
    token = settings.AUTH_TOKEN
    if not token:
        raise SystemExit(
            "COGNITA_AUTH_TOKEN must be set to serve over HTTP.\n"
            'Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    if len(token) < 16:
        raise SystemExit("COGNITA_AUTH_TOKEN is too short — use at least 16 characters.")
    if not hasattr(mcp, "streamable_http_app"):
        raise SystemExit(
            "This version of the mcp package has no Streamable HTTP transport. "
            "Upgrade with: pip install -U 'mcp[cli]'"
        )

    _configure_allowed_hosts(mcp)

    # The MCP app already serves /mcp, so it mounts at the root rather than
    # under a prefix — mounting at /mcp would put the endpoint at /mcp/mcp.
    inner = mcp.streamable_http_app()

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Mount("/", app=inner),
        ],
        # The session manager lives in the inner app's lifespan. Mounting does
        # not carry it across, so it is passed on explicitly; without it every
        # request fails on an uninitialised task group.
        lifespan=lambda _: inner.router.lifespan_context(inner),
    )
    app.add_middleware(BearerTokenMiddleware, token=token)
    return app


async def serve_http(mcp: FastMCP, host: str, port: int) -> None:
    app = build_app(mcp)
    logger.info("Cognita listening on http://%s:%d/mcp (bearer token required)", host, port)
    config = uvicorn.Config(app, host=host, port=port, log_level=settings.LOG_LEVEL.lower())
    await uvicorn.Server(config).serve()

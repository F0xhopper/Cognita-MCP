import jwt

from cognita.auth.domain import AuthenticatedUser
from cognita.core.config import settings


class InvalidTokenError(Exception):
    pass


class TokenExpiredError(InvalidTokenError):
    pass


class JWTService:
    """Stateless — one instance is fine for the app lifetime."""

    def verify_token(self, token: str) -> AuthenticatedUser:
        try:
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise InvalidTokenError("Token is invalid") from exc

        user_id: str | None = payload.get("sub")
        if not user_id:
            raise InvalidTokenError("Token missing sub claim")

        return AuthenticatedUser(
            user_id=user_id,
            email=payload.get("email", ""),
            role=payload.get("role", "authenticated"),
        )


_jwt_service = JWTService()


def get_jwt_service() -> JWTService:
    return _jwt_service

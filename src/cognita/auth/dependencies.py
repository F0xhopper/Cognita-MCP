from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cognita.auth.domain import AuthenticatedUser
from cognita.auth.service import InvalidTokenError, JWTService, TokenExpiredError, get_jwt_service

_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    svc: JWTService = Depends(get_jwt_service),
) -> AuthenticatedUser:
    try:
        return svc.verify_token(credentials.credentials)
    except TokenExpiredError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user_id(user: AuthenticatedUser = Depends(get_current_user)) -> str:
    return user.user_id

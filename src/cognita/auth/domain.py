from dataclasses import dataclass


@dataclass(frozen=True)
class AuthenticatedUser:
    """Verified identity extracted from a Supabase JWT."""

    user_id: str   # UUID string — Supabase auth.users.id (sub claim)
    email: str
    role: str      # 'authenticated' | 'anon'

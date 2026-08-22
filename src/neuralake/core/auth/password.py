import re

from passlib.context import CryptContext

_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def hash_password(plain: str) -> str:
    return _ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    if _SHA256_RE.match(hashed):
        import hashlib

        return hashlib.sha256(plain.encode()).hexdigest() == hashed
    return _ctx.verify(plain, hashed)


def needs_rehash(hashed: str) -> bool:
    if _SHA256_RE.match(hashed):
        return True
    return _ctx.needs_update(hashed)

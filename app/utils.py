from flask import request


def get_client_ip():
    """Extracts the real IP address of the client, even if behind a proxy."""
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0]  # Take the first IP in the list
    return request.remote_addr  # Fallback to remote_addr
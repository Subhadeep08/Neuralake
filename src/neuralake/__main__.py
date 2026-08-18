import sys

import uvicorn

from neuralake.config.settings import get_settings


def main():
    settings = get_settings()

    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        import asyncio
        from neuralake.mcp.server import run_stdio
        asyncio.run(run_stdio())
        return

    uvicorn.run(
        "neuralake.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()

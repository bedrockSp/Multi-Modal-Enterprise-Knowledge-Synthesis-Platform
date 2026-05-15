from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import socketio

from app.middlewares.auth import AuthMiddleware
from app.middlewares.auth_paths import auth_paths
from app.routes import (
    document_creator,
    documents,
    excel_skill,
    export,
    extra,
    health,
    insights,
    query,
    sensing,
    settings,
    strategic_analysis,
    strategic_roadmap,
    technical_analysis,
    technical_roadmap,
    thread,
    upload,
    user,
)
from app.socket_handler import cancel_all_heartbeats, sio

fastapi_app = FastAPI()


@fastapi_app.on_event("startup")
async def startup_event():
    from core.sensing.scheduler import start_scheduler
    await start_scheduler()


@fastapi_app.on_event("shutdown")
async def shutdown_event():
    await cancel_all_heartbeats()

excluded_routes = [("POST", "/user"), ("POST", "/user/login")]
fastapi_app.add_middleware(
    AuthMiddleware, included_paths=auth_paths, excluded_routes=excluded_routes
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fastapi_app.mount("/static", StaticFiles(directory="app/public"), name="static")


fastapi_app.include_router(query.router)
fastapi_app.include_router(user.router)
fastapi_app.include_router(upload.router)
fastapi_app.include_router(health.router)
fastapi_app.include_router(thread.router)
fastapi_app.include_router(extra.router)
fastapi_app.include_router(strategic_roadmap.router)
fastapi_app.include_router(insights.router)
fastapi_app.include_router(technical_roadmap.router)
fastapi_app.include_router(strategic_analysis.router)
fastapi_app.include_router(technical_analysis.router)
fastapi_app.include_router(documents.router)
fastapi_app.include_router(export.router)
fastapi_app.include_router(document_creator.router)
fastapi_app.include_router(excel_skill.router)
fastapi_app.include_router(settings.router)
fastapi_app.include_router(sensing.router)

app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

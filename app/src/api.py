# app/src/api.py

from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging

from Simulation.sim import Simulator  # Ensure correct import path
from Simulation.connections import active_connections, lock  # Import from connections.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
router = APIRouter()

# CORS Configuration
origins = [
    "http://localhost:3000",  # React's default port
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Shared variables to store simulator instance and task
simulator_instance = None
simulation_task = None

class SimulationParameters(BaseModel):
    poi: dict
    graph: dict
    numDoctor: int
    numNurse: int
    numPatient: int
    numSteps: int

def reset_simulation():
    global simulator_instance, simulation_task
    simulator_instance = None
    simulation_task = None
    logger.info("Simulation state has been reset.")

@router.get("/", response_class=HTMLResponse, tags=["ROOT"])
async def root():
    return """
    <html>
        <head>
            <title>Research Skill Seminar - Dummy App</title>
        </head>
        <body>
            <h1>Research Skill Seminar - Dummy App</h1>
            <p>This is a service to test the deployment of a IVIA app.</p>
        </body>
    </html>
    """

@router.get(
    "/health",
    responses={
        200: {"description": "API is up and running"},
        503: {"description": "Service is unavailable"},
    },
    tags=["default"],
    summary="Checks the health of the API",
)
async def health_get() -> None:
    logger.info("Health check: API is up and running.")
    pass

@router.get("/message", tags=["test"])
async def get_message():
    logger.info("Received GET /message request.")
    return {"message": "Hello, World!"}

@router.post("/simulation/start", tags=["simulation"])
async def start_simulation(params: SimulationParameters):
    global simulator_instance, simulation_task

    if simulator_instance and not simulator_instance.stop_simulation:
        logger.warning("Attempted to start a simulation while one is already running.")
        raise HTTPException(status_code=400, detail="Simulation already running.")

    logger.info("Starting a new simulation.")

    # Initialize the simulator instance with the given parameters
    simulator_instance = Simulator(
        poi=params.poi,
        graph=params.graph,
        nurses=params.numNurse,
        doctors=params.numDoctor,
        patients=params.numPatient,
        steps=params.numSteps
    )

    # Start the simulation in a background task
    simulation_task = asyncio.create_task(simulator_instance.run_simulation())

    # Attach a callback to reset simulation state when the task is done
    simulation_task.add_done_callback(lambda t: reset_simulation())

    logger.info("Simulation started successfully.")

    return {"message": "Simulation started"}

@router.post("/simulation/stop", tags=["simulation"])
async def stop_simulation():
    global simulator_instance

    if simulator_instance and not simulator_instance.stop_simulation:
        simulator_instance.pause_event.set()  # Resume the simulation before stopping
        logger.info("Stopping the simulation.")
        simulator_instance.stop_simulation = True
        # Wait for the simulation to acknowledge stopping
        await simulator_instance.stop_event.wait()
        logger.info("Simulation has been stopped.")
        return {"message": "Simulation stop signal sent."}
    else:
        logger.warning("Attempted to stop a simulation that is not running.")
        raise HTTPException(status_code=400, detail="No active simulation to stop or simulation already stopped.")

@router.post("/simulation/pause", tags=["simulation"])
async def pause_simulation():
    global simulator_instance

    if simulator_instance and not simulator_instance.stop_simulation:
        logger.info("Pausing the simulation.")
        simulator_instance.pause_event.clear()  # Pause the simulation
        return {"message": "Simulation paused."}
    else:
        logger.warning("Attempted to pause a simulation that is not running.")
        raise HTTPException(status_code=400, detail="No active simulation to pause.")

@router.post("/simulation/resume", tags=["simulation"])
async def resume_simulation():
    global simulator_instance

    if simulator_instance and not simulator_instance.stop_simulation:
        logger.info("Resuming the simulation.")
        simulator_instance.pause_event.set()  # Resume the simulation
        return {"message": "Simulation resumed."}
    else:
        logger.warning("Attempted to resume a simulation that is not running.")
        raise HTTPException(status_code=400, detail="No active simulation to resume.")

@router.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket client connected.")

    try:
        while True:
            # Keep the connection open
            data = await websocket.receive_text()
            # Optionally handle incoming messages from the client
            # For example, handle control messages via WebSocket
            message = json.loads(data)
            action = message.get("action")
            if action == "pause":
                await pause_simulation()
            elif action == "resume":
                await resume_simulation()
            elif action == "stop":
                await stop_simulation()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        active_connections.remove(websocket)
        logger.error(f"WebSocket connection error: {e}")

app.include_router(router)

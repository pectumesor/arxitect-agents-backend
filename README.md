### Arxitect — Final Project for Fundamentals of Web Engineering at ETH Zürich
As our final project for the course Fundamentals of Web Engineering at ETH Zürich, we chose to contribute to an ongoing project from the IVIA Lab called Arxitect.

Arxitect is a web application designed for architects to upload their design drawings and test how well their hospital floor plans function in simulated scenarios. The application supports simulations like:

Shortest path calculations between key areas (e.g., ER to surgery).

Detection of congestion in narrow hallways due to potential worker/patient collisions.

### Project Concept
The core idea was to run a simulation using agents—ideally trained via Unity or Reinforcement Learning—that would navigate the hospital layout. These simulations would highlight flaws in the design, such as overly narrow hallways or inefficient routes.

### My Contribution: Python Backend
I was responsible for developing the entire Python backend, using:

- FastAPI – for building the web API.

- OpenAI Gym – to model the simulation environment.

### Agents & Environment
Multiple agent types were created, each with distinct **goals** and **routes**.

Each agent's behavior was managed by a **Finite State Machine** (FSM).

A custom Gym environment was created to:

- Draw the hospital walls using the uploaded design data.

- Simulate realistic movement patterns.

Agents were required to pass through Points of Interest (POIs), defined via the Arxitect web interface.

### Agent Behavior
While the original plan was to train the agents using reinforcement learning for more realistic, human-like movement, time constraints prevented us from implementing this. (Note: RL was not required for grading.)

Instead, I hardcoded the agents' behaviors via their step() functions:

- At each step, an agent sampled an angle between 0° and 180°.

- The agent then updated its facing direction and moved a fixed distance.

- Collisions with walls were detected and avoided accordingly.

### Integration with Frontend
The frontend passed hospital layout data and POI information to the backend.

A WebSocket connection enabled:

- Starting and pausing the simulation in real-time.

- Streaming agent data back to the frontend.

### Output & Visualization
The backend returned:

- New agent positions at each step.

- Collision data (e.g., agent-to-agent collisions).

This allowed the frontend to:

- Draw travel paths for each agent.

- Generate heatmaps to visualize congestion areas.


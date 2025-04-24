# app/src/connections.py
import threading
from typing import List
from fastapi import WebSocket

active_connections: List[WebSocket] = []
lock = threading.Lock()

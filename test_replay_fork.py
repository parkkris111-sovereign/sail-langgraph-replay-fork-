import json
import sqlite3
from typing import TypedDict, Optional, Dict, Any
from pathlib import Path
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --------------------------------------------------#
# 1. Host-Side SQLite Authorization Boundary
# --------------------------------------------------#
class HostAuthorizationBoundary:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS authorization_ledger (
                    authorization_id TEXT PRIMARY KEY,
                    principal TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    consumed_at_ledger_index INTEGER
                )
            """)
            # Seed a valid test authorization nonce
            conn.execute("""
                INSERT OR IGNORE INTO authorization_ledger 
                (authorization_id, principal, tool, consumed)
                VALUES (?, ?, ?, ?)
            """, ("AUTH-NONCE-001", "agent-prime", "restricted-tool", 0))
            conn.commit()

    def consume_once(self, authorization_id: str, principal: str, tool: str, arguments: dict) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT consumed FROM authorization_ledger WHERE authorization_id = ?
            """, (authorization_id,))
            row = cursor.fetchone()

            if not row:
                return {
                    "accepted": False,
                    "reason": "AUTHORIZATION_NOT_FOUND",
                    "authorization_id": authorization_id
                }

            is_consumed = row[0]
            if is_consumed == 1:
                return {
                    "accepted": False,
                    "reason": "AUTHORIZATION_ALREADY_CONSUMED",
                    "authorization_id": authorization_id
                }

            # Atomically consume the authorization
            cursor.execute("""
                UPDATE authorization_ledger 
                SET consumed = 1, consumed_at_ledger_index = (SELECT COUNT(*) FROM authorization_ledger)
                WHERE authorization_id = ?
            """, (authorization_id,))
            conn.commit()

            return {
                "accepted": True,
                "reason": "OK",
                "authorization_id": authorization_id
            }

# --------------------------------------------------#
# 2. Agent State Definition
# --------------------------------------------------#
class AgentState(TypedDict, total=False):
    principal: str
    authorization_id: str
    tool: str
    arguments: dict
    fork_label: str
    execution_status: str
    boundary_response: dict

# --------------------------------------------------#
# 3. Graph Construction
# --------------------------------------------------#
def build_graph(boundary: HostAuthorizationBoundary):
    def prepare(state: AgentState):
        return {
            "execution_status": "READY_FOR_AUTHORIZATION"
        }

    def protected_action(state: AgentState):
        response = boundary.consume_once(
            authorization_id=state["authorization_id"],
            principal=state["principal"],
            tool=state["tool"],
            arguments=state["arguments"],
        )
        if not response["accepted"]:
            return {
                "execution_status": "REPLAY_REJECTED",
                "boundary_response": response,
            }
        return {
            "execution_status": "EXECUTED",
            "boundary_response": response,
        }

    builder = StateGraph(AgentState)
    builder.add_node("prepare", prepare)
    builder.add_node("protected_action", protected_action)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "protected_action")
    builder.add_edge("protected_action", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

# --------------------------------------------------#
# 4. Adversarial Temporal Fork Test
# --------------------------------------------------#
def test_pre_consumption_checkpoint_fork(tmp_path):
    database = tmp_path / "sail_replay_ledger.db"
    boundary = HostAuthorizationBoundary(database)
    graph = build_graph(boundary)

    thread_config = {"configurable": {"thread_id": "adversarial-fork-thread-01"}}

    initial_input: AgentState = {
        "principal": "agent-prime",
        "authorization_id": "AUTH-NONCE-001",
        "tool": "restricted-tool",
        "arguments": {"payload": "test-execution"}
    }

    # Step A: Run graph up to pre-consumption checkpoint (after 'prepare', before 'protected_action')
    # LangGraph checkpointer captures state at node boundaries.
    # We invoke the graph in steps or inspect checkpoints.
    
    # Run initial execution through to completion (consumes authorization)
    print("\n[TEST] Executing initial workflow stream...")
    events = list(graph.stream(initial_input, thread_config, stream_mode="values"))
    
    # Retrieve checkpoint history to find state *before* protected_action
    checkpoint_history = list(graph.get_state_history(thread_config))
    
    # Identify the pre-consumption checkpoint (state where prepare finished, before protected_action)
    pre_consumption_config = None
    for cp in checkpoint_history:
        # Looking for the checkpoint config corresponding to the state right after 'prepare'
        if cp.values.get("execution_status") == "READY_FOR_AUTHORIZATION":
            pre_consumption_config = cp.config
            break

    assert pre_consumption_config is not None, "Pre-consumption checkpoint not found in history."
    print("[TEST] Pre-consumption checkpoint successfully isolated.")

    # Verify initial execution succeeded
    final_state = graph.get_state(thread_config).values
    assert final_state.get("execution_status") == "EXECUTED"
    assert final_state.get("boundary_response")["accepted"] is True
    print("[TEST] Initial authorization and execution verified: ACCEPTED / EXECUTED.")

    # Step B: Adversarial Fork - Re-invoke graph using the pre-consumption checkpoint
    # This simulates workflow persistence restoring a state from before consumption.
    print("[TEST] Executing temporal fork using pre-consumption checkpoint...")
    
    fork_result = graph.invoke(None, pre_consumption_config)
    
    # Step C: Verify rejection of replayed authorization
    boundary_response = fork_result.get("boundary_response", {})
    print(f"[TEST] Boundary response on replayed fork: {boundary_response}")

    assert fork_result.get("execution_status") == "REPLAY_REJECTED"
    assert boundary_response.get("accepted") is False
    assert boundary_response.get("reason") == "AUTHORIZATION_ALREADY_CONSUMED"
    assert boundary_response.get("authorization_id") == "AUTH-NONCE-001"

    print("[VERIFICATION SUCCESS] SAIL-LANGGRAPH-REPLAY-FORK-01: REPLAY REJECTED AS EXPECTED.")

    # Step D: Generate Evidence Artifact
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    evidence_path = artifacts_dir / "sail_replay_fork_evidence.json"

    evidence_artifact = {
        "designation": "SAIL-LANGGRAPH-REPLAY-FORK-01",
        "status": "PASS",
        "verification_steps": {
            "initial_authorization": "ACCEPTED",
            "initial_action": "EXECUTED",
            "pre_consumption_checkpoint": "RESTORED",
            "temporal_fork": "CREATED",
            "replayed_authorization": "REJECTED",
            "duplicate_execution": "NONE",
            "ledger_records": 1
        },
        "boundary_rejection_payload": boundary_response
    }

    with open(evidence_path, "w") as f:
        json.dump(evidence_artifact, f, indent=2)
    print(f"[TEST] Evidence artifact written to {evidence_path}")

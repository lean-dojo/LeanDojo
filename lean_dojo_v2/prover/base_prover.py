"""Base agent class for Lean theorem proving agents."""

import time
from abc import ABC, abstractmethod
from typing import Optional

import networkx as nx
from loguru import logger
from pantograph import Server
from pantograph.expr import GoalState, Site, Tactic
from pantograph.search import Agent, SearchResult, SearchState
from pantograph.server import ServerError, TacticFailure

from lean_dojo_v2.database.models.theorems import Theorem


class BaseProver(Agent, ABC):
    """Base class for Lean theorem proving agents.

    This class provides common functionality for search-based theorem proving agents,
    including the search algorithm, guidance strategy, and path finding utilities.
    """

    def __init__(self):
        super().__init__()
        self.theorem: Optional[Theorem] = None

    @abstractmethod
    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        """Generate the next tactic for the given goal state.

        Args:
            state: The current goal state
            goal_id: The ID of the goal to solve

        Returns:
            The tactic to apply, or None if no tactic can be generated
        """
        pass

    def guidance(self, state: GoalState) -> list[float]:
        """Simple guidance: prioritize goals with fewer variables."""
        priorities = []
        for goal in state.goals:
            priority = 1.0 / (len(goal.variables) + 1)
            priorities.append(priority)
        return priorities

    def search(
        self,
        server: Server,
        goal: Optional[str] = None,
        theorem: Optional[Theorem] = None,
        verbose: bool = False,
    ):
        """Execute search-based theorem proving.

        Args:
            server: The Lean server to interact with
            theorem: The theorem to prove
            verbose: Whether to print verbose output

        Returns:
            Tuple of (SearchResult, list of used tactics)
        """
        self.reset()
        self.theorem = theorem
        if isinstance(theorem, Theorem) and theorem is not None:
            goal = server.env_inspect(theorem.full_name)["type"]["pp"]
        elif goal is None:
            raise ValueError("Either theorem or goal must be provided")

        goal_state = server.goal_start(goal)

        # Initialize the search graph
        search_graph = nx.DiGraph()

        # Direct implementation of Agent.search
        assert server.is_automatic(), "Search must be run in automatic mode"

        n_goals_root = len(goal_state.goals)
        time_start = time.time()

        initial_state = SearchState(
            goal_state,
            parent=None,
            parent_goal_id=None,
            priorities=[0.0 for _ in goal_state.goals],
        )
        search_stack = [initial_state]

        # Add initial state to graph
        search_graph.add_node(0, state=initial_state, step=0)

        max_steps = 100
        max_trials_per_goal = 5
        solved_node_id = None

        for i_step in range(max_steps):
            assert search_stack, "No states in search stack"

            if verbose:
                print(f"I={i_step}: len(S) = {len(search_stack)}")
            search_state = search_stack[-1]
            current_node_id = len(search_stack) - 1

            assert isinstance(search_state, SearchState)

            if search_state.is_solved:
                # Mark the final node as solved
                search_graph.nodes[current_node_id]["solved"] = True
                solved_node_id = current_node_id
                result = SearchResult(
                    n_goals_root=n_goals_root,
                    duration=time.time() - time_start,
                    success=True,
                    steps=i_step,
                )

                # Find shortest path to solved state
                shortest_path = self._find_shortest_path_to_solved(
                    search_graph, solved_node_id
                )
                used_tactics = [edge[1] for edge in shortest_path]
                return result, used_tactics

            # Find the unsolved goal with the highest priority
            goal_id = search_state.next_goal_id

            if search_state.trials[goal_id] > max_trials_per_goal:
                # force halt the search
                tactic = None
            else:
                # Generate tactic for this goal
                tactic = self.next_tactic(search_state.goal_state, goal_id)

            if verbose:
                print(f"Next tactic: {tactic}")
            if not tactic:
                # resets the feedback
                search_state.tactic_feedback = None
                # pop the current state and continue to the next
                search_stack.pop(-1)
                if not search_stack:
                    if verbose:
                        print("Search stack has been exhausted")
                    self.reset()
                    result = SearchResult(
                        n_goals_root=n_goals_root,
                        duration=time.time() - time_start,
                        success=False,
                        steps=i_step,
                    )
                    return result, None
                continue

            try:
                search_state.trials[goal_id] += 1
                goal_state = search_state.goal_state
                if verbose:
                    print(
                        f"{goal_state.state_id}.{goal_id}: {tactic} on {goal_state.goals[goal_id]}"
                    )
                next_goal_state = server.goal_tactic(
                    goal_state, tactic, site=Site(goal_id, auto_resume=False)
                )
                # Generate priorities for the next goal state
                priorities = (
                    [0.0 for _ in next_goal_state.goals]
                    if len(next_goal_state.goals) <= 1
                    else self.guidance(next_goal_state)
                )
                next_state = SearchState(
                    goal_state=next_goal_state,
                    parent=search_state,
                    parent_goal_id=goal_id,
                    priorities=priorities,
                )
                search_stack.append(next_state)

                # Add new state to graph
                next_node_id = len(search_stack) - 1
                search_graph.add_node(next_node_id, state=next_state, step=i_step + 1)

                # Add edge from current state to next state with tactic as edge label
                search_graph.add_edge(
                    current_node_id,
                    next_node_id,
                    tactic=tactic,
                    goal_id=goal_id,
                    step=i_step + 1,
                )

            except TacticFailure as t:
                if verbose:
                    print(f"Tactic failed: {t}")
                search_state.tactic_feedback = str(t)
                # Add failed tactic as edge to a failure node
                failure_node_id = f"failure_{current_node_id}_{goal_id}_{search_state.trials[goal_id]}"
                search_graph.add_node(
                    failure_node_id, state=None, step=i_step + 1, failed=True
                )
                search_graph.add_edge(
                    current_node_id,
                    failure_node_id,
                    tactic=tactic,
                    goal_id=goal_id,
                    step=i_step + 1,
                    failed=True,
                )
                # try the next tactic. this one failed
            except ServerError as e:
                raise RuntimeError(f"While executing tactic: {tactic}") from e

        if verbose:
            print("Search iteration limit exhausted")

        self.reset()
        result = SearchResult(
            n_goals_root=n_goals_root,
            duration=time.time() - time_start,
            success=False,
            steps=max_steps,
        )
        return result, None

    @abstractmethod
    def generate_whole_proof(
        self,
        theorem: Theorem,
    ) -> str:
        """Generate a complete proof for the given theorem.

        Args:
            theorem: The theorem to prove

        Returns:
            A complete proof string
        """
        pass

    def _find_shortest_path_to_solved(
        self, search_graph: nx.DiGraph, solved_node_id: int
    ):
        """
        Find the shortest path from node 0 to the solved state.

        Returns:
            List of tuples: [(node_id, tactic, goal_id), ...]
        """
        try:
            # Find shortest path from node 0 to solved node
            shortest_path_nodes = nx.shortest_path(search_graph, 0, solved_node_id)

            # Extract tactics and goal_ids from the path
            path_info = []
            for i in range(len(shortest_path_nodes) - 1):
                current_node = shortest_path_nodes[i]
                next_node = shortest_path_nodes[i + 1]

                # Get edge data
                edge_data = search_graph.get_edge_data(current_node, next_node)
                if edge_data and not edge_data.get("failed", False):
                    tactic = edge_data.get("tactic", "unknown")
                    goal_id = edge_data.get("goal_id", -1)
                    path_info.append((current_node, tactic, goal_id))

            return path_info

        except nx.NetworkXNoPath:
            logger.warning("No path found to solved state")
            return None

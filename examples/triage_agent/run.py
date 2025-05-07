
# import sys
# sys.path.append(r'E:\swarm\swarm\swarm\repl')
# from myrepl import run_demo_loop
# from e.swarm.swarm.swarm.repl.myrepl import run_demo_loop

from agents import triage_agent
from swarm.repl import run_demo_loop

if __name__ == "__main__":
    run_demo_loop(triage_agent)

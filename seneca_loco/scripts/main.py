import sys
from pathlib import Path
from simulation.movement import walking
 
def main():
    print("Starting...")
    walking.walk()
 
if __name__ == "__main__":
    main()


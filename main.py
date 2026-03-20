import sys
import os

def print_help():
    print("\n--- Command Center: Gesture Recognition Project ---")
    print("Usage: python main.py [command]")
    print("\nAvailable commands:")
    print("  train     - Runs the training script for the gesture model.")
    print("  classify  - Runs the real-time gesture classification using webcam.")
    print("  test      - Runs environment tests for MediaPipe.")
    print("  help      - Shows this message.")
    print("\nNote: Notebooks for data collection are in the 'notebooks/' folder.")

def main():
    if len(sys.argv) < 2:
        print_help()
        return

    cmd = sys.argv[1].lower()

    # Base paths
    src_dir = os.path.join(os.getcwd(), 'src')
    tests_dir = os.path.join(os.getcwd(), 'tests')

    if cmd == "train":
        print("Starting Training...")
        os.system(f"python {os.path.join(src_dir, 'treinar_modelo_gestos.py')}")
    elif cmd == "classify":
        print("Starting Real-time Classification...")
        os.system(f"python {os.path.join(src_dir, 'classificar_gestos_webcam.py')}")
    elif cmd == "test":
        print("Running tests...")
        os.system(f"python {os.path.join(tests_dir, 'test_mediapipe.py')}")
    elif cmd == "help":
        print_help()
    else:
        print(f"Unknown command: {cmd}")
        print_help()

if __name__ == "__main__":
    main()

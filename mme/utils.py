"""utils.py
Some utils functionalities
"""
import os, select, sys
import pickle

"""
In Windows NT import msvcrt
"""
if os.name == "nt":
    import msvcrt


def heardEnter():
    """
    Function which listens for the user pressing ENTER key

    Returns:
        bool: True if the user presses something on the key, False otherwise
    """
    if os.name == "nt":
        if msvcrt.kbhit():
            if msvcrt.getch() == b"q":
                print("Quit key pressed.")
                return True
        else:
            return False
    else:
        i, o, e = select.select([sys.stdin], [], [], 0.0001)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                return True
        return False


def save(path, o):
    """
    Function which saves the object o in a Pickle file in path

    Args:
        path: path where to save the file
        o: object to save
    """
    with open(path, "wb") as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore(path):
    """
    Function which loads the object in the Pickle file pointed by path

    Args:
        path: path where the file to restore is file

    Returns:
        o: object read
    """
    with open(path, "rb") as handle:
        o = pickle.load(handle)
    return o

import os
import random
import string

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_FOLDER = r"C:\Users\vinothg\Downloads\New folder\Knowledge_Base"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".jfif"
}


# ============================================================
# WINDOWS LONG PATH SUPPORT
# ============================================================

def long_path(path):
    """
    Convert a Windows path to extended-length format.
    This helps with very long filenames/folder paths.
    """
    path = os.path.abspath(path)

    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path

    return path


# ============================================================
# GENERATE RANDOM 10 LETTER NAME
# ============================================================

def generate_random_name():
    letters = string.ascii_letters
    return "".join(random.choices(letters, k=10))


# ============================================================
# RENAME IMAGES
# ============================================================

def rename_images(root_folder):

    root_folder = long_path(root_folder)

    renamed_count = 0
    skipped_count = 0
    failed_count = 0

    print("=" * 70)
    print("STARTING IMAGE RENAME")
    print("=" * 70)
    print(f"Root folder: {root_folder}")
    print()

    # Walk through all folders and subfolders
    for current_folder, subfolders, files in os.walk(root_folder):

        for filename in files:

            # Get extension
            extension = os.path.splitext(filename)[1].lower()

            # Skip non-image files
            if extension not in IMAGE_EXTENSIONS:
                continue

            old_path = os.path.join(current_folder, filename)

            # Check whether file still exists
            if not os.path.isfile(old_path):
                print(f"SKIPPED - File not found:")
                print(filename)
                print()
                skipped_count += 1
                continue

            # Generate unique new filename
            while True:

                random_name = generate_random_name()

                new_filename = random_name + extension

                new_path = os.path.join(
                    current_folder,
                    new_filename
                )

                if not os.path.exists(new_path):
                    break

            # Try to rename
            try:

                os.rename(old_path, new_path)

                print(filename)
                print(f"    -> {new_filename}")
                print()

                renamed_count += 1

            except FileNotFoundError:

                print("SKIPPED - File disappeared before rename:")
                print(old_path)
                print()

                skipped_count += 1

            except PermissionError:

                print("SKIPPED - Permission denied:")
                print(old_path)
                print()

                skipped_count += 1

            except OSError as e:

                print("FAILED:")
                print(old_path)
                print(f"Error: {e}")
                print()

                failed_count += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print("RENAME COMPLETED")
    print("=" * 70)

    print(f"Successfully renamed : {renamed_count}")
    print(f"Skipped              : {skipped_count}")
    print(f"Failed               : {failed_count}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not os.path.isdir(ROOT_FOLDER):

        print("ERROR: Root folder does not exist:")
        print(ROOT_FOLDER)

    else:

        rename_images(ROOT_FOLDER)
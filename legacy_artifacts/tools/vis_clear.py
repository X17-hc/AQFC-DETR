import os
import shutil
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / "logs"

    # Support running on the server path as well
    server_logs = Path("/workspace/DQDetr/logs")
    if server_logs.exists():
        logs_dir = server_logs

    vis_clear_dir = logs_dir / "vis_clear"

    source_mapping = {
        "vis_advanced": "_advanced",
        "vis_origin": "_origin",
        "vis_tta": "_tta",
    }

    for src_dir_name, suffix in source_mapping.items():
        src_dir = logs_dir / src_dir_name
        if not src_dir.exists():
            print(f"[SKIP] Source directory not found: {src_dir}")
            continue

        copied_count = 0
        for src_file in src_dir.iterdir():
            if not src_file.is_file():
                continue

            stem = src_file.stem
            ext = src_file.suffix

            if not stem.endswith("_Result"):
                continue

            image_id = stem[:-len("_Result")]

            target_dir = vis_clear_dir / image_id
            target_dir.mkdir(parents=True, exist_ok=True)

            target_file = target_dir / f"{image_id}{suffix}{ext}"
            shutil.copy2(src_file, target_file)
            copied_count += 1
            print(f"[COPY] {src_file.name} -> {target_file}")

        print(f"[DONE] {src_dir_name}: {copied_count} files copied.\n")


if __name__ == "__main__":
    main()

"""Print concise metadata summary for an HDF5 file.

Usage:
  python -m scripts.utils.inspect_hdf5_metadata --path /path/to/file.h5
"""

import argparse
from typing import Iterable

import h5py


def _iter_items(group: h5py.Group, prefix: str = "") -> Iterable[tuple[str, object]]:
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        yield path, value
        if isinstance(value, h5py.Group):
            yield from _iter_items(value, path)


def _print_attrs(attrs: h5py.AttributeManager, indent: str = "  ") -> None:
    if not attrs:
        print(f"{indent}(none)")
        return
    for key in sorted(attrs.keys()):
        print(f"{indent}{key}: {attrs[key]}")


def inspect(path: str, recursive: bool = True) -> None:
    with h5py.File(path, "r") as h5:
        print(f"File: {path}")
        print("\n[File attrs]")
        _print_attrs(h5.attrs)

        print("\n[Datasets/Groups]")
        if recursive:
            items = list(_iter_items(h5))
        else:
            items = [(f"/{k}", v) for k, v in h5.items()]

        if not items:
            print("  (none)")
            return

        for obj_path, obj in items:
            if isinstance(obj, h5py.Dataset):
                print(f"\n- Dataset {obj_path}")
                print(f"  shape: {obj.shape}")
                print(f"  dtype: {obj.dtype}")
                print("  attrs:")
                _print_attrs(obj.attrs, indent="    ")
            else:
                print(f"\n- Group {obj_path}")
                print("  attrs:")
                _print_attrs(obj.attrs, indent="    ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HDF5 metadata and attrs.")
    parser.add_argument("--path", required=True, help="Path to input .h5 file")
    parser.add_argument(
        "--non_recursive",
        action="store_true",
        help="Only inspect root-level datasets/groups",
    )
    args = parser.parse_args()

    inspect(args.path, recursive=not args.non_recursive)


if __name__ == "__main__":
    main()

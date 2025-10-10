### Shared data workflow with STOCKYARD and symlinks

- Why use STOCKYARD
  - STOCKYARD is the global shared filesystem across TACC systems; a single path works on multiple clusters.
  - Keeping shared data there avoids per-system duplication and lets all project members access the same canonical data.

- Shared directory layout (canonical source of truth)
  - Root: `$STOCKYARD/gw_shared_data/`
  - Subfolders:
    - `template_banks/` → RP bank HDF5s (e.g., `rp_bank_*.h5`)
    - `mismatch_cubes/` → per-mcz mismatch cubes (e.g., `mismatch_cubes_*.h5`)
    - `best_match/` → aggregated best-match HDF5s (e.g., `best_match_*.h5`)

- Group ownership and permissions (one-time)
  - Ensure the directory belongs to your project group (e.g., `G-827397`)
    - `chgrp -R G-827397 "$STOCKYARD/gw_shared_data"`
  - Enable setgid so new files inherit group:
    - `chmod g+s "$STOCKYARD/gw_shared_data"`
  - Make the tree group-readable (and optionally writable):
    - `chmod -R g+rX "$STOCKYARD/gw_shared_data"`
    - add `g+rwX` if collaborators must write
  - Set your shell umask (e.g., `umask 027`) so new files are group-readable and not world-readable.

- Copy/sync new outputs into STOCKYARD
  - Template banks:
    - `rsync -a data/template_banks/ "$STOCKYARD/gw_shared_data/template_banks/"`
  - Mismatch cubes:
    - `rsync -a data/contours/mismatch_cubes/ "$STOCKYARD/gw_shared_data/mismatch_cubes/"`
  - Best match:
    - `rsync -a data/contours/best_match/ "$STOCKYARD/gw_shared_data/best_match/"`
  - After syncing, ensure group is correct:
    - `chgrp -R G-827397 "$STOCKYARD/gw_shared_data"`

- Convenient access from $WORK via symlinks
  - Create symlinks in `$WORK` pointing to the canonical locations:
    - `ln -sfn "$STOCKYARD/gw_shared_data/template_banks" "$WORK/gw_shared_data_template_banks"`
    - `ln -sfn "$STOCKYARD/gw_shared_data/mismatch_cubes" "$WORK/gw_shared_data_mismatch_cubes"`
    - `ln -sfn "$STOCKYARD/gw_shared_data/best_match" "$WORK/gw_shared_data_best_match"`
  - Your code or command line can then use `$WORK/gw_shared_data_*` paths, while collaborators can use the canonical `$STOCKYARD/gw_shared_data/...` paths.

- Option: link project-local paths to shared
  - If code expects `data/template_banks`, replace directories with symlinks:
    - `ln -sfn "$STOCKYARD/gw_shared_data/template_banks" data/template_banks`
    - `ln -sfn "$STOCKYARD/gw_shared_data/mismatch_cubes" data/contours/mismatch_cubes`
    - `ln -sfn "$STOCKYARD/gw_shared_data/best_match" data/contours/best_match`

- Verification checklist
  - Group/permissions:
    - `ls -ld "$STOCKYARD/gw_shared_data" "$STOCKYARD/gw_shared_data"/*`
  - Symlinks:
    - `ls -l "$WORK"/gw_shared_data_*`
  - Group membership: `getent group G-827397`

- Notes
  - STOCKYARD is the recommended single shared location; $WORK is system-specific but symlinks make access convenient.
  - Always ensure new files are written with group-readable perms (umask 027, setgid on dirs).



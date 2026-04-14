# TACC STOCKYARD Shared Data Workflow

STOCKYARD is the global shared filesystem across TACC systems; a single path works on multiple clusters. Keeping canonical data there avoids per-system duplication and lets all project members access the same files.

For sharing project files across TACC users see: https://docs.tacc.utexas.edu/tutorials/sharingprojectfiles/

For ACL-based permission management see: https://docs.tacc.utexas.edu/tutorials/acls/

## Shared Directory Layout

Root: `$STOCKYARD/gw_shared_data/`

| Subfolder | Contents |
|---|---|
| `template_banks/` | RP bank HDF5s (`rp_bank_*.h5`) |
| `mismatch_cubes/` | Per-mcz mismatch cubes (`mismatch_cubes_*.h5`) |
| `best_match/` | Aggregated best-match HDF5s (`best_match_*.h5`) |

## Group Ownership and Permissions (one-time)

```bash
chgrp -R G-827397 "$STOCKYARD/gw_shared_data"
chmod g+s "$STOCKYARD/gw_shared_data"         # setgid: new files inherit group
chmod -R g+rX "$STOCKYARD/gw_shared_data"     # group-readable; use g+rwX if collaborators write
umask 027                                      # new files: group-readable, not world-readable
```

## Sync New Outputs to STOCKYARD

```bash
rsync -a data/template_banks/          "$STOCKYARD/gw_shared_data/template_banks/"
rsync -a data/mismatch/mismatch_cubes/ "$STOCKYARD/gw_shared_data/mismatch_cubes/"
rsync -a data/mismatch/best_match/     "$STOCKYARD/gw_shared_data/best_match/"
# Fix group ownership after sync
chgrp -R G-827397 "$STOCKYARD/gw_shared_data"
```

## Symlinks for Convenient Access

Create symlinks in `$WORK` pointing to canonical STOCKYARD locations:

```bash
ln -sfn "$STOCKYARD/gw_shared_data/template_banks" "$WORK/gw_shared_data_template_banks"
ln -sfn "$STOCKYARD/gw_shared_data/mismatch_cubes" "$WORK/gw_shared_data_mismatch_cubes"
ln -sfn "$STOCKYARD/gw_shared_data/best_match"     "$WORK/gw_shared_data_best_match"
```

Or point project-local paths directly to shared:

```bash
ln -sfn "$STOCKYARD/gw_shared_data/template_banks"  data/template_banks
ln -sfn "$STOCKYARD/gw_shared_data/mismatch_cubes"  data/mismatch/mismatch_cubes
ln -sfn "$STOCKYARD/gw_shared_data/best_match"       data/mismatch/best_match
```

## Verification Checklist

```bash
ls -ld "$STOCKYARD/gw_shared_data" "$STOCKYARD/gw_shared_data"/*  # group/permissions
ls -l "$WORK"/gw_shared_data_*                                     # symlinks
getent group G-827397                                               # group membership
```

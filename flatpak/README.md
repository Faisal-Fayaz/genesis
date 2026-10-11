# Genesis Flatpak packaging

The manifest (`io.github.Faisal_Fayaz.genesis.yml`) builds Genesis as a Flatpak;
the rest of this directory is the road to Flathub. Status: **draft, never
built** — the first local build is the next task, not a later one.

## Build locally

Prerequisites: `flatpak`, `flatpak-builder`, and the GNOME 48 SDK:

```sh
flatpak install -y flathub org.gnome.Platform//48 org.gnome.Sdk//48
```

From the repository root:

```sh
flatpak-builder --user --install --force-clean build-dir flatpak/io.github.Faisal_Fayaz.genesis.yml
flatpak run io.github.Faisal_Fayaz.genesis --help
```

(flatpak-builder itself: `flatpak --user install flathub
org.flatpak.Builder`, then `flatpak run org.flatpak.Builder ...`.)
Build trees (`build-dir/`, `.flatpak-builder/`) are git-ignored. Re-run
with `--force-clean` after manifest edits.

Verified 2026-10-11: clean build with no warnings, `genesis --version`
and `genesis doctor` green inside the sandbox (exact numpy/pillow pins,
GTK and X11 present), and a headless `gen-random` render producing a
valid PNG.

## Known unknowns (verify during the first build)

1. **dconf access for wallpaper-apply.** PARTLY VERIFIED (2026-10-11):
   `gsettings get` on the desktop background key works from inside the
   sandbox, so the `--filesystem=xdg-config/dconf` grant reaches the host
   settings. Writes go through the same path but were deliberately not
   test-built (a write test would change a real desktop's wallpaper).
   `genesis install` inside the sandbox correctly refuses (no systemctl)
   and points at `--print`.
2. **The daily timer under Flatpak.** DECIDED: the Flatpak is the app +
   renderer; automation runs host-side through the sandboxed binary:
   `flatpak run --command=genesis io.github.Faisal_Fayaz.genesis daily` as the
   `ExecStart` of a host user unit (verified `--dry-run` paths; a full
   `daily` would apply to a real desktop and stays a host-side action).
   `genesis install` remains host-CLI territory and is documented as such.
3. **Live mode in the sandbox.** Needs X11 socket (granted) and performs
   full-frame SHM blits; verify fps and SHM availability inside the
   sandbox, plus graceful behavior on Wayland hosts. Untested: needs a
   real X11 session, and the sandbox was only exercised headlessly.
4. **Runtime version.** 48 was chosen for availability; bump to current
   stable before submitting and re-run the suite against it.
5. ~~Python wheels need network at build time.~~ RESOLVED: wheels arrive as
   sha256-verified file sources; the build is fully offline. Keep the two
   runtime pins in the manifest in step with requirements.txt.

## Flathub submission checklist

Flathub accepts new apps via PR to
[github.com/flathub/flathub](https://github.com/flathub/flathub) adding
`io.github.Faisal_Fayaz.genesis.yml`. Before opening it:

- [x] Manifest builds clean with `--force-clean` and the app launches.
- [x] `genesis doctor` inside the sandbox reports sensibly (dconf reads
      verified; writes deliberately untested).
- [x] Metainfo validates clean (`appstreamcli validate`; one warning left
      before the repo existed — homepage URL — now filled).
- [x] Screenshots committed (app UI, silk, nebula) and wired as remote URLs.
- [x] Genesis module uses `type: git` pinned to the release tag, so local
      builds validate the exact submission artifact. Bump the tag per release.
- [x] Icon renders legibly (eyeballed at 256/48; dark tile with wave lines).
- [x] Desktop entry `Exec`/`Icon` rewritten at build time (verified in the
      installed export).
- [x] OARS content rating accurate (unrated/neutral: no content).
- [x] Timer story decided and documented (item 2 above).

## Filing (manual by the author — read this)

Flathub's AI policy forbids AI agents from opening or automating
submission PRs, so the author files; the body below it is prepared
(`gh pr create --body-file`). Two earlier malformed attempts (#10597 wrong
base branch, #10598 missing checklist) were bot-closed in seconds with no
reviewer involvement and are disclosed in the body.

```sh
gh pr create --repo flathub/flathub --base new-pr \
  --head Faisal-Fayaz:add-genesis-newpr \
  --title "Add io.github.Faisal_Fayaz.genesis" \
  --body-file /tmp/flathub-pr-body.md   # prepared draft, review before sending
```

Video for the template: `assets/screenshots/live-demo.mp4` (16 s live
session under Xvfb, committed). Fork branch `add-genesis-newpr` tracks
flathub `new-pr` plus the manifest; force-push it after manifest changes.
- [ ] Confirm license scanning passes: GPL-3.0-or-later, no bundled
      incompatible code (numpy/Pillow are fine).

## After acceptance

- Point README install instructions at Flathub first, venv second.
- Keep `<releases>` and CHANGELOG.md in lockstep per release.
- The Flathub repo needs its own updates for runtime bumps; subscribe to
  GNOME runtime EOL notices.

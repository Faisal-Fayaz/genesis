# Genesis Flatpak packaging

The manifest (`org.genesis.wallpapers.yml`) builds Genesis as a Flatpak;
the rest of this directory is the road to Flathub. Status: **draft, never
built** — the first local build is the next task, not a later one.

## Build locally

Prerequisites: `flatpak`, `flatpak-builder`, and the GNOME 48 SDK:

```sh
flatpak install -y flathub org.gnome.Platform//48 org.gnome.Sdk//48
```

From the repository root:

```sh
flatpak-builder --user --install --force-clean build-dir flatpak/org.genesis.wallpapers.yml
flatpak run org.genesis.wallpapers --help
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
2. **The daily timer under Flatpak.** systemd user units cannot be installed
   from inside the sandbox. Decide before submission: either the Flatpak is
   the app + renderer (timer stays a host-side `genesis install` step), or
   it ships a `flatpak run --command=...` wrapper the host timer calls.
   Do not submit promising a working timer you have not tested.
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
`org.genesis.wallpapers.yml`. Before opening it:

- [x] Manifest builds clean with `--force-clean` and the app launches.
- [x] `genesis doctor` inside the sandbox reports sensibly (dconf reads
      verified; writes deliberately untested).
- [x] Metainfo validates clean (`appstreamcli validate`; one warning left
      before the repo existed — homepage URL — now filled).
- [x] Screenshots committed (app UI, silk, nebula) and wired as remote URLs.
- [ ] Switch the genesis module source from `type: dir` (local builds) to
      `type: git` with the repository URL — Flathub does not accept local
      directory sources.
- [ ] Icon: `assets/genesis.svg` installed as
      `org.genesis.wallpapers.svg` renders legibly at 16–512 px (eyeball it).
- [ ] Desktop entry `Exec` matches the Flatpak command layout (rewritten at
      build time — verify in the installed export).
- [ ] OARS content rating still accurate (currently unrated/neutral).
- [ ] Decide and document the timer story (item 2 above) in the app
      description or README — submitters get asked.
- [ ] Confirm license scanning passes: GPL-3.0-or-later, no bundled
      incompatible code (numpy/Pillow are fine).

## After acceptance

- Point README install instructions at Flathub first, venv second.
- Keep `<releases>` and CHANGELOG.md in lockstep per release.
- The Flathub repo needs its own updates for runtime bumps; subscribe to
  GNOME runtime EOL notices.

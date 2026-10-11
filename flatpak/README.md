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

Build trees (`build-dir/`, `.flatpak-builder/`) are git-ignored. Re-run
with `--force-clean` after manifest edits.

## Known unknowns (verify during the first build)

1. **dconf access for wallpaper-apply.** The manifest grants
   `--filesystem=xdg-config/dconf`, which is the starting guess for letting
   sandboxed `genesis apply` / `genesis daily` write GNOME wallpaper keys.
   If `gsettings set` inside the sandbox cannot see the session bus, the
   honest fallback is documenting that wallpaper-apply runs outside the
   sandbox (installed timer + host CLI) while the Flatpak ships the
   renderer, library, gallery, and app UI.
2. **The daily timer under Flatpak.** systemd user units cannot be installed
   from inside the sandbox. Decide before submission: either the Flatpak is
   the app + renderer (timer stays a host-side `genesis install` step), or
   it ships a `flatpak run --command=...` wrapper the host timer calls.
   Do not submit promising a working timer you have not tested.
3. **Live mode in the sandbox.** Needs X11 socket (granted) and performs
   full-frame SHM blits; verify fps and SHM availability inside the
   sandbox, plus graceful behavior on Wayland hosts.
4. **Runtime version.** 48 was chosen for availability; bump to current
   stable before submitting and re-run the suite against it.
5. **Python wheels.** Built from PyPI at build time against
   `requirements.txt` pins. If Flathub's build network restricts this,
   vendor the wheels or switch to `pip --no-index` with a local source.

## Flathub submission checklist

Flathub accepts new apps via PR to
[github.com/flathub/flathub](https://github.com/flathub/flathub) adding
`org.genesis.wallpapers.yml`. Before opening it:

- [ ] Manifest builds clean with `--force-clean` and the app launches.
- [ ] `genesis doctor` inside the sandbox reports sensibly (dconf verdict
      recorded above — update this file with the outcome).
- [ ] Metainfo validates: `flatpak run --command=flatpak-builder-lint ...`
      or `appstreamcli validate assets/org.genesis.wallpapers.metainfo.xml`.
- [ ] Metainfo TODOs resolved: homepage URL, 2–4 screenshots, `<releases>`
      matching the tag.
- [ ] Icon: `assets/genesis.svg` installed as
      `org.genesis.wallpapers.svg` renders legibly at 16–512 px.
- [ ] Desktop entry `Exec` matches the Flatpak command layout.
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

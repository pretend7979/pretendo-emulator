<!--
# SPDX-FileCopyrightText: Copyright 2026 PreTenDo Emulator contributors
# SPDX-License-Identifier: GPL-3.0-or-later
-->

<p align="center">
  <img src="dist/pretendo-logo.png" alt="PreTenDo Emulator" width="420">
</p>

<h4 align="center">A free, open-source Switch 2 emulator for Windows</h4>

<p align="center">
  <a href="https://github.com/pretend7979/pretendo-emulator/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/pretend7979/pretendo-emulator?style=flat-square&color=e8141c&label=release"></a>
  <a href="https://github.com/pretend7979/pretendo-emulator/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/pretend7979/pretendo-emulator/total?style=flat-square&color=22efd8"></a>
  <a href="LICENSE.txt"><img alt="License: GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-6b6f85?style=flat-square"></a>
  <img alt="Platform: Windows x64" src="https://img.shields.io/badge/platform-Windows%20x64-6b6f85?style=flat-square">
</p>

<p align="center">
  <a href="#download">Download</a> &nbsp;&bull;&nbsp;
  <a href="#features">Features</a> &nbsp;&bull;&nbsp;
  <a href="#playing">Playing</a> &nbsp;&bull;&nbsp;
  <a href="#building">Building</a> &nbsp;&bull;&nbsp;
  <a href="#license">License</a>
</p>

## Download

Get the latest build from [Releases](https://github.com/pretend7979/pretendo-emulator/releases):
unzip `PreTenDo-Emulator-<version>-windows-x64.zip` anywhere and run `pretendo.exe`. It checks
these releases for updates itself.

Want all your consoles in one place? [Pretendulator](https://github.com/pretend7979/pretendulator)
is a game library that downloads PreTenDo Emulator for you and starts your Switch 2 games in it.

## Features

- **Switch 2 games** from `.nup` dumps of your own cartridges and downloads.
- **Per-game settings**: graphics, CPU and controls can differ for each game.
- **Updates itself** from this repository's releases.
- **Works with [Pretendulator](https://github.com/pretend7979/pretendulator)**, which can download,
  update and launch it, and open a game's settings directly.

## Playing

PreTenDo Emulator plays games from your own console. It ships no games, keys or firmware: dump
them from hardware you own.

Switch 2 games come as `.nup` files. Per-game settings are under a game's Properties; a frontend
can open them directly with `pretendo.exe --game-settings "<game file>"`.

## Building

See the [build guide](docs/Build.md). Release builds are made with
`dist/packaging/release_pretendo.py`, which builds, packages and publishes a version.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [AI policy](docs/policies/AI.md).

## License

PreTenDo Emulator is licensed under the GPLv3 (or any later version); see
[LICENSE.txt](LICENSE.txt). It is based on the Eden emulator, which grew out of yuzu; their
copyright notices are kept in the source files.

"Nintendo Switch" is a trademark of Nintendo. PreTenDo Emulator is not affiliated with Nintendo.

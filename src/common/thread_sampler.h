// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common::ThreadSampler {

/// EDEN_NX2_SAMPLE=<file> (debug, Windows): samples the call stacks of the registered threads
/// about every millisecond and writes per-address self and inclusive counts to <file> every 10 s.
/// EDEN_NX2_SAMPLE_START=<seconds> delays the first sample. Does nothing when unset.
void RegisterCurrentThread(const char* name);

} // namespace Common::ThreadSampler

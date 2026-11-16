# Copyright (c) Meta Platforms, Inc. and affiliates.

# This file handles configuration of what features we have based on dynamic build
# time checks.

set(ZL_HAVE_X86_64_ASM FALSE)
if (${IS_X86_64_ARCH})
    if (NOT MSVC AND NOT EMSCRIPTEN)
        message(
            STATUS
            "Enabling x86-64 assembly"
        )
        set(ZL_HAVE_X86_64_ASM TRUE)
    endif()
endif()

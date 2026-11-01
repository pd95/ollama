# Attribution for speech ports and embedded pronunciation data is not a Go
# module dependency, so go-licenses alone cannot collect it.
cmake_minimum_required(VERSION 3.24)

foreach(_variable IN ITEMS SOURCE_DIR OUTPUT_DIR)
    if(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_variable} is required")
    endif()
endforeach()

set(_model_dir "${SOURCE_DIR}/mlxrunner/model")
set(_license_files
    qwen3asr/QWEN_ASR_LICENSE
    qwen3tts/MLX_AUDIO_LICENSE
    kokoro/MLX_AUDIO_LICENSE
    voxtraltts/LICENSE.MLX-Audio
    chatterbox/MLX_AUDIO_LICENSE
    chatterbox/HIFT_LICENSE
    MLX_LM_LICENSE
    kokoro/english/data/LICENSE)

# Validate every source before writing a payload; missing attribution must fail.
foreach(_file IN ITEMS SPEECH_NOTICE ${_license_files})
    if(NOT EXISTS "${_model_dir}/${_file}")
        message(FATAL_ERROR "Missing speech attribution: ${_file}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
configure_file("${_model_dir}/SPEECH_NOTICE" "${OUTPUT_DIR}/SPEECH_NOTICE" COPYONLY)
set(_output "${OUTPUT_DIR}/SPEECH_LICENSE")
file(WRITE "${_output}" "License texts for native MLX speech ports and pronunciation dictionaries.\nSee SPEECH_NOTICE for source attribution and modification descriptions.\n")
set(_seen)
foreach(_file IN LISTS _license_files)
    file(SHA256 "${_model_dir}/${_file}" _hash)
    if(_hash IN_LIST _seen)
        continue()
    endif()
    list(APPEND _seen "${_hash}")
    file(READ "${_model_dir}/${_file}" _license)
    file(APPEND "${_output}"
        "\n================================================================================\n"
        "${_file}\n"
        "================================================================================\n"
        "${_license}\n")
endforeach()

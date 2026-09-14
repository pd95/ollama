#ifndef OLLAMA_MLX_MEDIA_MINIAUDIO_WRAPPER_H
#define OLLAMA_MLX_MEDIA_MINIAUDIO_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

typedef struct ollama_miniaudio_mp3_decoder ollama_miniaudio_mp3_decoder;

ollama_miniaudio_mp3_decoder* ollama_miniaudio_mp3_init(
    const void* data, size_t size, uint32_t* source_channels,
    uint32_t* source_rate);
int ollama_miniaudio_mp3_read(
    ollama_miniaudio_mp3_decoder* decoder, void* output,
    uint64_t frame_count, uint64_t* frames_read);
void ollama_miniaudio_mp3_uninit(ollama_miniaudio_mp3_decoder* decoder);

#endif

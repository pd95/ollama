#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_WAV
#define MA_NO_FLAC
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_GENERATION
#define MA_NO_CUSTOM
#define MA_NO_STDIO
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "miniaudio_wrapper.h"

#include <stdlib.h>

struct ollama_miniaudio_mp3_decoder {
    ma_decoder decoder;
};

ollama_miniaudio_mp3_decoder* ollama_miniaudio_mp3_init(
    const void* data, size_t size, uint32_t target_rate,
    uint32_t* source_channels, uint32_t* source_rate) {
    ollama_miniaudio_mp3_decoder* result;
    ma_decoder_config config;
    ma_format input_format;

    if (data == NULL || size == 0 || target_rate == 0 ||
        source_channels == NULL || source_rate == NULL) {
        return NULL;
    }
    result = (ollama_miniaudio_mp3_decoder*)calloc(1, sizeof(*result));
    if (result == NULL) {
        return NULL;
    }
    config = ma_decoder_config_init(ma_format_f32, 1, target_rate);
    config.encodingFormat = ma_encoding_format_mp3;
    if (ma_decoder_init_memory(data, size, &config, &result->decoder) != MA_SUCCESS) {
        free(result);
        return NULL;
    }
    if (result->decoder.pBackend == NULL ||
        ma_data_source_get_data_format(result->decoder.pBackend, &input_format,
            source_channels, source_rate, NULL, 0) != MA_SUCCESS) {
        ma_decoder_uninit(&result->decoder);
        free(result);
        return NULL;
    }
    return result;
}

int ollama_miniaudio_mp3_read(
    ollama_miniaudio_mp3_decoder* decoder, void* output,
    uint64_t frame_count, uint64_t* frames_read) {
    ma_result result;
    ma_uint64 decoded = 0;
    if (decoder == NULL || output == NULL || frames_read == NULL) {
        return -1;
    }
    result = ma_decoder_read_pcm_frames(&decoder->decoder, output, (ma_uint64)frame_count, &decoded);
    *frames_read = (uint64_t)decoded;
    return result == MA_SUCCESS || result == MA_AT_END ? 0 : -1;
}

void ollama_miniaudio_mp3_uninit(ollama_miniaudio_mp3_decoder* decoder) {
    if (decoder != NULL) {
        ma_decoder_uninit(&decoder->decoder);
        free(decoder);
    }
}

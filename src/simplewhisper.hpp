/*
 *  Copyright 2025 (C) Victor Hogeweij <Hoog-V>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * This file is part of the SimpleLLama library
 *
 * Author:          Victor Hogeweij <Hoog-V>
 *
 */

#ifndef SIMPLEWHISPER_HPP
#define SIMPLEWHISPER_HPP
#include <whisper.h>
#include <map>
#include <string>
#include <thread>
#include <vector>

/*
 * Settings for the model inference,
 * Such as usage of gpu, threads etc.
 */
struct simplewhisper_model_params
{
    /* How many threads to use ?, default is 4*/
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    /* Number of gpu-layers to export to gpu, default = all 
    *  More layers means more vram usage! But better performance
    */
    int32_t n_gpu_layers = 999;
    int32_t audio_ctx  = 0;
    int32_t max_tokens = 32;

    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;

    /* Whether or not to timestamp responses */
    bool no_timestamps  = true;
    /* Whether or not to give a verbose prompt */
    bool verbose_prompt = false;
    /* Whether or not to use gpu, would highly suggest to leave enabled */
    bool use_gpu = true;
    /* whether to use flash attention */
    bool flash_attn = false;

    std::string language    = "en";
    std::string model_wsp = "";
    std::string bot_name    = "LLaMA";
    /* Custom prompt for model init, if no prompt given, it will use default */
    std::string prompt = "";
    std::string path_session = ""; // path to file for saving/loading model eval state
};

class SimpleWhisper
{
public:
    /**
     * @brief Construct a new whisper wrapper object
     *
     * @param params A struct containing configuration for model inference; Runtime as well as behavioural settings
     */
    SimpleWhisper(simplewhisper_model_params params);

    /**
     * @brief Initialize whisper by initializing a talk session with a initializer response
     */
    void init();

    /**
     * @brief Run inference (this reuses the initialised session)
     *
     * @param input_text user_prompt, to react on
     * @return std::string The response from the llama model
     */
    std::string do_inference(std::vector<float> &audio_samples);

    /**
     * @brief Destroy the llama wrapper object
     *
     */
    ~SimpleWhisper()
    {
        whisper_print_timings(m_ctx_wsp);
        whisper_free(m_ctx_wsp);
    }

private:
int64_t m_t_ms = 0;                  ///< Inference timing in milliseconds.
float m_prob0 = 0.0f;                ///< Initial probability.
struct whisper_context_params m_cparams;
struct whisper_context * m_ctx_wsp;
simplewhisper_model_params m_params;
};



#endif /* SIMPLEWHISPER_HPP */
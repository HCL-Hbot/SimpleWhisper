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

#include "audio_async.hpp"
#include "simplewhisper.hpp"
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define AUDIO_DEFAULT_MIC -1
#define AUDIO_SAMPLE_RATE 16000

bool keepRunning = true;

// Signal handler for SIGINT
void signalHandler(int signum) { keepRunning = false; }

void high_pass_filter(std::vector<float> &data, float cutoff,
                      float sample_rate) {
  const float rc = 1.0f / (2.0f * M_PI * cutoff);
  const float dt = 1.0f / sample_rate;
  const float alpha = dt / (rc + dt);

  float y = data[0];

  for (size_t i = 1; i < data.size(); i++) {
    y = alpha * (y + data[i] - data[i - 1]);
    data[i] = y;
  }
}

bool vad_simple(std::vector<float> &pcmf32, int sample_rate, int last_ms,
                float vad_thold, float freq_thold, bool verbose) {
  const int n_samples = pcmf32.size();
  const int n_samples_last = (sample_rate * last_ms) / 1000;

  if (n_samples_last >= n_samples) {
    // not enough samples - assume no speech
    return false;
  }

  if (freq_thold > 0.0f) {
    high_pass_filter(pcmf32, freq_thold, sample_rate);
  }

  float energy_all = 0.0f;
  float energy_last = 0.0f;

  for (int i = 0; i < n_samples; i++) {
    energy_all += fabsf(pcmf32[i]);

    if (i >= n_samples - n_samples_last) {
      energy_last += fabsf(pcmf32[i]);
    }
  }

  energy_all /= n_samples;
  energy_last /= n_samples_last;

  if (verbose) {
    fprintf(
        stderr,
        "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n",
        __func__, energy_all, energy_last, vad_thold, freq_thold);
  }

  if (energy_last > vad_thold * energy_all) {
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  // Register signal handler
  std::signal(SIGINT, signalHandler);

  /* Create new model params struct with for the model settings */
  simplewhisper_model_params model_params;
  model_params.model_wsp =
      "ggml-large-v3-turbo-q5_0.bin"; /* Model name, downloaded automatically by
                                         cmake */

  std::shared_ptr<audio_async> audio = std::make_shared<audio_async>(20 * 1000);

  /* Make a new instance of SimpleLLama */
  SimpleWhisper sw(model_params);

  /* Initialize the model runtime */
  sw.init();

  /* Create audio buffer and reserve some space */
  std::vector<float> audio_buffer;
  audio_buffer.reserve(2560);

  /* Init the audio on default microphone (-1) and with sample_rate of 16KHz */
  audio->init(AUDIO_DEFAULT_MIC, AUDIO_SAMPLE_RATE);
  /* Start audio capture */
  audio->resume();
  /* Clear any samples in queue */
  audio->clear();
  float vad_thold = 0.6f;
  float freq_thold = 100.0f;
  bool print_energy = false;
  int32_t voice_ms = 10000;

  while (keepRunning) {
    /* Get 1-second of audio */
    audio->get(2000, audio_buffer);

    bool voice_activity_detected =
        ::vad_simple(audio_buffer, WHISPER_SAMPLE_RATE, 1250, vad_thold,
                     freq_thold, print_energy);
    if (voice_activity_detected) {
      std::cout << "Voice detected!" << '\n';
      audio->get(voice_ms, audio_buffer);
      std::string text_heard = sw.do_inference(audio_buffer);

      /* Was there any text returned from the whisper inference? If no text, we
       * obviously don't want to run llama on it!*/
      if (text_heard.empty()) {
        /* No words were captured! Continue with capturing new audio; */
        audio->clear();
        continue;
      }
      /* Print the text we got from whisper inference  */
      fprintf(stdout, "%s%s%s", "\033[1m", text_heard.c_str(), "\033[0m");
      fflush(stdout);
      audio->clear();
    }

    /* Sleep for 200 milliseconds, to let CPU relax for bit */
    /* Otherwise we will never reach our 2% CPU Utilization :) */
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return 0;
}
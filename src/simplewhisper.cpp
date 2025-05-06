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
#include "simplewhisper.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

const std::string k_prompt_whisper =
    R"(A conversation with a person called {1}.)";
std::string trim(const std::string &s) {
  std::regex e("^\\s+|\\s+$");
  return std::regex_replace(s, e, "");
}

static std::string transcribe(whisper_context *ctx,
                              const simplewhisper_model_params &params,
                              const std::vector<float> &pcmf32,
                              const std::string prompt_text, float &prob,
                              int64_t &t_ms) {
  const auto t_start = std::chrono::high_resolution_clock::now();

  prob = 0.0f;
  t_ms = 0;

  std::vector<whisper_token> prompt_tokens;

  whisper_full_params wparams =
      whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

  prompt_tokens.resize(1024);
  prompt_tokens.resize(whisper_tokenize(
      ctx, prompt_text.c_str(), prompt_tokens.data(), prompt_tokens.size()));

  wparams.print_progress = false;
  wparams.print_special = params.print_special;
  wparams.print_realtime = false;
  wparams.print_timestamps = !params.no_timestamps;
  wparams.translate = params.translate;
  wparams.no_context = true;
  wparams.single_segment = true;
  wparams.max_tokens = params.max_tokens;
  wparams.language = params.language.c_str();
  wparams.n_threads = params.n_threads;

  wparams.prompt_tokens =
      prompt_tokens.empty() ? nullptr : prompt_tokens.data();
  wparams.prompt_n_tokens = prompt_tokens.empty() ? 0 : prompt_tokens.size();

  wparams.audio_ctx = params.audio_ctx;

  if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
    return "";
  }

  int prob_n = 0;
  std::string result;

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char *text = whisper_full_get_segment_text(ctx, i);

    result += text;

    const int n_tokens = whisper_full_n_tokens(ctx, i);
    for (int j = 0; j < n_tokens; ++j) {
      const auto token = whisper_full_get_token_data(ctx, i, j);

      prob += token.p;
      ++prob_n;
    }
  }

  if (prob_n > 0) {
    prob /= prob_n;
  }

  const auto t_end = std::chrono::high_resolution_clock::now();
  t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
             .count();

  return result;
}

static std::vector<std::string> get_words(const std::string &txt) {
  std::vector<std::string> words;

  std::istringstream iss(txt);
  std::string word;
  while (iss >> word) {
    words.push_back(word);
  }

  return words;
}

std::string replace(const std::string &s, const std::string &from,
                    const std::string &to) {
  std::string result = s;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.length(), to);
    pos += to.length();
  }
  return result;
}

SimpleWhisper::SimpleWhisper(simplewhisper_model_params params) {
  m_params = params;
  m_cparams = whisper_context_default_params();
  m_cparams.use_gpu = params.use_gpu;
  m_cparams.flash_attn = params.flash_attn;

  m_ctx_wsp =
      whisper_init_from_file_with_params(params.model_wsp.c_str(), m_cparams);
  if (!m_ctx_wsp) {
    fprintf(stderr, "No whisper.cpp model specified. Please provide using -mw "
                    "<modelfile>\n");
    return;
  }
}

void SimpleWhisper::init() {

  const std::string prompt_whisper =
      ::replace(k_prompt_whisper, "{1}", m_params.bot_name);
}

std::string SimpleWhisper::do_inference(std::vector<float> &audio_samples) {
  std::string all_heard;
  all_heard = ::trim(::transcribe(m_ctx_wsp, m_params, audio_samples,
                                  k_prompt_whisper, m_prob0, m_t_ms));
  const auto words = get_words(all_heard);

  std::string wake_cmd_heard;
  std::string text_heard;
  for (int i = 0; i < (int)words.size(); ++i) {
    if (i < 1) {
      wake_cmd_heard += words[i] + " ";
    } else {
      text_heard += words[i] + " ";
    }
  }

  // remove text between brackets using regex
  {
    std::regex re("\\[.*?\\]");
    text_heard = std::regex_replace(text_heard, re, "");
  }

  // remove text between brackets using regex
  {
    std::regex re("\\(.*?\\)");
    text_heard = std::regex_replace(text_heard, re, "");
  }

  // remove all characters, except for letters, numbers, punctuation and ':',
  // '\'', '-', ' '
  text_heard = std::regex_replace(
      text_heard, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), "");

  // take first line
  text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));

  // remove leading and trailing whitespace
  text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
  text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");

  text_heard.insert(0, 1, ' ');
  text_heard += "\n" + std::string("LLama") + ":";
  fprintf(stdout, "%s%s%s", "\033[1m", text_heard.c_str(), "\033[0m");
  fflush(stdout);

  return text_heard;
}
